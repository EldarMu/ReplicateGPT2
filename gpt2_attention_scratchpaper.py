"""
My own experiment space, this experiment is me trying various ways to implement "scratchpaper attention"
which is a residual hooked up between just the attention blocks that influences the attention but is per-head not per-token.
like notes for heads to pass forward.
"""

from collections import defaultdict
from dataclasses import dataclass
import math
import tiktoken
import time
import torch
import torch.nn as nn
from torch.nn import functional as torchF
from transformers import GPT2LMHeadModel

@dataclass
class GPT2Parameters:
  max_seq_len: int = 1024
  vocab_size: int = 50257
  num_decoders: int = 12
  num_heads: int = 12
  hidden_dim: int = 768
  scratch_enabled: bool = True
  scratch_dim_per_head: int = 64


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).rsqrt()
        return x * rms * self.gamma


class DecayWriter(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_heads = config.num_heads
        scratch_dim = config.scratch_dim_per_head

        # RMSNorm to stabilize the input from the attention block
        self.norm = RMSNorm(scratch_dim)

        # Learnable, per-head gates. Parameterized by logits.
        # Initialize keep_logit high, so sigmoid(keep_logit) starts near 1.
        self.keep_logit = nn.Parameter(torch.full((num_heads, 1), 3.0)) # sigmoid(5.0) is ~0.953
        # Initialize add_logit low, so sigmoid(add_logit) starts near 0.
        self.add_logit = nn.Parameter(torch.full((num_heads, 1), -2.0)) # sigmoid(-2.0) is ~0.12

    def forward(self, scratch_in, scratch_mid):
        # scratch_in:  [B, Nh, T, Ds] - the state from the previous block
        # scratch_mid: [B, Nh, T, Ds] - the new information from this block's attention

        # Constrain gates to be between 0 and 1
        keep_gate = torch.sigmoid(self.keep_logit)
        add_gate = torch.sigmoid(self.add_logit)

        # Reshape gates from [Nh, 1] to [1, Nh, 1, 1] to broadcast correctly
        keep_gate = keep_gate.view(1, -1, 1, 1)
        add_gate = add_gate.view(1, -1, 1, 1)

        # Stabilize the new information
        normalized_scratch_mid = self.norm(scratch_mid)

        # The state update rule: a gated Exponential Moving Average (EMA)
        # S_out = keep * S_in + add * new_info
        scratch_out = keep_gate * scratch_in + add_gate * normalized_scratch_mid
        
        return scratch_out

class CausalSelfAttention(nn.Module):

  def __init__(self, config):
    super().__init__()
    assert config.hidden_dim % config.num_heads == 0
    self.qkv_projection = nn.Linear(
      in_features=config.hidden_dim,
      out_features=3*config.hidden_dim
    )
    self.out_projection = nn.Linear(
      in_features=config.hidden_dim,
      out_features=config.hidden_dim
    )
    self.out_projection.NANOGPT_SCALE_INIT = 1
    self.num_heads = config.num_heads
    self.hidden_dim = config.hidden_dim

    self.scratch_enabled = config.scratch_enabled
    if self.scratch_enabled:
        self.scratch_dim_per_head = config.scratch_dim_per_head
        Ds = self.scratch_dim_per_head
        Nh = self.num_heads
        self.Wq_s = nn.Parameter(torch.randn(Nh, Ds, Ds) * 0.02)
        self.Wk_s = nn.Parameter(torch.randn(Nh, Ds, Ds) * 0.02)
        self.Wv_s = nn.Parameter(torch.randn(Nh, Ds, Ds) * 0.02)
        self.register_buffer('lambda_s', torch.tensor(0.0))
        # no fucking idea the name is shit but I think AI wanted this to be a per-head scratch gate?
        # absolute shit name if that's the case, fucking asshole thinks he's writing a python notebook
        # so no need to use even slightly reasonable names.
        self.per_head_scratch_scl = nn.Parameter(torch.zeros(self.num_heads))

    self.register_buffer(
      name='bias',
      tensor=torch.tril(
        torch.ones(
          size=(config.max_seq_len, config.max_seq_len)
        ).view(1, 1, config.max_seq_len, config.max_seq_len)
      )
    )

  
  def forward(self, data, scratch_in=None):
    use_scratch = self.scratch_enabled and (scratch_in is not None)
    if not use_scratch:
      batch_size, tokens, hidden_dim = data.size()
      qkv = self.qkv_projection(data)
      q,k,v = qkv.split(self.hidden_dim, dim=2)
      q = q.view(batch_size, tokens, self.num_heads,
                self.hidden_dim // self.num_heads).transpose(1, 2)
      k = k.view(batch_size, tokens, self.num_heads,
                self.hidden_dim // self.num_heads).transpose(1, 2)
      v = v.view(batch_size, tokens, self.num_heads,
                self.hidden_dim // self.num_heads).transpose(1, 2)
      out_data = torchF.scaled_dot_product_attention(q, k, v, is_causal=True)
      out_data = out_data.transpose(1, 2).contiguous().view(
        batch_size, tokens, hidden_dim
      )
      out_data = self.out_projection(out_data)

      batch_size, tokens, _ = data.size()
      zero_scratch = torch.zeros(batch_size, self.num_heads, tokens,
        getattr(self, "scratch_dim_per_head", 1), device=data.device, dtype=data.dtype)
      return out_data, zero_scratch, {}
    else:
      batch_size, tokens, hidden_dim = data.size()
      dim_per_head = hidden_dim // self.num_heads
      scratch_dim = self.scratch_dim_per_head

      # Get projections same as before
      qkv = self.qkv_projection(data)
      q, k, v = qkv.split(self.hidden_dim, dim=2)
      Qc = q.view(batch_size, tokens, self.num_heads,
                self.hidden_dim // self.num_heads).transpose(1, 2)
      Kc = k.view(batch_size, tokens, self.num_heads,
                self.hidden_dim // self.num_heads).transpose(1, 2)
      Vc = v.view(batch_size, tokens, self.num_heads,
                self.hidden_dim // self.num_heads).transpose(1, 2)

      # Get Scratch Projections
      # scratch_in has shape [batch_size, num_heads, tokens, scratch_dim]
      # W*_s have shape [num_heads, scratch_dim, scratch_dim]
      # 'bhts,hsd->bhtd' is einsum notation for a batched matrix multiply per head.
      # It means: for each batch(b) and head(h), multiply the token matrix [T, Ds]
      # with that head's specific weight matrix [Ds, Ds] to get a result [T, Ds].
      Qs = torch.einsum('bhts,hsd->bhtd', scratch_in, self.Wq_s)
      Ks = torch.einsum('bhts,hsd->bhtd', scratch_in, self.Wk_s)
      Vs = torch.einsum('bhts,hsd->bhtd', scratch_in, self.Wv_s)

      # Manually Compute Dual Logits & Combine
      # Content logits: shape [batch_size, num_heads, tokens, tokens]
      logits_c = torch.matmul(Qc, Kc.transpose(-2, -1)) / math.sqrt(dim_per_head)

      # Scratch logits: shape [batch_size, num_heads, tokens, tokens]
      logits_s = torch.matmul(Qs, Ks.transpose(-2, -1)) / math.sqrt(scratch_dim)
      
      # tensor-gated lambda; avoids branching & graph breaks
      lambda_gate = (self.lambda_s > 1e-8).to(logits_c.dtype).view(1,1,1,1)
      # per-head scalar value for models to learn so variable scratch usage.
      head_scl = torch.exp(self.per_head_scratch_scl).view(1, -1, 1, 1)
      # Combine the logits, applying the lambda_s scaling factor
      total_logits = logits_c + (lambda_gate * self.lambda_s).view(1,1,1,1) * head_scl * logits_s
      # if I get rid of head scaling can use like:
      # total_logits = logits_c + (lambda_gate * self.lambda_s).view(1,1,1,1) * logits_s
      
      # The stored bias has shape [1, 1, max_seq_len, max_seq_len]
      scratch_delta = (lambda_gate * self.lambda_s).view(1,1,1,1) * head_scl * logits_s
      unmasked_logits = logits_c + scratch_delta

      # get diagnostic vals before -inf masking
      delta_logits_rms = scratch_delta.pow(2).mean().sqrt().detach()
      row_std_ratio = (logits_s.std(dim=-1) / (logits_c.std(dim=-1) + 1e-6)).mean().detach()      
      
      # mask for attention
      total_logits = unmasked_logits.masked_fill(self.bias[:, :, :tokens, :tokens] == 0, float('-inf'))

      # Get attention weights from the combined scores
      attention_weights = torchF.softmax(total_logits, dim=-1)

      # Mix content values Vc using the combined weights
      content_out = torch.matmul(attention_weights, Vc)
      
      # Mix scratch values Vs using the same combined weights
      scratch_out_mid = torch.matmul(attention_weights, Vs)
      
      # Reshape content_out back to standard tensor format for the residual stream
      attn_out = content_out.transpose(1, 2).contiguous().view(batch_size, tokens, hidden_dim)
      attn_out = self.out_projection(attn_out)

      # collect some data on our new layers in lieue of visualizing.
      diags = {
          'delta_logits_rms': delta_logits_rms,
          'row_std_ratio': row_std_ratio,
      }
      return attn_out, scratch_out_mid, diags

class MLP(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.ffn1 = nn.Linear(
      in_features=config.hidden_dim,
      out_features=4*config.hidden_dim
    )
    self.activation = nn.GELU()
    self.ffn2 = nn.Linear(
      in_features=4*config.hidden_dim,
      out_features=config.hidden_dim
    )
    self.ffn2.NANOGPT_SCALE_INIT = 1

  def forward(self, data):
    data = self.ffn1(data)
    data = self.activation(data)
    data = self.ffn2(data)
    return data

class Block(nn.Module):

    def __init__(self, config):
      super().__init__()
      self.ln_before_att = nn.LayerNorm(config.hidden_dim)
      self.attention = CausalSelfAttention(config)
      self.ln_before_fnn = nn.LayerNorm(config.hidden_dim)
      self.mlp = MLP(config)

      if config.scratch_enabled:
        self.writer = DecayWriter(config)
        self.S_bias = nn.Parameter(torch.randn(config.num_heads, config.scratch_dim_per_head) * 0.02)
    
    def forward(self, data, scratch_in=None):
      if hasattr(self, 'writer') and scratch_in is None:
          # Initialize scratch state to zeros + bias
          batch_size, tokens, _ = data.size()
          with torch.no_grad():
              x = self.ln_before_att(data)
              x_per_head = x.view(batch_size, tokens, self.attention.num_heads, -1).transpose(1, 2)
              # Take the first scratch_dim features from each head's content vector.
              scratch_in = x_per_head[..., :self.attention.scratch_dim_per_head].contiguous()

      attn_output, scratch_mid, diags = self.attention(self.ln_before_att(data), scratch_in)
      data = data + attn_output
      
      scratch_out = None
      if hasattr(self, 'writer'):
          scratch_out = self.writer(scratch_in, scratch_mid)

      data = data + self.mlp(self.ln_before_fnn(data))

      if scratch_out is not None:
          diags['s_rms_out'] = (scratch_out.pow(2).mean().sqrt()).detach()

      return data, scratch_out, diags

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
        dict(
          embedding_matrix = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_dim
          ),
          pos_embed=nn.Embedding(
            num_embeddings=config.max_seq_len,
            embedding_dim=config.hidden_dim
          ),
          heads=nn.ModuleList([
            Block(config) for _ in range(config.num_decoders)
          ]),
          final_layernorm=nn.LayerNorm(config.hidden_dim)
        ))
        self.lm_head = nn.Linear(
          in_features=config.hidden_dim,
          out_features=config.vocab_size,
          bias=False)
        self.lm_head.weight = self.transformer.embedding_matrix.weight

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.num_decoders) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, predictions=None):
      batch_size, seq_len = idx.size()
      assert seq_len <= self.config.max_seq_len, f"cannot use sequence length {seq_len} > {self.config.max_seq_len}"
      pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
      pos_embed = self.transformer.pos_embed(pos)
      token_embed = self.transformer.embedding_matrix(idx)
      data = token_embed + pos_embed
      scratch = None
      all_diags = []
      for decoder in self.transformer.heads:
        data, scratch, diags = decoder(data, scratch)
        if diags:
          all_diags.append(diags)
        # Not intending to use last scratch state for anything.

      data = self.transformer.final_layernorm(data)
      logits = self.lm_head(data)
      loss = None
      if predictions is not None:
        loss = torchF.cross_entropy(logits.view(-1, logits.size(-1)), predictions.view(-1))
      
      return logits, loss, all_diags

def train_model():
  device = "cpu"
  if torch.cuda.is_available():
      device = "cuda"
      torch.cuda.manual_seed(42)
  print(f"using device: {device}")

  torch.set_float32_matmul_precision('high')
  token_length = 256
  batch_size = 16
  train_loader = train_gpt2.DataLoaderLite(batch_size, token_length)

  model = train_gpt2.GPT(train_gpt2.GPT2Parameters())
  model.to(device)
  model = torch.compile(model)
  optimizer = train_gpt2.AdamW_Impl(model.parameters(), device=device)
  max_steps = 1000
  print("step|loss|λ|keep|add|s_rms|ratio|W_norm|grad_norm|delta")
  for i in range(max_steps):
    # time per-step.
    t0 = time.time()
    data, labels = train_loader.next_batch()
    # load to device (GPU memory if available)
    data, labels = data.to(device), labels.to(device)
    # Set the lambda_s schedule for the scratchpad. Needs to be done every step.
    current_lambda = train_gpt2.get_lambda_s(i, warmup_steps=100)
    # We need to set this value on every CausalSelfAttention module
    # Because the model is compiled, we access the original via _orig_mod
    for block in model._orig_mod.transformer.heads:
        block.attention.lambda_s.fill_(current_lambda)

    optimizer.zero_grad()
    logits, loss, all_diags = model(data, labels)  

    loss.backward()
    # needs adjusting since doesn't scale
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = train_gpt2.get_lr(i)
    # since I implemented my own optimizer, there's no param_groups
    # and I'm going to move it to device to match everything else.
    optimizer.lr = torch.tensor(lr).to(device)
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000
    tokens_per_sec = (train_loader.batch_size *
                      train_loader.token_length) / (t1 - t0)
    # Log every 10 steps
    if i % 10 == 0:
        # Calculate average gate values across all layers
        avg_keep_gates = []
        avg_add_gates = []
        avg_delta_logits_rms = []
        avg_row_std_ratio = []
        avg_s_rms = []
        avg_W_norms = []
        # Use torch.no_grad to avoid tracking these operations for gradients
        with torch.no_grad():
            for block in model._orig_mod.transformer.heads:
                if hasattr(block, 'writer'):
                    avg_keep_gates.append(torch.sigmoid(block.writer.keep_logit).mean().item())
                    avg_add_gates.append(torch.sigmoid(block.writer.add_logit).mean().item())
                    avg_W_norms.append(block.attention.Wq_s.norm().item())
                    avg_W_norms.append(block.attention.Wk_s.norm().item())
                    avg_W_norms.append(block.attention.Wv_s.norm().item())
            if all_diags:
                for diags in all_diags:
                    avg_row_std_ratio.append(diags['row_std_ratio'].item())
                    if 's_rms_out' in diags:
                      avg_s_rms.append(diags['s_rms_out'].item())
                    if 'delta_logits_rms' in diags:
                      avg_delta_logits_rms.append(diags['delta_logits_rms'].item())

        # Avoid division by zero if no writers found
        avg_keep = sum(avg_keep_gates) / len(avg_keep_gates) if avg_keep_gates else 0
        avg_add = sum(avg_add_gates) / len(avg_add_gates) if avg_add_gates else 0
        row_std_ratio = sum(avg_row_std_ratio) / len(avg_row_std_ratio) if avg_row_std_ratio else 0
        delta_logits_rms = sum(avg_delta_logits_rms)/len(avg_delta_logits_rms) if avg_delta_logits_rms else 0
        s_rms = sum(avg_s_rms) / len(avg_s_rms) if avg_s_rms else 0
        W_norm = sum(avg_W_norms) / len(avg_W_norms) if avg_W_norms else 0

        print(
            f"{i}|{loss.item():.4f}|{current_lambda:.2f}|"
            f"{avg_keep:.3f}|{avg_add:.3f}|{s_rms:.2f}|"
            f"{row_std_ratio:.3f}|{W_norm:.2f}|{norm:.2f}|{delta_logits_rms:.2f}"
        )


def get_lambda_s(it, warmup_steps=50, max_lambda=1.0):
    if it < warmup_steps:
        # Linear warmup from 0.0 to max_lambda
        return max_lambda * (it + 1) / warmup_steps
    return max_lambda

def get_lr(it):
  max_lr = 6e-4
  min_lr = max_lr * 0.1
  warmup_steps = 20
  max_steps = 80
  # scale to max value linearly for warmup steps
  if it < warmup_steps:
      return max_lr * (it+1) / warmup_steps
  # If we're past max_steps clamp down to min_lr
  if it > max_steps:
      return min_lr
  # In between, use cosine decay down to min learning rate
  # That means smooth non-linear motion down.
  decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
  assert 0 <= decay_ratio <= 1
   # coeff starts at 1 and goes to 0
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
  return min_lr + coeff * (max_lr - min_lr)

class AdamW_Impl:
    def __init__(self, params, lr=3e-4, device='cpu'):
        """
        Very basic AdamW implementation.
        Tailored to this specific model, for educational purposes.
        Also I hardcoded values usually passed into constructor so I don't have to
        write all kinds of "raise ValueError" checks.

        AdamW formula is:
        t = our step (iteration) value. We increment this each time we perform a step,
        so for this implementation, it's once per epoch (since we're doing full-batch).
        Starts at 1, not 0.

        g(t) = gradient at iteration t.
        (we get the gradients from loss.backwards() auto-populated in params for us)

        Weight adjustment, for the stored weights from previous step.
        For us this means subtract lr-adjusted 1% (1e-2) of the existing weight.
        w(t) = w(t-1) - lr * weight_decay * w(t-1)
        
        first moment:
        m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
        For our implementation, this means 90% of previous step's smoothed gradients, 10% of current.
        Kind of like EMA, for gradient smoothing.

        second moment:
        v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
        Like EMA of the scale of the gradients.
        This ends up in the divisor so it slows down lr when gradients are large and vice-versa.

        m(t) adjustment:
        since we start with zeros, the value would increase super slow, so we adjust it by this,
        and since the beta is 0.9, first step is m(t)/0.1 then /0.19, then /0.27 and so on.
        m_hat = m(t) / (1 - beta1^t)

        v(t) adjustment, same idea, but /0.01, then /0.019, then /0.03 and so on.
        v_hat = v(t) / (1 - beta2^t)

        Without the betas/epsilon the final formula would be:
        w(t) = w(t-1) - lr * g(t)/(sqrt(g(t)^2)) so like lr with gradient sign retention.
        (just showing so it's easier for you to connect all the dots)

        finally, we update the weights:
        v_hat can easily underflow, so we add a tiny number to it.
        w(t) = w(t-1) - lr * m_hat / (sqrt(v_hat) + eps)

        so the weight gets updated by the smoothed gradient divided by the
        smoothed scale of the gradients, multiplied by the learning rate.
        """
        # Learning rate
        # can be set externally
        self.lr = torch.tensor(lr).to(device)          
        # Beta coefficients for Adam
        # beta1 for the first moment (gradient smoothing)
        # beta2 for the second moment (gradient scaling)
        # beta2 updated from default in pytorch (0.999) to gpt2 value (0.995)  
        self.betas = (torch.tensor(0.9).to(device), torch.tensor(0.995).to(device))
        # Tiny number to prevent division by 0
        self.eps = torch.tensor(1e-8).to(device)
        # How much to decay weights by on each step (1%*lr)
        self.weight_decay = torch.tensor(1e-2).to(device)

        # Model weights and biases and some non-learnable params.
        # Provided as a generator, turned into list here since
        # generators are intended for one run and we might iterate more than once.
        self.parameters = list(params)
        # defaultdict makes it so instead of KeyError you get a default value.
        self.state = defaultdict(dict)

        # add step, first moment, second moment vars for each tensor.
        for p in self.parameters:
            if p.requires_grad:
                # step gets set to 0 here because it gets incremeneted first thing in step() 
                self.state[p]['step'] = torch.tensor(0).to(device)
                # initialize with same-shaped zeros tensors
                # preserve_format = try to make the memory layout match the params layout
                # so the gpu can do operations involving both faster.
                self.state[p]['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format).to(device)
                self.state[p]['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format).to(device)

    def zero_grad(self):
        """Reset gradients to zero for all weights.
           Don't want to pollute between epochs."""
        for p in self.parameters:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        """Performs a single optimization step."""
        for p in self.parameters:
            # skip non-learnable params
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[p]

            state['step'] += 1
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            beta1, beta2 = self.betas
            lr = self.lr

            # Weight decay applied first like in pytorch's single_tensor impl.
            # Weight decay: w(t) = w(t-1) - lr * weight_decay * w(t-1)
            # mul_ = in-place element-wise multiplication of the tensor
            #
            # updated - now only happens for parameters that have more than 2 dims
            # unlike Karpathy's code, this is in the optimizer itself since I rolled my own
            # p.requires_grad is baked in since we check if p.grad is None above.
            if (p.dim() >= 2):
              p.data.mul_(1 - lr * self.weight_decay)
                

            # m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
            # lerp_ = in-place per-element weighted average of two tensors (grad and exp_avg)
            exp_avg.lerp_(grad, 1 - beta1)

            # v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
            # mul_ = multiply v(t-1) by beta2 in-place
            # addcmul_ = in-place element-wise multiply grad by grad, then by (1-beta2),
            # then add to v(t-1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            # m_hat and v_hat divisor calculation
            step = state['step']
            # 1 - beta1^t
            bias_correction1 = torch.tensor(1) - torch.pow(beta1, state['step'])
            # 1 - beta2^t
            bias_correction2 = torch.tensor(1) - torch.pow(beta2, state['step'])

            # Final update denominator,
            # exp_avg_sq.div_(bias_correction2) = v_t / (1 - beta2^t) = v_hat
            # and then the whole thing is the (sqrt(v_hat) + eps)
            #
            # previously did this as one step like:
            # denom = torch.sqrt(exp_avg_sq.div_(bias_correction2)).add_(self.eps)
            # but kept getting infs/nans.
            # now doing square rooting here, which works because
            # sqrt(x)/sqrt(y) = sqrt(x/y)
            bias_correction2_sqrt = torch.sqrt(bias_correction2)
            denom = exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(self.eps)

            # Update the weights
            # w(t) = w(t-1) - lr * m_hat / (sqrt(v_hat) + eps)
            # since we didn't calculate m_hat separately, and m_hat is m(t) / (1 - beta1^t)
            # we just divide lr by (1-beta1^t) so we can use m(t) as-is.
            step_size = lr / bias_correction1
            # param_update = m(t) / (sqrt(v_hat) + eps)
            param_update = exp_avg / denom
            # alpha is just a multiplier for the values, so this is equivalent to
            # w(t-1) = w(t-1) + (m(t) / (sqrt(v_hat) + eps))*  -(lr / (1-beta1^t))
            # which is equivalent to w(t-1) - lr * m_hat / (sqrt(v_hat) + eps)
            p.data.add_(param_update, alpha=-step_size)

# need to do data, labels = data.to_device(device), labels.to_device(device)
class DataLoaderLite:
    def __init__(self, batch_size, token_length, debug_print=False):
        self.batch_size = batch_size
        self.token_length = token_length
        #!wget -nc https://raw.githubusercontent.com/karpathy/char-rnn/refs/heads/master/data/tinyshakespeare/input.txt
        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        if debug_print:
          print(f"loaded {len(self.tokens)} tokens")
          print(f"1 epoch = {len(self.tokens) // (batch_size * token_length)} batches")
        # index for text, moves up as we use chunks of text
        self.current_position = 0

    def next_batch(self):
        batch_size, token_length = self.batch_size, self.token_length
        buf = self.tokens[self.current_position : 
                          self.current_position+batch_size*token_length+1]
        data = (buf[:-1]).view(batch_size, token_length)
        labels = (buf[1:]).view(batch_size, token_length)
        # advance the position in the tensor
        self.current_position += batch_size * token_length
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (batch_size * token_length
                                    + 1) > len(self.tokens):
            self.current_position = 0
        return data, labels