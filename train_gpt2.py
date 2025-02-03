"""
Code is from Andrej Karpathy's 'Let's reproduce GPT-2 (124M)'
https://www.youtube.com/watch?v=l8pRSuU81PU
I have modified names in many places where I felt like it helped me
to understand it easier, and provided various comments.
This is meant to be a base from which to start adding various papers'
improvements.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as torchF
from transformers import GPT2LMHeadModel
import tiktoken

# store all the parameters up top for easy modification.
@dataclass
class GPT2Parameters:
  max_seq_len: int = 1024
  # In more complex models (not this one), if we have fewer tokens, we pad with 
  # special <pad> tokens, and have an extra ignore_pad_tokens mask in attention.
  vocab_size: int = 50257
  # number of decoders stacked in model
  num_decoders: int = 12
  # number of heads attention is split between
  num_heads: int = 12
  # token embedding dimension, must be evenly divisible by num_heads,
  # by default also FFN width BUT that's usually 4x hidden_dim in newer models.
  hidden_dim: int = 768

class CausalSelfAttention(nn.Module):

  def __init__(self, config):
    super().__init__()
    # hidden_dim (per-token features / embedding dimension) must be
    # evenly splittable between the heads
    assert config.hidden_dim % config.num_heads == 0
    # this might seem weird, basically since we're going to need to
    # generate the 3 matrices (Q, K, V) from the input, the way it's done
    # here is to matmul to expand the hidden dimension to 3x and then 'split' it
    # The choice is made for hardware efficiency.
    # Libraries/hardware are very optimized for big matmul operations.
    self.qkv_projection = nn.Linear(
      in_features=config.hidden_dim,
      out_features=3*config.hidden_dim
    )
    # final matrix to allow mix-and-match between attention heads
    self.out_projection = nn.Linear(
      in_features=config.hidden_dim,
      out_features=config.hidden_dim
    )
    # storing these values for reference in forward()
    self.num_heads = config.num_heads
    self.hidden_dim = config.hidden_dim
    # we're creating the attention mask here as part of the model's parameters.
    # It's a diagonal matrix with everything above (0, 0) to (n, n) as zeroes.
    self.register_buffer(
      name='bias',
      # make the upper triangular corner of the matrix into zeros,
      # diagonal controls where the triangular starts, 0 means normal,
      # -1 will erase (0, 0), 1 will leave (0, 1) untouched too.
      tensor=torch.tril(
        # we make the mask here, tiny example:
        # [1, 1] -> triangular(diagonal=0) -> [1, 0]
        # [1, 1]                              [1, 1]
        # real dims are (max_seq_len, max_seq_len)
        torch.ones(
          size=(config.max_seq_len, config.max_seq_len)
          # made (max_seq_len, max_seq_len)
          # but we also need batch size, heads, just 1 dim per.
        ).view(1, 1, config.max_seq_len, config.max_seq_len)
      )
      # and we're done, we've made a diagonal matrix of size
      # (max_seq_len, max_seq_len) where the upper triangle is zeroed out.
    )

  
  def forward(self, data):
    # explicitly list the dimensions of the tensor.
    batch_size, tokens, hidden_dim = data.size()
    qkv = self.qkv_projection(data)
    # the second dimension is hidden_dim, which we expanded to 3x its size
    # and now we 'split' along that into hidden_dim sized chunks.
    # https://pytorch.org/docs/stable/generated/torch.split.html#torch.split
    # each chunk is a *view* into the original tensor.
    # What *view* means is that you're making new tensors but they're still
    # pointing to the same underlying memory, so no new data is being written,
    # it's just a way to have the model operate on different parts of that data
    # as if the data really was split. This plays nice with the hardware.
    q,k,v = qkv.split(self.hidden_dim, dim=2)
    # For each of the below, we start off with a matrix like (B, T, hidden)
    # and we're going to split it among the heads, and we don't actually
    # want to split and re-concatenate matrices, because that's way more work
    # for the GPU. BUT - we can instead make a 'view' with a different number
    # of dimensions, and all we're really doing there is changing strides.
    #
    # ignoring batch size dim here's what's happening (wall of text incoming)
    # underlying memory layout: [0, 1, 2, ... 11] (contiguous)
    #
    # matrix version: [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]] = (3, 4)
    # so three tokens, and 4 features per token. 
    # Stride is (4, 1),
    # we need to move forward 4 elements to get to the next token's index,
    # but only 1 element to get to the next feature for a token.
    #
    # Now we make a (T, num_heads (2), features_per_head) view into the matrix.
    # [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]] (3, 2, 2)
    # Strides are (4, 2, 1). Matmul needs dims next to each other,
    # and since right now it's (T, H, D) and attention needs the token dim,
    # we have to swap those two dimensions, in the normal version that's dims
    # 1 and 2 but in this batch-free version it's 0 and 1. 
    #
    # so we transpose, and we get:
    # [[[0, 1], [4, 5], [8, 9]], [[2, 3], [6, 7], [10, 11]]]
    # so the tokens seem all out of order, but the heads are now combined.
    # dim is (2, 3, 2), stride is (2, 4, 1) - move 2 forward for next head,
    # move 4 forward for next token, move 1 forward for next feature.
    #
    # Remember, the stride movement is still along the "flat memory"
    # and that hasn't changed.
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # Nothing we've done here has forced us to move data in the underlying array
    # so the underlying memory is continuous, but the tensor is no longer
    # contiguous, since its ordering requires jumps (doesn't match flat memory)
    #
    # contiguous memory is required by some operations, it can be restored
    # by transposing back along same dims if the in-between operations were
    # all in-place and don't require contiguity, so be careful.
    q = q.view(batch_size, tokens, self.num_heads,
              self.hidden_dim // self.num_heads).transpose(1, 2)
    k = k.view(batch_size, tokens, self.num_heads,
              self.hidden_dim // self.num_heads).transpose(1, 2)
    v = v.view(batch_size, tokens, self.num_heads,
              self.hidden_dim // self.num_heads).transpose(1, 2)
    # the attention operation, note division by sqrt(hidden_dim_per_head)
    # have to specify which dims get transposed, using pythonic notation
    # to refer to last two dims here, could've also said (2, 3),
    # same for k.size(-1) it means the last dim's length.
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    # mask_fill is looking at the registered buffer bias and wherever it's
    # 0 it fills in the value -inf which works because e^(-inf) == 0.
    att = att.masked_fill(self.bias[:,:,:tokens,:tokens] == 0, float('-inf'))
    # recall Q@KT is going to look like:
    # [
    #  [q1k1, q1k2, q1k3, ...],
    #  [q2k1, q2k2, q2k3, ...],
    #  ...
    #]
    # so when we do it along the last dim (innermost) we're making sure that
    # the softmax normalizes per-query for every key value.
    att = torchF.softmax(
      input=att,
      dim=-1
    )
    # (max_seq_len x max_seq_len) @ (max_seq_len x hidden_dim)
    # result is (max_seq_len x hidden_dim) so we're back to Q, K, V dims
    out_data = att @ v
    # transpose back, flip head and token dims.
    # then make contiguous because some of the operations
    # likely made copies and lost the stride info we had earlier.
    # then, get rid of the num_heads dimension to get original dims back.
    out_data = out_data.transpose(1, 2).contiguous().view(
      batch_size, tokens, hidden_dim
    )
    # one last linear projection befor output.
    out_data = self.out_projection(out_data)
    return out_data

class MLP(nn.Module):

  def __init__(self, config):
    super().__init__()
    # dimensions expanded to 4x hidden dim normally.
    self.ffn1 = nn.Linear(
      in_features=config.hidden_dim,
      out_features=4*config.hidden_dim
    )
    # original version approximated with tanh, not necessary anymore
    # the regular version is properly optimized now.
    self.activation = nn.GELU()
    # output back to hidden_dim
    self.ffn2 = nn.Linear(
      in_features=4*config.hidden_dim,
      out_features=config.hidden_dim
    )

  def forward(self, data):
    data = self.ffn1(data)
    data = self.activation(data)
    data = self.ffn2(data)
    return data

class Block(nn.Module):

    def __init__(self, config):
      super().__init__()
      # Pre-LayerNorm configuration: normalization applied BEFORE attention/FFN
      self.ln_before_att = nn.LayerNorm(config.hidden_dim)
      self.attention = CausalSelfAttention(config)
      self.ln_before_fnn = nn.LayerNorm(config.hidden_dim)
      # Feed-forward network == Multi-Layer Perceptron.
      self.mlp = MLP(config)
    
    def forward(self, data):
      # residual allows for easier gradient flow, which makes the
      # optimization task easier for the model.
      data = data + self.attention(self.ln_before_att(data))
      # residual after FFN.
      data = data + self.mlp(self.ln_before_fnn(data))
      return data 
      

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        # Your input is (max_seq_len x vocab_size)
        self.transformer = nn.ModuleDict(
        dict(
          # A learned weight matrix where we look up "max_seq_len" rows 
          # in a matrix of size (vocab_size, hidden_dim) based on the
          # int indices provided by the input matrix/list.
          embedding_matrix = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_dim
          ),
          # In this super early version position was also a learned matrix.
          # That got replaced with sinusoidal pos encoding, then RoPE.
          # (as I recall this gets added to your embedded input,
          # which is why it matches the dims (max_seq_len x hidden_dim))
          pos_embed=nn.Embedding(
            num_embeddings=config.max_seq_len,
            embedding_dim=config.hidden_dim
          ),
          # iteratively adds identical heads.
          heads=nn.ModuleList([
            Block(config) for _ in range(config.num_decoders)
          ]),
          # last layer normalization, exists outside the transformer block,
          # so it gets added here. We would normally use RMSNorm in 2024.
          final_layernorm=nn.LayerNorm(config.hidden_dim)
        ))
        # our output is (max_seq_len x hidden_dim), need to convert back to
        # vocab with learned weight matrix of size (hidden_dim, vocab_size).
        # like an inversion of embedding_matrix but we're not getting back
        # a definite token's index, we're getting some values that we will then
        # use softmax to turn into probabilities from [0, 1] that sum to 1,
        # because the probability of the correct token being one of the tokens
        # possible, is of course 100%.
        self.lm_head = nn.Linear(
          in_features=config.hidden_dim,
          out_features=config.vocab_size,
          bias=False)
    
    def forward(self, idx, predictions=None):
      # index of shape (batch_size, seq_len)
      batch_size, seq_len = idx.size()
      assert seq_len <= self.config.max_seq_len, f"cannot use sequence length {seq_len} > {self.config.max_seq_len}"
      # first we create the position tensor, it's just 0 to seq_len
      pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device) # shape is (seq_len,)
      # we run it through its embedding layer
      pos_embed = self.transformer.pos_embed(pos) # shape is (seq_len, hidden_dim)
      # then we embed the tokens with their own embedding layer
      token_embed = self.transformer.embedding_matrix(idx) # shape is (batch_size, seq_len, hidden_dim)
      # we add the two embeddings together, with the position embedding being broadcast along batch dimension
      data = token_embed + pos_embed # shape is (batch_size, seq_len, hidden_dim)
      # run through all the transformer blocks
      for decoder in self.transformer.heads:
        data = decoder(data)
      # final layer norm
      data = self.transformer.final_layernorm(data) # shape is (batch_size, seq_len, hidden_dim)
      # run through the linear layer to get back to vocab space
      logits = self.lm_head(data) # shape is (batch_size, seq_len, vocab_size)
      # Get cross-entropy loss if real labels provided.
      loss = None
      if predictions is not None:
        # logits starts as (batch_size, window, vocab_size) => (all_tokens, vocab_size)
        # logits.view(-1 = calculate this dim automatically so batch*tokens = all_tokens,
        #             logits.size(-1) = vocab_size dim)
        # predictions.view(-1) = flatten so (batch_size, token_indices) turns into (all_token_indices).
        # May seem odd, we're calculating cross entropy for the whole training set instead of per-batch,
        # per-batch. The reason is that this is a small model with a toy dataset.
        loss = cross_entropy_impl(logits.view(-1, logits.size(-1)), predictions.view(-1))
      
      return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """We're only really going to load one, but we can include this data for all"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        # random debug line, why not.
        print("loading weights from pretrained gpt: %s" % model_type)

        # these are the params for GPT2Parameters from the paper.
        config_args = {
            'gpt2':         dict(num_decoders=12, num_heads=12, hidden_dim=768),  # 124M params
            'gpt2-medium':  dict(num_decoders=24, num_heads=16, hidden_dim=1024), # 350M params
            'gpt2-large':   dict(num_decoders=36, num_heads=20, hidden_dim=1280), # 774M params
            'gpt2-xl':      dict(num_decoders=48, num_heads=25, hidden_dim=1600), # 1558M params
        }[model_type]
        # model_type is a key, we're declaring the full dict then pulling just one value.

        # again, from paper. These are the same for all GPT2 models.
        # rename to match our modified GPT2Parameters class.
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['max_seq_len'] = 1024 # always 1024 for GPT model checkpoints

        # create a from-scratch initialized nanoGPT model
        config = GPT2Parameters(**config_args)
        model = GPT(config)
        state_dict = model.state_dict()
        # we do full mapping of names from loaded to this model later.
        sd_keys = state_dict.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attention.bias')] # discard this mask / buffer, not a param

        # init the huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        statedict_loaded = model_hf.state_dict()

        # all their linear layers are transposed because imported from tf, so they're gonna need
        # to be re-transposed to match our model's expectations.
        sd_keys_loaded = statedict_loaded.keys()
        sd_keys_loaded = [k for k in sd_keys_loaded if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_loaded = [k for k in sd_keys_loaded if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_loaded) == len(sd_keys), f"mismatched keys: {len(sd_keys_loaded)} != {len(sd_keys)}"
        for k in sd_keys_loaded:
            new_key = old_name_to_new(k)
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert statedict_loaded[k].shape[::-1] == state_dict[new_key].shape
                with torch.no_grad():
                    state_dict[new_key].copy_(statedict_loaded[k].t())
            else:
                # vanilla copy over the other parameters
                assert statedict_loaded[k].shape == state_dict[new_key].shape
                with torch.no_grad():
                    state_dict[new_key].copy_(statedict_loaded[k])

        return model
    
# Because I insisted on naming everything according to my preferred method,
# I need to translate the layer names from the loaded paper into my names.
# And I need it to handle variable number of heads.

common_layer_mapping = {
    'transformer.wte.weight': 'transformer.embedding_matrix.weight',
    'transformer.wpe.weight': 'transformer.pos_embed.weight',
    'transformer.ln_f.weight': 'transformer.final_layernorm.weight',
    'transformer.ln_f.bias': 'transformer.final_layernorm.bias',
    'lm_head.weight': 'lm_head.weight'
}

head_layer_mapping = {
       'ln_1.weight': 'ln_before_att.weight',
       'ln_1.bias': 'ln_before_att.bias', 
       'attn.c_attn.weight': 'attention.qkv_projection.weight',
       'attn.c_attn.bias': 'attention.qkv_projection.bias',
       'attn.c_proj.weight': 'attention.out_projection.weight',
       'attn.c_proj.bias': 'attention.out_projection.bias',
       'ln_2.weight': 'ln_before_fnn.weight',
       'ln_2.bias': 'ln_before_fnn.bias',
       'mlp.c_fc.weight': 'mlp.ffn1.weight', 
       'mlp.c_fc.bias': 'mlp.ffn1.bias',
       'mlp.c_proj.weight': 'mlp.ffn2.weight',
       'mlp.c_proj.bias': 'mlp.ffn2.bias'
}

def old_name_to_new(name):
    # first we check if it's a common layer
    if name in common_layer_mapping:
        return common_layer_mapping[name]
    # then update transformer decoder block name
    if 'transformer.h.' in name:
        name = name.replace('.h.', '.heads.')
    # then we check if it's a head layer
    for old, new in head_layer_mapping.items():
        if old in name:
            return name.replace(old, new)
    # if it's neither, we just return the name as-is
    return name


def cross_entropy_impl(logits, labels):
  """
  Assumes logits are (input_size, vocab_size) and labels are (input_size)
  No rearranging going to happen in here.
  """
  # 1. Get probabilities using softmax,
  # apply softmax per-token so 1 since logits here are gonna be (all_tokens, vocab_size)
  probs = torch.softmax(logits, dim=1)

  # 2. Select probabilities corresponding to the true labels
  # first get indices vector 0..input_size-1
  index_range = torch.arange(labels.size(0))
  # then we access probabilities by iterating like this.
  # Indexing here is vector for rows (index_range), and vector for columns (labels)
  # and it'll go like "for i in range(index_range/labels): probs[index_range[i], labels[i]]"
  # and the output is a vector of scalar probabilities.
  predicted_probs = probs[index_range, labels]

  # 3. Formula for cross entropy is log(real probability) - log(predicted probability)
  # but real probability is 1 and log(1) = 0 so the formula gets reduced to
  # -log(predicted probability).
  neg_log_likelihood = -torch.log(predicted_probs)

  # 4. Average the loss to return a scalar value.
  loss = torch.mean(neg_log_likelihood)

  return loss

def test_model():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # This is a more complex test of the model being loaded properly
  # we do the same starting text, and run it in parallel this many times.
  # this is also our batch_size, and I use that term when describing matrix dims.
  num_parallel_responses = 5
  # the maximum length of a response (including the starting text)
  max_response_len = 30

  model = GPT.from_pretrained('gpt2')
  # this means inference mode, no dropout, no batchnorm
  model.eval()
  # move the model to GPU memory if GPU available.
  model.to(device)

  # get the tokenizer made for this model
  enc = tiktoken.get_encoding('gpt2')
  # this is the starting text, we're going to generate continuations for it.
  # returns list[int] of token indices
  tokens = enc.encode("Hello, I'm a language model,")
  # turn the list into a tensor of longs since models want tensors 
  tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
  # add batch dimension at the beginning, and repeat tensor 5 times in it.
  # so, [1, 2, 3] -> [[1, 2, 3], ....] 5 times
  # this is how we turn the initial "hello..." into 5 parallel responses.
  tokens = tokens.unsqueeze(0).repeat(num_parallel_responses, 1) # (5, 8)
  # move the input tensor to GPU memory if GPU available.
  input_data = tokens.to(device) # (batch_size, seq_len)

  # set the random seeds for reproducibility
  torch.manual_seed(8)
  torch.cuda.manual_seed(8)

  # while the length of the token dim is less than our max
  while input_data.size(1) < max_response_len:
      # no_grad means don't track gradients, so less memory used.
      # gradients not needed since not training.
      with torch.no_grad():
          # model outputs pre-probability logits.
          logits = model(input_data) # (batch, tokens, vocab_size)
          # take the logits for the last token (the one the model predicted)
          logits = logits[:, -1, :] # (batch, vocab_size)
          # softmax makes the values be [0, 1] and sum to 1.
          # dim = -1 (last) because that's where the vocab_size is.
          probs = torchF.softmax(logits, dim=-1)
          # basically sorts the probabilities while also keeping indices
          # and outputs the top 50 probabilities and their indices.
          # both are (batch_size, top_k) shaped.
          # topk_probs here becomes (5, 50), topk_indices is (5, 50)
          topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
          # multinomial is a probability distribution for picking from many categories.
          # picks a single index - any can be picked but
          # each has its probability's chance of being picked
          picked_ix = torch.multinomial(topk_probs, 1) # (batch_size, 1)
          # torch.gather is like "pick from this tensor at these indices"
          xcol = torch.gather(topk_indices, -1, picked_ix) # (batch_size, 1)
          # concatenate onto the existing input along dim 1 (tokens dim)
          # note - we concatenated the *index* of the next token, since
          # the model input is a sequence of token indices.
          input_data = torch.cat((input_data, xcol), dim=1)

  # print the num_parallel_responses generated completions.
  for i in range(num_parallel_responses):
      tokens = input_data[i, :max_response_len].tolist()
      decoded = enc.decode(tokens)
      print(">", decoded)

def train_model():
  # !wget https://raw.githubusercontent.com/karpathy/char-rnn/refs/heads/master/data/tinyshakespeare/input.txt
  # pros: fully ascii no preprocessing needed cons: very small dataset, can easily find better ones on github or huggingface.

  # something about loading to cuda is busted colab with torch when we load model here.
  device = 'cpu'

  # get the specific tokenizer used for gpt2 to tokenize/detokenize text.
  enc = tiktoken.get_encoding('gpt2')

  # rename if you saved the shakespear text as something else.
  with open('input.txt', 'r') as f:
      text = f.read()

  # encode the text into token indices.
  tokens = enc.encode(text)

  # Forced to use the cpu due to torch device weirdness in colab,
  # and at 12.7 gb RAM have to keep batch size low
  # together these batch/length params use 10.8 GB RAM (minus ~1.4 os/python usage)
  token_length = 128
  batch_size = 32
  # convert token indices to floats
  buf = torch.tensor(tokens[:batch_size*token_length + 1])

  # exclude first token from labels since it has no context
  labels = buf[1:].view(batch_size, token_length)
  # exclude last token from input since it has nothing to predict
  data = buf[:-1].view(batch_size, token_length)

  model = GPT(GPT2Parameters())
  model.to(device)
  optimizer = AdamW_Impl(model.parameters())
  for i in range(50):
      optimizer.zero_grad()
      logits, loss = model(data, labels)
      loss.backward()
      optimizer.step()
      print(f"step {i}, loss: {loss.item()}")

class AdamW_Impl:
    def __init__(self, params, device):
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
        self.lr = torch.tensor(3e-4).to(device)          
        # Beta coefficients for Adam
        # beta1 for the first moment (gradient smoothing)
        # beta2 for the second moment (gradient scaling)
        self.betas = (torch.tensor(0.9).to(device), torch.tensor(0.999).to(device))
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