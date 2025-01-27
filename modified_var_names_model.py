"""
Code is from Andrej Karpathy's 'Let's reproduce GPT-2 (124M)'
https://www.youtube.com/watch?v=l8pRSuU81PU
I have modified names in many places where I felt like it helped me
to understand it easier, and provided various comments.
This is meant to be a base from which to start adding various papers'
improvements.

Please note this version changes the variable names so normal
'from_pretrained' wouldn't work here, but I liked it for
actively engaging with the material.

"""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as torchF
from transformers import GPT2LMHeadModel

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
            print(f"Processing key: {k}") # DEBUG PRINT
            new_key = old_name_to_new(k)
            print(f"Translated key: {new_key}")
            print("State dict keys:", state_dict.keys())
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

# this is a simple test to see if the model loads correctly
model = GPT.from_pretrained('gpt2')
print("didn't crash yay!")