import torch
import torch.nn as nn
import math
import numpy as np

class IOEmbeddingBlock(nn.Module):
    """
    Args:
        d_model (int): Embedding Length
        vocabulary_size (int): Vocabulary Size 
    """

    def __init__(self, d_model: int, vocabulary_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocabulary_size
        self.embedding = nn.Embedding(vocabulary_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncodingBlock(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        # We only need dropout as there is no learning for
        # Positional Encoding
        self.dropout = nn.Dropout(dropout)

        positional_encoding = torch.zeros(seq_len, d_model)

        # convert it to column vector for outer product
        pos = torch.arange(0, seq_len, dtype=float).unsqueeze(1)
        two_i = torch.arange(0, d_model, 2, dtype=float)
        exponantial = torch.exp(two_i * (-math.log(10000.0) / d_model))

        # [seq_len,1] x [1, d_model/2] = [seq_len, d_model/2]
        positional_encoding[:, 0::2] = torch.sin(pos * exponantial)
        positional_encoding[:, 0::2] = torch.cos(pos * exponantial)

        # Create the batch dimension for python brodcast
        # [1, seq_len, d_model]
        positional_encoding = positional_encoding.unsqueeze(0)

        # Use register buffer to save positional_encoding when the
        # model is exported in disk.
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x):
        # All the example code truncates the seq_len (2nd axis) using
        # :x.shap[1]. Need to find more information on this.
        # Tested and it can be simply added. 
        x = x + (self.positional_encoding[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    

class MaskedMultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, head: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.head = head

        # d_model needs to be divisible by head
        assert d_model % head == 0, "d_model is not divisible by head"

        self.d_k = d_model // head

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)

        # Paper used another dense layer here
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def attention(self, q, k, v, mask, dropout: nn.Dropout):
        # Multiply query and key transpose
        # (batch, head, seq_len, d_k) @ (batch, head, d_k, seq_len)
        # = [seq_len, d_k] @ [d_k, seq_len] = [seq_len,seq_len]
        # = (batch, head, seq_len, seq_len)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # Whereever mask is zero fill with very small value and
            # not 0
            scores.masked_fill_(mask == 0, -1e8)

        # (batch, head, seq_len, seq_len)
        # dim=-1 is seq_len
        scores = scores.softmax(dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        # (batch, head, seq_len, seq_len) @ # (batch, head, seq_len, d_model)
        # (seq_len, seq_len) @ (seq_len, d_model) = (seq_len, d_model)
        # -> (batch, head, seq_len, d_model)
        return (scores @ v), scores

    def forward(self, q, k, v, mask): 

        query = self.Wq(q)
        key = self.Wk(k)
        value = self.Wv(v)

        # Attention head on embedding dimension and not on sequence/batch dimension.
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k)
        query = query.view(query.shape[0], query.shape[1], self.head, self.d_k)
        # All the calculations happens between (seq_len, d_k)
        # So need to take transpose of h and seq_len.
        # (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.transpose(1, 2)

        # Same way in one line
        key = key.view(key.shape[0], key.shape[1], self.head, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.head, self.d_k).transpose(1, 2)

        x, self.attention_scores = self.attention(
            query, key, value, mask, self.dropout
        )

        # Now need to merge head dimension to d_model

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.head * self.d_k)

        return self.Wo(x)

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        # Default d_ff was 2048 in the paper
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class LayerNormalizationBlock(nn.Module):
    def __init__(self, eps: float = 1e-7):
        super().__init__()

        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class SkipConnectionBlock(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.layer_norm_block = LayerNormalizationBlock()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm_block(x)))
    
class EndoderBlock(nn.Module):
    def __init__(self, attention_block: MaskedMultiHeadSelfAttentionBlock, ff_block: FeedForwardBlock, dropout: float):
        super().__init__()

        self.attention_block = attention_block
        self.ff_block = ff_block

        #self.skip_conn1=ResidualConnection(dropout)
        #self.skip_conn2=ResidualConnection(dropout)
        
        self.skip_connections = nn.ModuleList([SkipConnectionBlock(dropout) for _ in range(2)])
        

    def forward(self, x, src_mask):
        #x=self.skip_conn1(x,lambda x: self.attention(x, x, x, src_mask))
        #x=self.skip_conn2(x,self.ff)
        
        x = self.skip_connections[0](x, lambda x: self.attention_block(x, x, x, src_mask))
        x = self.skip_connections[1](x, self.ff_block)
        
        return x
    
class EncoderSequenceBlock(nn.Module):
    def __init__(self, encoder_blocks: nn.ModuleList):
        super().__init__()
        self.encoder_blocks = encoder_blocks
        self.norm = LayerNormalizationBlock()

    def forward(self, x, mask):
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, mask)
		# Additional Layer Normalization
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    def __init__(
        self,
        attention_block: MaskedMultiHeadSelfAttentionBlock,
        cross_attention_block: MaskedMultiHeadSelfAttentionBlock,
        ff_block: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()
        self.attention_block = attention_block
        self.cross_attention_block = cross_attention_block
        self.ff_block = ff_block

        self.skip_connections = nn.ModuleList(
            [SkipConnectionBlock(dropout) for _ in range(3)]
        )

    def forward(self, x, last_encoder_block_output, src_mask, tgt_mask):
        x = self.skip_connections[0](
            x, lambda x: self.attention_block(x, x, x, tgt_mask))
        x = self.skip_connections[1](
            x, lambda x: self.cross_attention_block(
                x, last_encoder_block_output, last_encoder_block_output, src_mask),
        )
        x = self.skip_connections[2](x, self.ff_block)

        return x

class DecoderSequenceBlock(nn.Module):
    def __init__(self, decoder_blocks: nn.ModuleList):
        super().__init__()
        self.decoder_blocks = decoder_blocks
        self.norm = LayerNormalizationBlock()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class LinearSoftmaxBlock(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # Softmax using the last dimension (embedding)
        # [batch, seq_len, d_model] -> [batch, seq_len]
        return torch.log_softmax(self.linear(x), dim=-1)

class Transformer(nn.Module):
    def __init__(
        self,
        input_embed_block: IOEmbeddingBlock,
        output_embed_block: IOEmbeddingBlock,
        input_pos_block: PositionalEncodingBlock,
        output_pos_block: PositionalEncodingBlock,
        encoder_seq_block: EncoderSequenceBlock,
        decoder_seq_block: DecoderSequenceBlock,        
        linear_softmax_block: LinearSoftmaxBlock,
    ):
        super().__init__()        
        self.input_embed_block = input_embed_block
        self.output_embed_block = output_embed_block
        self.input_pos_block = input_pos_block
        self.output_pos_block = output_pos_block
        self.encoder_seq_block = encoder_seq_block
        self.decoder_seq_block = decoder_seq_block
        self.linear_softmax_block = linear_softmax_block

    def encode(self, src, src_mask):
        src = self.input_embed_block(src)
        src = self.input_pos_block(src)
        return self.encoder_seq_block(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.output_embed_block(tgt)
        tgt = self.output_pos_block(tgt)
        return self.decoder_seq_block(tgt, encoder_output, src_mask, tgt_mask)

    def linear_projection(self, x):
        return self.linear_softmax_block(x)

