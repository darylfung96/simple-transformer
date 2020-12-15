import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# define the embedding for the inputs
class Embedder(nn.Module):
    def __init__(self, vocab_size, dim):
        super(Embedder, self).__init__()
        self.embed = nn.Embedding(vocab_size, dim)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_sequence_length=80):
        super(PositionalEncoding, self).__init__()
        self.dim = dim

        # create the positional encoding
        pe = torch.zeros(max_sequence_length, dim)
        for pos in range(max_sequence_length):
            for i in range(0, dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/dim)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # this is to make the positional encoding relatively smaller so we emphasize more on the actual features
        x = x * math.sqrt(self.dim)

        seq_length = x.size(1)
        x = x + Variable(self.pe[:, :seq_length], requires_grad=False)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, dims, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.dims = dims
        self.dim_each = dims//heads
        self.heads = heads

        self.q_linear = nn.Linear(dims, dims)
        self.k_linear = nn.Linear(dims, dims)
        self.v_linear = nn.Linear(dims, dims)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dims, dims)

    def attention(self, q, k, v, dims_each, mask, dropout):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dims_each)
        # mask it
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)

        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        k = self.k_linear(k).view(batch_size, self.heads, -1, self.dim_each)
        q = self.q_linear(q).view(batch_size, self.heads, -1, self.dim_each)
        v = self.v_linear(v).view(batch_size, self.heads, -1, self.dim_each)

        scores = self.attention(q, k, v, self.dim_each, mask, self.dropout)
        scores = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.dims)  # after transpose need contiguous
        output = self.out(scores)
        return output


class LinearMultiHeadAttention(nn.Module):
    def __init__(self, heads, seq_length, dims, k=128, dropout=0.1):
        super(LinearMultiHeadAttention, self).__init__()

        self.dims = dims
        self.dim_each = dims//heads
        self.heads = heads
        self.k = k

        self.q_linear = nn.Linear(dims, dims)
        self.k_linear = nn.Linear(dims, dims)
        self.v_linear = nn.Linear(dims, dims)

        def init_(tensor):
            dim = tensor.shape[-1]
            std = 1 / math.sqrt(dim)
            tensor.uniform_(-std, std)
            return tensor
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_length, k)))
        self.proj_v = nn.Parameter(init_(torch.zeros(seq_length, k)))

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dims, dims)

    def attention(self, q, k, v, dims_each, mask, dropout):
        # q (b, h, n, d)
        # k/v (b, h, k, d)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dims_each)  # b, h, n, k
        scores = scores.transpose(2, 3)  # b, h, k, n
        # mask it
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)

        output = torch.einsum('bhkn,bhkd->bhnd', scores, v)  # torch.matmul(scores, v)  # v (b, h, k, d)
        return output

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        strlen = q.size(1)

        k = self.k_linear(k)  # .view(batch_size, self.heads, -1, self.dim_each)
        q = self.q_linear(q).view(batch_size, self.heads, -1, self.dim_each)
        v = self.v_linear(v)  # .view(batch_size, self.heads, -1, self.dim_each)

        projected_k = torch.einsum('bnd,nk->bkd', k, self.proj_k[:strlen, :])
        projected_v = torch.einsum('bnd,nk->bkd', v, self.proj_v[:strlen, :])
        projected_k = projected_k.reshape(batch_size, self.k, self.heads, self.dim_each).transpose(1, 2).\
            expand(-1, self.heads, -1, -1)
        projected_v = projected_v.reshape(batch_size, self.k, self.heads, self.dim_each).transpose(1, 2).\
            expand(-1, self.heads, -1, -1)

        scores = self.attention(q, projected_k, projected_v, self.dim_each, mask, self.dropout)
        scores = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.dims)  # after transpose need contiguous
        output = self.out(scores)
        return output


class FeedForward(nn.Module):
    def __init__(self, dims, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.dims = dims
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(dims, d_ff)
        self.linear2 = nn.Linear(d_ff, dims)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


class Norm(nn.Module):
    def __init__(self, dims, eps=1e-6):
        super(Norm, self).__init__()

        self.dims = dims
        self.alpha = nn.Parameter(torch.ones(self.dims))
        self.bias = nn.Parameter(torch.zeros(self.dims))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class EncoderLayer(nn.Module):
    def  __init__(self, dims, heads, opt, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.norm1 = Norm(dims)
        self.norm2 = Norm(dims)
        self.attn = LinearMultiHeadAttention(heads, opt.max_strlen, dims, dropout=dropout)
        self.ff = FeedForward(dims)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.attn(x2, x2, x2, mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.ff(x2))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, dims, heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.norm1 = Norm(dims)
        self.norm2 = Norm(dims)
        self.norm3 = Norm(dims)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.attn1 = MultiHeadAttention(heads, dims)
        self.attn2 = MultiHeadAttention(heads, dims)
        self.ff = FeedForward(dims)

    def forward(self, x, encoder_outputs, src_mask, target_mask):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.attn1(x2, x2, x2, target_mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.attn2(x2, encoder_outputs, encoder_outputs, src_mask))
        x2 = self.norm3(x)
        x = x + self.dropout3(self.ff(x2))
        return x


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, vocab_size, dims, N, heads, opt):
        super(Encoder, self).__init__()
        self.embed = Embedder(vocab_size, dims)
        self.pe = PositionalEncoding(dims)
        self.layers = get_clones(EncoderLayer(dims, heads, opt), N)
        self.norm = Norm(dims)
        self.num_layers = N

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)

        for i in range(self.num_layers):
            x = self.layers[i](x, mask)
        return self.norm(x)


class HierarchicalEncoder(nn.Module):
    def __init__(self, vocab_size, dims, N, heads, opt):
        super(HierarchicalEncoder, self).__init__()
        self.embed = Embedder(vocab_size, dims)
        self.pe = PositionalEncoding(dims)
        self.layers = get_clones(EncoderLayer(dims, heads, opt), N)
        self.norm = Norm(dims)
        self.num_layers = N

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)

        encoder_outputs = []
        for i in range(self.num_layers):
            x = self.layers[i](x, mask)
            encoder_outputs.append(self.norm(x))
        return encoder_outputs


class Decoder(nn.Module):
    def __init__(self, vocab_size, dims, N, heads, opt):
        super(Decoder, self).__init__()
        self.opt = opt
        self.embed = Embedder(vocab_size, dims)
        self.pe = PositionalEncoding(dims)
        self.layers = get_clones(DecoderLayer(dims, heads), N)
        self.norm = Norm(dims)
        self.num_layers = N

    def forward(self, target, encoder_outputs, source_mask, target_mask):
        x = self.embed(target)
        x = self.pe(x)
        for i in range(self.num_layers):
            if self.opt.is_hierarchical:
                x = self.layers[i](x, encoder_outputs[i], source_mask, target_mask)
            else:
                x = self.layers[i](x, encoder_outputs, source_mask, target_mask)

        return self.norm(x)


class SimpleTransformer(nn.Module):
    def __init__(self, src_vocab, target_vocab, dims, N, heads, opt):
        super(SimpleTransformer, self).__init__()
        self.opt = opt
        if opt.is_hierarchical:
            self.encoder = HierarchicalEncoder(src_vocab, dims, N, heads, opt)
        else:
            self.encoder = Encoder(src_vocab, dims, N, heads, opt)
        self.decoder = Decoder(target_vocab, dims, N, heads, opt)
        self.out = nn.Linear(dims, target_vocab)

    def forward(self, src, target, src_mask, target_mask):
        encoder_outputs = self.encoder(src, src_mask)
        decoder_outputs = self.decoder(target, encoder_outputs, src_mask, target_mask)
        output = self.out(decoder_outputs)
        return output  # don't need to do softmax because loss function will apply softmax



