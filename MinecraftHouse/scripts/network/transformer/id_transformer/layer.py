import torch.nn as nn
from sub_layer import MultiHeadAttention, PositionwiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, dropout):
        super(EncoderLayer, self).__init__()

        self.global_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout)
        self.category_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model=d_model, d_inner=d_inner, dropout=dropout)

    def forward(self, enc_input, global_mask, category_mask):
        global_output, _ = self.global_attn(enc_input, enc_input, enc_input, mask=global_mask)
        category_output, _ = self.category_attn(enc_input, enc_input, enc_input, mask=category_mask)
        enc_output = global_output + category_output

        enc_output = self.pos_ffn(enc_output)

        return enc_output

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, dropout):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout)
        self.cross_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model=d_model, d_inner=d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, mask=None):
        dec_output, _ = self.self_attn(dec_input, dec_input, dec_input, mask=mask)
        dec_output, _ = self.cross_attn(dec_output, enc_output, enc_output, mask=mask)
        dec_output = self.pos_ffn(dec_output)

        return dec_output