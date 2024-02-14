import torch.nn as nn
from sub_layer import MultiHeadAttention, PositionwiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, dropout):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model=d_model, d_inner=d_inner, dropout=dropout)

    def forward(self, enc_input, mask=None):
        enc_output, attn = self.self_attn(enc_input, enc_input, enc_input, mask=mask)
        enc_output = self.pos_ffn(enc_output)

        return enc_output, attn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, dropout, use_additional_global_attn=False):
        super(DecoderLayer, self).__init__()
        self.use_additional_global_attn = use_additional_global_attn
        if self.use_additional_global_attn:
            self.local_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout)
            self.category_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout)
            self.id_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout)
            self.self_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout)
        else:
            self.self_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout)
        self.cross_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model=d_model, d_inner=d_inner, dropout=dropout)

    def forward(self, enc_output, dec_input, enc_mask=None, dec_mask=None, local_mask=None, category_mask=None, id_mask=None):
        if self.use_additional_global_attn:
            local_output, _ = self.local_attn(dec_input, dec_input, dec_input, mask=local_mask)
            category_output, _ = self.category_attn(dec_input, dec_input, dec_input, mask=category_mask)
            id_output, _ = self.id_attn(dec_input, dec_input, dec_input, mask=id_mask)
            self_output, _ = self.self_attn(dec_input, dec_input, dec_input, mask=dec_mask)
            dec_output = category_output + id_output + local_output + self_output
        else:
            dec_output, _ = self.self_attn(dec_input, dec_input, dec_input, mask=dec_mask)

        dec_output, _ = self.cross_attn(dec_output, enc_output, enc_output, mask=enc_mask)
        dec_output = self.pos_ffn(dec_output)

        return dec_output