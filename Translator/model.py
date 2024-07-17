import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout_p)
 
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
 
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
 
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)
 
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class Transformer(nn.Module):
    def __init__(self, d_model: int, dff: int, vocab_size: int,max_len: int, device, pad_token: int = 0):
        super(Transformer, self).__init__()

        self.pad_token = pad_token
        self.d_model = d_model
        self.device = device

        self.embd = nn.Embedding(vocab_size, d_model, self.pad_token)

        self.positional = PositionalEncoding(d_model, 0.1, max_len)
        self.transformer = nn.Transformer(d_model, dim_feedforward=dff)

        self.ffnn = nn.Linear(d_model, vocab_size)

    def forward(self, enc_in, dec_in):
        enc_pad_mask = self.pad_mask(enc_in, self.pad_token)
        dec_look_ahead_mask = self.look_ahead_masks(dec_in, self.pad_token)

        enc_in = self.embd(enc_in) * math.sqrt(self.d_model)
        dec_in = self.embd(dec_in) * math.sqrt(self.d_model)
        enc_in = self.positional(enc_in)
        dec_in = self.positional(dec_in)

        enc_in = enc_in.permute(1, 0, 2)
        dec_in = dec_in.permute(1, 0, 2)

        output = self.transformer(enc_in, dec_in, src_key_padding_mask=enc_pad_mask,tgt_mask=dec_look_ahead_mask)

        output = self.ffnn(output)

        return output.permute(1, 2, 0)

    def pad_mask(self, seq, pad_token: int = 0):
        mask = (seq == pad_token)
        return mask.to(device=self.device)

    
    def look_ahead_masks(self, seq, pad_token: int = 0):
        batch_size, size = seq.size()
        tgt_mask = torch.triu(torch.ones(size, size), diagonal=1)
        return tgt_mask.to(device=self.device)