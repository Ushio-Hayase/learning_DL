import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len,d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, d_model: int, dff: int, vocab_size: int,max_len: int, device, pad_token: int = 0):
        super(Transformer, self).__init__()

        self.pad_token = pad_token
        self.d_model = d_model
        self.device = device

        self.embd = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_token)

        self.positional = PositionalEncoding(d_model, 0.1, max_len)
        self.transformer = nn.Transformer(d_model, dim_feedforward=dff, dropout=0.1)

        self.ffnn = nn.Linear(d_model, vocab_size)


    def forward(self, enc_in, dec_in):
        enc_pad_mask = self.create_pad_mask(enc_in, self.pad_token)
        dec_look_ahead_mask = self.get_tgt_mask(dec_in.size(1))
        dec_pad_mask = self.create_pad_mask(dec_in, self.pad_token)

        enc_in = self.embd(enc_in) * math.sqrt(self.d_model)
        dec_in = self.embd(dec_in) * math.sqrt(self.d_model)


        enc_in = self.positional(enc_in)
        dec_in = self.positional(dec_in)


        enc_in = enc_in.permute(1, 0, 2)
        dec_in = dec_in.permute(1, 0, 2)

        

        output = self.transformer(enc_in, dec_in, src_key_padding_mask=enc_pad_mask,tgt_mask=dec_look_ahead_mask, tgt_key_padding_mask = dec_pad_mask)

        output = self.ffnn(output)

        return output.permute(1, 2, 0)

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask.to(device=self.device)
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int = 0) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token).to(device=self.device)