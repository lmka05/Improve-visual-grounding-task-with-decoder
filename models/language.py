import torch 
import torch.nn as nn

class LanguageEncoder(nn.Module):
    def __init__(self, glove_vectors, hidden_size = 512, pooling="max"):
        super().__init__()
        self.hidden_size = hidden_size
        self.pooling = pooling # chế độ pooling: "max", "mean", "last"
        vocab_size, embed_dim = glove_vectors.shape

        self.embedding = nn.Embedding.from_pretrained(
            glove_vectors,
            padding_idx=0,
            freeze = True 
        )
        
        self.gru = nn.GRU(
            input_size = embed_dim,
            hidden_size = hidden_size,
            num_layers =1,
            bidirectional = True,
            batch_first = True,
            bias = True,
            dropout = 0.0,
        )

    def forward(self, ref_inds):
        mask = (ref_inds == 0) 
        emb = self.embedding(ref_inds)
        output, hidden = self.gru(emb) 

        if self.pooling == "max":
            output = output.masked_fill(mask.unsqueeze(-1), float('-inf'))
            y = output.max(dim =1, keepdim = True).values 
        elif self.pooling == "mean":
            output = output.masked_fill(mask.unsqueeze(-1), 0.0)
            valid_counts = (~mask).sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1) 
            y = output.sum(dim=1, keepdim=True) / valid_counts 
        elif self.pooling == "last":
            y = torch.cat([hidden[-2], hidden[-1]], dim=-1).unsqueeze(1) 
        
        return y
