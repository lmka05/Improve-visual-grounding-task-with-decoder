import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinePositionalEncoding2D(nn.Module):
    def __init__(self, num_feature, temperature=10000, normalize=True):
        super().__init__()
        self.num_feature = num_feature
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, mask):
        not_mask = ~mask  
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_feature, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feature)
        pos_x = x_embed[:, :, :, None] / dim_t  
        pos_y = y_embed[:, :, :, None] / dim_t  

        B, H, W = mask.shape
        pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()], dim=4).view(B, H, W, -1)
        pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()], dim=4).view(B, H, W, -1)

        pos = torch.cat([pos_y, pos_x], dim=3).permute(0, 3, 1, 2)

        return pos

def quantize_bbox(bbox, img_meta, num_bin=1000):
    B = bbox.shape[0]
    pad_w = torch.tensor([m['pad_shape'][1] for m in img_meta],
                         device=bbox.device, dtype=bbox.dtype)  
    pad_h = torch.tensor([m['pad_shape'][0] for m in img_meta],
                         device=bbox.device, dtype=bbox.dtype)  

    scale = torch.stack([pad_w, pad_h, pad_w, pad_h], dim=1)  
    tokens = (bbox / scale * num_bin).long()
    tokens = tokens.clamp(0, num_bin - 1)

    return tokens


def dequantize_bbox(tokens, img_meta, num_bin=1000):
    B = tokens.shape[0]

    pad_w = torch.tensor([m['pad_shape'][1] for m in img_meta],
                         device=tokens.device, dtype=torch.float32)
    pad_h = torch.tensor([m['pad_shape'][0] for m in img_meta],
                         device=tokens.device, dtype=torch.float32)

    scale = torch.stack([pad_w, pad_h, pad_w, pad_h], dim=1)  # [B, 4]
    bbox = tokens.float() / num_bin * scale

    return bbox

class SeqHead(nn.Module):

    def __init__(self, in_ch=1024, d_model=256, nhead=8, dim_feedforward=1024,
                 dropout=0.1, enc_layers=6, dec_layers=3,
                 num_bin=1000, label_smoothing=0.1, token_weights=None):
        super().__init__()

        self.d_model = d_model
        self.num_bin = num_bin
        self.vocab_size = num_bin + 1  
        self.seq_len = 4              

        # Input Projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_ch, d_model, kernel_size=1, bias=True),
            nn.GroupNorm(32, d_model), 
        )

        # Positional Encodings
        self.pos_enc_2d = SinePositionalEncoding2D(
            num_feature=d_model // 2, 
            normalize=True
        )
        self.pos_enc_1d = nn.Embedding(
            num_embeddings=self.seq_len + 1,  
            embedding_dim=d_model,             
        )

        # Token Embedding
        self.token_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,  
            embedding_dim=d_model,
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)

        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, self.vocab_size),  # → 1001 classes
        )

        # Loss
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction='none')

        if token_weights is not None:
            self.register_buffer('token_weights',
                torch.tensor(token_weights, dtype=torch.float32))
        else:
            self.token_weights = None

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def _encode(self, x_fused, img_metas):
        B = x_fused.shape[0]
        x = self.input_proj(x_fused)
        _, _, H, W = x.shape
        input_h = img_metas[0]['pad_shape'][0]  
        input_w = img_metas[0]['pad_shape'][1]  
        x_mask = x_fused.new_ones((B, H, W))

        for i in range(B):
            img_h, img_w, _ = img_metas[i]['img_shape']
            feat_h = int(round(img_h * H / input_h))
            feat_w = int(round(img_w * W / input_w))
            feat_h = min(feat_h, H)
            feat_w = min(feat_w, W)
            x_mask[i, :feat_h, :feat_w] = 0

        x_mask = x_mask.bool()

        # Positional encoding
        x_pos = self.pos_enc_2d(x_mask)  

        # Flatten spatial dims
        x = x.flatten(2).transpose(1, 2)          
        x_pos = x_pos.flatten(2).transpose(1, 2)  
        x_mask = x_mask.flatten(1)                  

        # Thêm positional encoding vào features
        x_with_pos = x + x_pos

        # Transformer Encoder
        memory = self.encoder(
            x_with_pos,
            src_key_padding_mask=x_mask
        )  # [B, H*W, 256]

        return memory, x_mask, x_pos

    def forward_train(self, x_fused, gt_bbox, img_metas):
        B = x_fused.shape[0]
        device = x_fused.device

        # Encode visual features -> memory
        memory, x_mask, x_pos = self._encode(x_fused, img_metas)

        # Quantize GT bbox -> tokens
        gt_tokens = quantize_bbox(gt_bbox, img_metas, self.num_bin)

        # Tạo target (thêm END token ở cuối)
        end_token = torch.full((B, 1), self.num_bin, dtype=torch.long, device=device)
        targets = torch.cat([gt_tokens, end_token], dim=1)  

        # Tạo decoder input
        start_embed = torch.zeros(B, 1, self.d_model, device=device)  
        gt_embeds = self.token_embedding(gt_tokens)                    
        seq_input = torch.cat([start_embed, gt_embeds], dim=1)         

        # Thêm positional encoding 1D cho sequence
        seq_pos = self.pos_enc_1d(
            torch.arange(self.seq_len + 1, device=device)  
        ).unsqueeze(0).expand(B, -1, -1)
        seq_input = seq_input + seq_pos

        # Tạo causal mask
        causal_mask = self._generate_causal_mask(self.seq_len + 1, device) 

        # Decode
        decoder_out = self.decoder(
            seq_input,                         
            memory,                             
            tgt_mask=causal_mask,              
            memory_key_padding_mask=x_mask,    
        )  # [B, 5, 256]

        # Predict logits
        logits = self.predictor(decoder_out)

        # Compute loss
        per_token_loss = self.loss_fn(
            logits.reshape(-1, self.vocab_size),  
            targets.reshape(-1)                    
        )

        if self.token_weights is not None:
            weights = self.token_weights.repeat(B)
            per_token_loss = per_token_loss * weights

        loss = per_token_loss.mean()

        return loss

    @torch.no_grad()
    def forward_test(self, x_fused, img_metas):
        B = x_fused.shape[0]
        device = x_fused.device

        # Encode
        memory, x_mask, x_pos = self._encode(x_fused, img_metas)

        # Auto-regressive generation
        start_embed = torch.zeros(B, 1, self.d_model, device=device)
        seq_input = start_embed
        output_tokens = []

        for step in range(self.seq_len):  
            cur_len = seq_input.shape[1]
            seq_pos = self.pos_enc_1d(
                torch.arange(cur_len, device=device)
            ).unsqueeze(0).expand(B, -1, -1)
            seq_with_pos = seq_input + seq_pos

            # Causal mask
            causal_mask = self._generate_causal_mask(cur_len, device)

            # Decode
            decoder_out = self.decoder(
                seq_with_pos,
                memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=x_mask,
            )

            # Predict token tại vị trí cuối
            logits = self.predictor(decoder_out[:, -1, :]) 
            next_token = logits.argmax(dim=-1)               

            output_tokens.append(next_token)

            # Thêm token mới vào sequence cho step tiếp theo
            next_embed = self.token_embedding(next_token).unsqueeze(1) 
            seq_input = torch.cat([seq_input, next_embed], dim=1)

        # Stack tokens và dequantize
        pred_tokens = torch.stack(output_tokens, dim=1) 
        pred_bbox = dequantize_bbox(pred_tokens, img_metas, self.num_bin)

        return pred_bbox


# # TEST
# if __name__ == "__main__":
#     print("Test SeqHead")

#     head = SeqHead(
#         in_ch=1024, d_model=256, nhead=8, dim_feedforward=1024,
#         dropout=0.1, enc_layers=6, dec_layers=3,
#         num_bin=1000, label_smoothing=0.1
#     )

#     total = sum(p.numel() for p in head.parameters())
#     print(f"Params: {total:,}")

#     # Giả lập input
#     B = 2
#     x_fused = torch.randn(B, 1024, 40, 40)
#     gt_bbox = torch.tensor([[100.0, 50.0, 400.0, 300.0],
#                              [200.0, 100.0, 500.0, 400.0]])
#     img_metas = [
#         {'pad_shape': (640, 640, 3), 'img_shape': (480, 640, 3)},
#         {'pad_shape': (640, 640, 3), 'img_shape': (640, 480, 3)},
#     ]

#     # Test training
#     print("Test forward_train")
#     loss = head.forward_train(x_fused, gt_bbox, img_metas)
#     print(f"Loss: {loss.item():.4f}")

#     # Test inference
#     print("Test forward_test")
#     pred_bbox = head.forward_test(x_fused, img_metas)
#     print(f"Predicted bbox: {pred_bbox}")
#     print(f"Shape: {pred_bbox.shape}")  # [2, 4]

#     print("SeqHead test passed")
