import torch
import torch.nn as nn
from .backbone import VisualEncoder
from .language import LanguageEncoder
from .fusion import SimpleFusion
from .transformer import SeqHead


class SeqTRDet(nn.Module):
    def __init__(self, config, glove_vectors):
        super().__init__()

        # Visual Encoder
        self.vis_enc = VisualEncoder(freeze_layers=True)

        # Language Encoder
        self.lan_enc = LanguageEncoder(
            glove_vectors=glove_vectors,
            hidden_size=config.gru_hidden, 
            pooling=config.pooling,
        )

        # Fusion
        self.fusion = SimpleFusion(
            vis_channels=[512, 1024, 2048]
        )

        # Sequence Head
        self.head = SeqHead(
            in_ch=config.backbone_out_channels,  # 1024
            d_model=config.d_model,               # 256
            nhead=config.nhead,                    # 8
            dim_feedforward=config.dim_feedforward, # 1024
            dropout=config.dropout,                # 0.1
            enc_layers=config.enc_layers,          # 6
            dec_layers=config.dec_layers,          # 3
            num_bin=config.num_bin,                 # 1000
            label_smoothing=config.label_smoothing, # 0.1
            token_weights=config.token_weights,      # Per-token weights (ablation)
        )

    def forward(self, img, ref_inds, img_shapes, gt_bbox=None):
        B = img.shape[0]
        img_metas = []
        for i in range(B):
            img_metas.append({
                'pad_shape': (int(img_shapes[i, 0]), int(img_shapes[i, 1]), 3),
                'img_shape': (int(img_shapes[i, 2]), int(img_shapes[i, 3]), 3),
            })

        # Trích xuất visual features
        vis_feats = self.vis_enc(img)  

        # Mã hóa câu mô tả -> 1 vector
        lang_feat = self.lan_enc(ref_inds)  

        # Kết hợp visual + language
        x_fused = self.fusion(vis_feats, lang_feat)  

        # Sinh tọa độ bbox
        if gt_bbox is not None:
            loss = self.head.forward_train(x_fused, gt_bbox, img_metas)
            return loss
        else:
            pred_bbox = self.head.forward_test(x_fused, img_metas)
            return pred_bbox

# if __name__ == "__main__":
#     import sys
#     sys.path.insert(0, '.')
#     from config import Config

#     print("Test SeqTRDet (Full Model)")

#     vocab_size = 100
#     fake_glove = torch.randn(vocab_size, Config.glove_dim)
#     fake_glove[0] = 0  

#     model = SeqTRDet(Config, fake_glove)

#     total = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     frozen = total - trainable
#     print(f"Total params:     {total:,}")
#     print(f"Trainable params: {trainable:,}")
#     print(f"Frozen params:    {frozen:,}")

#     B = 2
#     img = torch.randn(B, 3, 640, 640)
#     ref_inds = torch.randint(1, vocab_size, (B, Config.max_token))
#     gt_bbox = torch.tensor([[100.0, 50.0, 400.0, 300.0],
#                              [200.0, 100.0, 500.0, 400.0]])
#     img_metas = [
#         {'pad_shape': (640, 640, 3), 'img_shape': (480, 640, 3)},
#         {'pad_shape': (640, 640, 3), 'img_shape': (640, 480, 3)},
#     ]

#     print("Training mode")
#     model.train()
#     loss = model(img, ref_inds, img_metas, gt_bbox=gt_bbox)
#     print(f"Loss: {loss.item():.4f}")

#     loss.backward()
#     grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
#     print(f"Parameters with gradient: {grad_count}")

#     print("Inference mode")
#     model.eval()
#     pred_bbox = model(img, ref_inds, img_metas, gt_bbox=None)
#     print(f"Predicted bbox: {pred_bbox}")
#     print(f"Shape: {pred_bbox.shape}") 

#     print("SeqTRDet full model test passed")
