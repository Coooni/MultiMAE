import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalEncoderWrapper(nn.Module):
    def __init__(self, base_model, embed_dim=768, num_classes=3):
        super().__init__()
        self.base = base_model
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.decoder = None  # lazy init

    def forward(self, x_dict_seq):
        """
        x_dict_seq: {'s1': [B,T,C,H,W], 's2':[B,T,C,H,W]}
        """
        B, T = x_dict_seq['s1'].shape[:2]
        temporal_feats = []

        for t in range(T):
            input_t = {k: v[:, t] for k, v in x_dict_seq.items()}  # {'s1':[B,C,H,W], 's2':[B,C,H,W]}

            # ✅ 1️⃣ 각 modality adapter → patch tokens 얻기
            token_list = []
            for domain, tensor in input_t.items():
                adapter = self.base.input_adapters[domain]
                tokens = adapter(tensor)  # [B, N_patches, D]
                token_list.append(tokens)
            
            x_tokens = torch.cat(token_list, dim=1)  # [B, N_total, D]

            # ✅ 2️⃣ Encoder 통과
            encoded = self.base.encoder(x_tokens)  # [B, N_tokens, D]
            temporal_feats.append(encoded)         # keep full token maps

        # ✅ 3️⃣ 시간 평균
        fused_feat = torch.stack(temporal_feats, dim=1).mean(dim=1)  # [B, N_tokens, D]

        # ✅ 4️⃣ 토큰 → 2D feature map 복원
        # assuming patch_size = 16 and image_size = 224 → 14x14 tokens
        H_toks = W_toks = int(fused_feat.shape[1] ** 0.5)  # e.g. 196 → 14x14
        feat_2d = fused_feat.transpose(1, 2).reshape(B, self.embed_dim, H_toks, W_toks)  # [B, D, H', W']

        # ✅ 5️⃣ Decoder lazy init (segmentation head)
        if self.decoder is None:
            print(f"⚙️ Initializing segmentation decoder: ConvTranspose2d({self.embed_dim} → {self.num_classes})")
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(self.embed_dim, 256, kernel_size=2, stride=2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, self.num_classes, kernel_size=1)
            ).to(feat_2d.device)

        # ✅ 6️⃣ Segmentation map 생성
        out = self.decoder(feat_2d)  # [B, num_classes, H, W]
        out = F.interpolate(out, size=(224, 224), mode="bilinear", align_corners=False)  # upsample to full res

        return {"cdl": out}
