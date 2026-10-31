"""DRISHTI - Official HMR 2.0 / 4DHumans Architecture & Inference Engine

Implementation of HMR 2.0 (4DHumans: ViT-Huge Backbone + SMPL Cross-Attention Decoder).
Compatible with official pretrained checkpoint (epoch=35-step=1000000.ckpt).
Runs on Apple Silicon MPS, NVIDIA CUDA, and CPU.
"""

from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.layers import drop_path, to_2tuple, trunc_normal_

from app.core.logging import logger


# ---------------------------------------------------------------------------
# Rotation representations
# ---------------------------------------------------------------------------

def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """Official 4DHumans / PyTorch3D 6D continuous rotation to 3x3 rotation matrix.
    Reference: Zhou et al., CVPR 2019; 4D-Humans (Goel et al., 2023)
    """
    x = x.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1, dim=-1)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def aa_to_rotmat(theta: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle representation to rotation matrix."""
    norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    norm_quat = quat / quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rot_mat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
        2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
        2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2
    ], dim=1).view(B, 3, 3)
    return rot_mat


# ---------------------------------------------------------------------------
# ViT Backbone (ViT-Huge from 4DHumans)
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    def __init__(self, drop_prob: Optional[float] = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None, act_layer: Any = nn.GELU, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ViTAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, qk_scale: Optional[float] = None, attn_drop: float = 0.0, proj_drop: float = 0.0, attn_head_dim: Optional[int] = None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = False, qk_scale: Optional[float] = None, drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0, act_layer: Any = nn.GELU, norm_layer: Any = nn.LayerNorm, attn_head_dim: Optional[int] = None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ViTAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size: Tuple[int, int] = (256, 192), patch_size: int = 16, in_chans: int = 3, embed_dim: int = 1280, ratio: int = 1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio ** 2)
        self.patch_shape = (int(img_size[0] // patch_size[0] * ratio), int(img_size[1] // patch_size[1] * ratio))
        self.origin_patch_shape = (int(img_size[0] // patch_size[0]), int(img_size[1] // patch_size[1]))
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        pad = 4 + 2 * (ratio // 2 - 1)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=(patch_size[0] // ratio), padding=pad)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)


class ViT(nn.Module):
    def __init__(
        self,
        img_size: Tuple[int, int] = (256, 192),
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 0,
        embed_dim: int = 1280,
        depth: int = 32,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.55,
        norm_layer: Any = None,
        ratio: int = 1,
        last_norm: bool = True,
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, ratio=ratio)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        dpr = [float(x.item()) for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            ViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)

        if self.pos_embed is not None:
            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

        for blk in self.blocks:
            x = blk(x)

        x = self.last_norm(x)
        xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()
        return xp


# ---------------------------------------------------------------------------
# Transformer Decoder Head (SMPL Parameter Head from 4DHumans)
# ---------------------------------------------------------------------------

class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: Callable):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), *args, **kwargs)


class DecoderFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if not (heads == 1 and dim_head == dim) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class DecoderCrossAttention(nn.Module):
    def __init__(self, dim: int, context_dim: int = 1280, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if not (heads == 1 and dim_head == dim) else nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q = self.to_q(x)
        q, k, v = map(lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.heads), [q, k, v])
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class TransformerCrossAttn(nn.Module):
    def __init__(self, dim: int = 1024, depth: int = 6, heads: int = 8, dim_head: int = 64, mlp_dim: int = 1024, dropout: float = 0.0, context_dim: int = 1280):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            sa = DecoderSelfAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ca = DecoderCrossAttention(dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ff = DecoderFeedForward(dim, mlp_dim, dropout=dropout)
            self.layers.append(
                nn.ModuleList([
                    PreNorm(dim, sa),
                    PreNorm(dim, ca),
                    PreNorm(dim, ff),
                ])
            )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        for self_attn, cross_attn, ff in self.layers:
            x = self_attn(x) + x
            x = cross_attn(x, context=context) + x
            x = ff(x) + x
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_tokens: int = 1, token_dim: int = 1, dim: int = 1024, depth: int = 6, heads: int = 8, mlp_dim: int = 1024, dim_head: int = 64, dropout: float = 0.0, context_dim: int = 1280):
        super().__init__()
        self.to_token_embedding = nn.Linear(token_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))
        self.dropout = nn.Dropout(dropout)
        self.transformer = TransformerCrossAttn(dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout, context_dim=context_dim)

    def forward(self, inp: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.to_token_embedding(inp)
        b, n, _ = x.shape
        x = self.dropout(x)
        x = x + self.pos_embedding[:, :n]
        x = self.transformer(x, context=context)
        return x


class SMPLTransformerDecoderHead(nn.Module):
    """Official 4DHumans SMPL Transformer Decoder Head."""

    def __init__(self, mean_params_path: Optional[Path] = None):
        super().__init__()
        dim = 1024
        npose = 144  # 24 joints * 6D continuous rotation
        self.transformer = TransformerDecoder(num_tokens=1, token_dim=1, dim=dim, depth=6, heads=8, mlp_dim=1024, dim_head=64, dropout=0.0, context_dim=1280)
        self.decpose = nn.Linear(dim, npose)
        self.decshape = nn.Linear(dim, 10)
        self.deccam = nn.Linear(dim, 3)

        # Load mean SMPL parameters
        if mean_params_path and mean_params_path.exists():
            mean_data = np.load(str(mean_params_path))
            init_pose = torch.from_numpy(mean_data["pose"].astype(np.float32)).unsqueeze(0)
            init_shape = torch.from_numpy(mean_data["shape"].astype(np.float32)).unsqueeze(0)
            init_cam = torch.from_numpy(mean_data["cam"].astype(np.float32)).unsqueeze(0)
        else:
            init_pose = torch.zeros(1, 144, dtype=torch.float32)
            init_shape = torch.zeros(1, 10, dtype=torch.float32)
            init_cam = torch.tensor([[0.9, 0.0, 0.0]], dtype=torch.float32)

        self.register_buffer("init_body_pose", init_pose)
        self.register_buffer("init_betas", init_shape)
        self.register_buffer("init_cam", init_cam)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Backbone spatial feature maps of shape (B, 1280, 16, 12)
        Returns:
            Tuple of (pred_body_pose_6d (B, 144), pred_betas (B, 10), pred_cam (B, 3))
        """
        batch_size = x.shape[0]
        context = einops.rearrange(x, "b c h w -> b (h w) c")

        # Zero input token for single-person decoder
        token = torch.zeros(batch_size, 1, 1, device=x.device, dtype=x.dtype)
        token_out = self.transformer(token, context=context).squeeze(1)

        pred_body_pose = self.decpose(token_out) + self.init_body_pose.expand(batch_size, -1)
        pred_betas = self.decshape(token_out) + self.init_betas.expand(batch_size, -1)
        pred_cam = self.deccam(token_out) + self.init_cam.expand(batch_size, -1)

        return pred_body_pose, pred_betas, pred_cam


# ---------------------------------------------------------------------------
# Full HMR 2.0 Network
# ---------------------------------------------------------------------------

class HMR2Network(nn.Module):
    """Complete HMR 2.0 network combining ViT-Huge backbone and SMPL Transformer Decoder Head."""

    def __init__(self, mean_params_path: Optional[Path] = None):
        super().__init__()
        self.backbone = ViT(
            img_size=(256, 192),
            patch_size=16,
            embed_dim=1280,
            depth=32,
            num_heads=16,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path_rate=0.55,
        )
        self.smpl_head = SMPLTransformerDecoderHead(mean_params_path=mean_params_path)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Executes forward pass.
        
        Args:
            x: Preprocessed input image tensor of shape (B, 3, 256, 256)
               Normalized by ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
               
        Returns:
            Dict containing:
                'pose_6d': (B, 144) 6D rotation representation for all 24 joints
                'betas': (B, 10) SMPL shape blend shape coefficients
                'pred_cam': (B, 3) weak-perspective camera [scale, tx, ty]
                'rotmat': (B, 24, 3, 3) full rotation matrices for all 24 joints
                'global_orient': (B, 1, 3, 3) root orientation
                'body_pose': (B, 23, 3, 3) relative poses of the 23 articulated joints
        """
        # Crop central 192 columns to match 256x192 (4:3) ViT aspect ratio
        x_cropped = x[:, :, :, 32:-32]
        conditioning_feats = self.backbone(x_cropped)

        pred_pose_6d, pred_betas, pred_cam = self.smpl_head(conditioning_feats)
        batch_size = pred_pose_6d.shape[0]

        # Convert 6D rotation representations to 3x3 rotation matrices
        rotmat = rot6d_to_rotmat(pred_pose_6d.view(-1, 6)).view(batch_size, 24, 3, 3)

        return {
            "pose_6d": pred_pose_6d,
            "betas": pred_betas,
            "pred_cam": pred_cam,
            "rotmat": rotmat,
            "global_orient": rotmat[:, [0]],
            "body_pose": rotmat[:, 1:],
        }

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Path,
        mean_params_path: Optional[Path] = None,
        device: torch.device = torch.device("cpu"),
    ) -> "HMR2Network":
        """Loads weights from the official PyTorch Lightning checkpoint."""
        logger.info(f"Instantiating HMR 2.0 network on device: {device}...")
        model = cls(mean_params_path=mean_params_path)

        logger.info(f"Loading weights from {checkpoint_path}...")
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Strip Lightning prefixes ('backbone.', 'smpl_head.')
        model_state = model.state_dict()
        filtered_dict = {}
        matched_keys = 0
        total_keys = len(model_state)

        for k, v in state_dict.items():
            # Skip discriminator or loss buffers
            if k.startswith("discriminator.") or "loss" in k:
                continue
            if k in model_state:
                filtered_dict[k] = v
                matched_keys += 1
            else:
                # Try prefix matches
                for prefix in ["model.", "hmr2."]:
                    if k.startswith(prefix) and k[len(prefix):] in model_state:
                        filtered_dict[k[len(prefix):]] = v
                        matched_keys += 1
                        break

        missing, unexpected = model.load_state_dict(filtered_dict, strict=False)
        logger.info(
            f"HMR 2.0 state_dict loaded: {matched_keys}/{total_keys} matched keys "
            f"({len(missing)} missing, {len(unexpected)} unexpected ignored)."
        )

        model.to(device)
        model.eval()
        return model
