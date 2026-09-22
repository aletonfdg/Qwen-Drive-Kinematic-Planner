"""
model.py — UnifiedE2EModel v11
"""

import torch
import torch.nn as nn


class DifferentiableKinematicPlanner(nn.Module):
    """
    Unicycle model (point-mass + yaw rate). Predicts control signals
    (acceleration, yaw rate) and integrates them into (x, y, yaw).

    Limits: acceleration in [-6.0, +4.0] m/s^2 (asymmetric),
            yaw rate in [-1.5, +1.5] rad/s.
    """

    def __init__(self, in_features: int, points: int = 50, dt: float = 0.1):
        super().__init__()
        self.points = points
        self.dt = dt

        self.dyn_mlp = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, points * 2),
        )

    def forward(self, feat: torch.Tensor, v0_mps: torch.Tensor) -> torch.Tensor:
        B = feat.size(0)
        out = self.dyn_mlp(feat).view(B, self.points, 2)

        raw_accel = torch.tanh(out[:, :, 0])
        accels = torch.where(raw_accel >= 0, raw_accel * 4.0, raw_accel * 6.0)
        yaw_rates = torch.tanh(out[:, :, 1]) * 1.5

        v_profile = torch.clamp(
            v0_mps.unsqueeze(1) + torch.cumsum(accels * self.dt, dim=1), min=0.0
        )
        yaw_profile = torch.cumsum(yaw_rates * self.dt, dim=1)

        dx = v_profile * torch.cos(yaw_profile) * self.dt
        dy = v_profile * torch.sin(yaw_profile) * self.dt

        x_pts = torch.cumsum(dx, dim=1)
        y_pts = torch.cumsum(dy, dim=1)

        return torch.stack([x_pts, y_pts, yaw_profile], dim=-1)


class UnifiedE2EModel(nn.Module):
    """
    Full v11 pipeline: VLM (Qwen-Drive-1.0 backbone + LoRA) -> Cross-Attention
    Scene Pool -> Fusion with CAN telemetry -> Kinematic Planner + Controller Head.

    Args:
        vlm: LoRA-wrapped VLM module (typically peft.PeftModel wrapping
             base_model.vlm from QwenDriveForPlanning.from_pretrained(...))
        hidden_dim: scene pooling embedding size (default 512)
        traj_points: number of trajectory steps (default 50 -> 5.0s @ dt=0.1)
        traj_dt: integration timestep in seconds
        vlm_hidden_size: VLM hidden size (auto-detected from vlm.config if None)
    """

    def __init__(
        self,
        vlm,
        hidden_dim: int = 512,
        traj_points: int = 50,
        traj_dt: float = 0.1,
        vlm_hidden_size: int = None,
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.vlm = vlm
        self.compute_dtype = compute_dtype

        if vlm_hidden_size is None:
            vlm_cfg = getattr(vlm, "config", None)
            if hasattr(vlm_cfg, "text_config") and hasattr(vlm_cfg.text_config, "hidden_size"):
                vlm_hidden_size = vlm_cfg.text_config.hidden_size
            elif hasattr(vlm_cfg, "hidden_size"):
                vlm_hidden_size = vlm_cfg.hidden_size
            else:
                vlm_hidden_size = 2560  # Qwen-Drive-1.0-4B fallback

        self.scene_proj = nn.Sequential(
            nn.Linear(vlm_hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.query_embed = nn.Parameter(torch.randn(4, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # CAN telemetry: [speed_kmh, accel_norm] -> 128-dim
        self.dyn_embed = nn.Sequential(
            nn.Linear(2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        fusion_dim = hidden_dim + 128  # 512 + 128 = 640

        self.kinematic_planner = DifferentiableKinematicPlanner(
            in_features=fusion_dim,
            points=traj_points,
            dt=traj_dt,
        )

        self.controller = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),  # [speed_norm, steer_norm]
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        can_state,
        mm_token_type_ids=None,
        **kwargs,
    ):
        """
        can_state: [B, 2] = (speed_kmh, accel_norm), accel_norm clipped to [-5, 5].
        NOTE: speed is in km/h, not m/s and not normalized.
        """
        pv = pixel_values.to(dtype=self.compute_dtype)

        outputs = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pv,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden = outputs.hidden_states[-1].float()
        token_features = self.scene_proj(last_hidden)

        b = token_features.size(0)
        queries = self.query_embed.unsqueeze(0).expand(b, -1, -1)
        key_padding_mask = (attention_mask == 0)
        attn_out, _ = self.cross_attn(
            query=queries,
            key=token_features,
            value=token_features,
            key_padding_mask=key_padding_mask,
        )
        pooled_scene = self.cross_norm(attn_out).mean(dim=1)

        can_feat = self.dyn_embed(can_state.float())
        fusion_feat = torch.cat([pooled_scene, can_feat], dim=-1)

        v0_mps = can_state[:, 0] / 3.6  # km/h -> m/s

        trajectory_meters = self.kinematic_planner(fusion_feat, v0_mps)
        controls = self.controller(fusion_feat)

        return {"trajectory": trajectory_meters, "controls": controls}