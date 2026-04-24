#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

CNN + MLP actor-critic network for Robot Vacuum.
清扫大作战 CNN + MLP 双头策略网络。
"""

import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config


def _make_fc(in_dim, out_dim, gain=1.41421):
    """Create a linear layer with orthogonal initialization.

    创建正交初始化的线性层。
    """
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class Model(nn.Module):
    """CNN + MLP actor-critic model for Robot Vacuum.

    清扫大作战 CNN + MLP 双头策略网络。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "robot_vacuum"
        self.device = device

        obs_dim = Config.DIM_OF_OBSERVATION  # 269
        act_num = Config.ACTION_NUM  # 8
        local_dim, global_dim, legal_dim = Config.FEATURE_SPLIT_SHAPE
        self.local_map_hw = int(local_dim**0.5)
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.legal_dim = legal_dim

        # Local map encoder / 局部地图编码器（15x15 -> feature）
        self.local_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            _make_fc(32 * self.local_map_hw * self.local_map_hw, 64),
            nn.ReLU(),
        )

        # Global scalar encoder / 全局标量编码器
        self.global_encoder = nn.Sequential(
            _make_fc(global_dim, 32),
            nn.ReLU(),
        )

        # Legal-action encoder / 合法动作约束特征编码器
        self.legal_encoder = nn.Sequential(
            _make_fc(legal_dim, 16),
            nn.ReLU(),
        )

        fused_dim = 64 + 32 + 16

        # Shared fusion trunk / 共享融合层
        self.fusion = nn.Sequential(
            _make_fc(fused_dim, 128),
            nn.ReLU(),
            _make_fc(128, 64),
            nn.ReLU(),
        )

        # Actor branch / 策略分支
        self.actor_mlp = nn.Sequential(
            _make_fc(64, 64),
            nn.ReLU(),
        )
        self.actor_head = _make_fc(64, act_num, gain=0.01)

        # Critic branch / 价值分支
        self.critic_mlp = nn.Sequential(
            _make_fc(64, 64),
            nn.ReLU(),
        )
        self.critic_head = _make_fc(64, 1, gain=0.01)

    def forward(self, s, inference=False):
        """Forward pass.

        前向传播。
        """
        x = s.to(torch.float32)
        local_view = x[:, : self.local_dim].reshape(-1, 1, self.local_map_hw, self.local_map_hw)
        global_state = x[:, self.local_dim : self.local_dim + self.global_dim]
        legal_action = x[:, self.local_dim + self.global_dim : self.local_dim + self.global_dim + self.legal_dim]

        local_feat = self.local_encoder(local_view)
        global_feat = self.global_encoder(global_state)
        legal_feat = self.legal_encoder(legal_action)

        fused = torch.cat([local_feat, global_feat, legal_feat], dim=1)
        shared = self.fusion(fused)

        actor_hidden = self.actor_mlp(shared)
        critic_hidden = self.critic_mlp(shared)

        logits = self.actor_head(actor_hidden)
        value = self.critic_head(critic_hidden)
        return [logits, value]

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
