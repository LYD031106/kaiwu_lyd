#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Simple MLP policy network for Robot Vacuum.
"""

import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config


def _make_fc(in_dim, out_dim, gain=1.41421):
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "robot_vacuum"
        self.device = device

        obs_dim = Config.DIM_OF_OBSERVATION
        act_num = Config.ACTION_NUM

        self.backbone = nn.Sequential(
            _make_fc(obs_dim, 256),
            nn.ReLU(),
            _make_fc(256, 128),
            nn.ReLU(),
        )

        self.actor_head = _make_fc(128, act_num, gain=0.01)
        self.critic_head = _make_fc(128, 1, gain=0.01)

    def forward(self, s, inference=False):
        x = s.to(torch.float32)
        h = self.backbone(x)
        logits = self.actor_head(h)
        value = self.critic_head(h)
        return [logits, value]

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
