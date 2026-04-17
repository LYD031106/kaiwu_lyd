#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Robot Vacuum PPO agent.
"""


class Config:
    LOCAL_VIEW_SIZE = 21 * 21
    GLOBAL_FEATURE_SIZE = 38

    # Full 21x21 local view + structured state + legal action mask
    FEATURES = [
        LOCAL_VIEW_SIZE,
        GLOBAL_FEATURE_SIZE,
        8,
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURES)
    DIM_OF_OBSERVATION = FEATURE_LEN

    ACTION_NUM = 8
    VALUE_NUM = 1

    GAMMA = 0.99
    LAMDA = 0.95

    INIT_LEARNING_RATE_START = 0.0003
    BETA_START = 0.001
    CLIP_PARAM = 0.2
    VF_COEF = 0.5

    LABEL_SIZE_LIST = [ACTION_NUM]
    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()

    # 训练模型加载控制
    LOAD_MODEL_SWITCH = False  # 是否开启读取 checkpoint 的逻辑
    LOAD_MODEL_PATH = ""  # 读取的 checkpoint 文件名路径 (如果为空且开关为True，则走原本的 default path)

    # 训练模型保存控制
    SAVE_MODEL_SWITCH = True  # 是否开启保存 checkpoint 的逻辑
    SAVE_MODEL_PATH = ""  # 保存的 checkpoint 文件名路径 (如果为空且开关为True，则走原本的 default path)

