#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor for Robot Vacuum.
清扫大作战特征预处理器。
"""

import numpy as np

from agent_ppo.reward.reward_charge import RewardCharge
from agent_ppo.reward.reward_context import RewardContext
from agent_ppo.reward.reward_explore import RewardExplore


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值线性归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    if v_max == v_min:
        return 0.0
    return (v - v_min) / (v_max - v_min)


class Preprocessor:
    """Feature preprocessor for Robot Vacuum.

    清扫大作战特征预处理器。
    """

    GRID_SIZE = 128
    VIEW_HALF = 21  # Full local view radius (21×21) / 完整局部视野半径
    LOCAL_HALF = 5  # Cropped view radius (11×11) / 裁剪后的视野半径

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all internal state at episode start.

        对局开始时重置所有状态。
        """
        self.step_no = 0
        self.battery = 600
        self.battery_max = 600

        self.cur_pos = (0, 0)

        self.dirt_cleaned = 0
        self.last_dirt_cleaned = 0
        self.total_dirt = 1

        # Global passable map (0=obstacle, 1=passable), used for ray computation
        # 维护全局通行地图（0=障碍, 1=可通行），用于射线计算
        self.passable_map = np.ones((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)

        # Nearest dirt distance
        # 最近污渍距离
        self.nearest_dirt_dist = 200.0
        self.last_nearest_dirt_dist = 200.0

        self._view_map = np.zeros((21, 21), dtype=np.float32)
        self._legal_act = [1] * 8

        # 自己维护的指标 用于reward构造
        # 最近的充电桩距离
        self.last_charge_distance = 200.0
        self.charge_distance = 200.0
        self.charge_dis_delta = 0  #0 代表更近 1 代表更远   
        self.charge_dir = np.zeros(8, dtype=np.float32)
        self.reward_charge = RewardCharge()
        self.reward_explore = RewardExplore()


    def pb2struct(self, env_obs, last_action):
        """Parse and cache essential fields from observation dict.

        从 env_obs 字典中提取并缓存所有需要的状态量。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        hero = frame_state["heroes"]

        self.step_no = int(observation["step_no"])
        self.cur_pos = (int(hero["pos"]["x"]), int(hero["pos"]["z"]))

        # Battery / 电量
        self.battery = int(hero["battery"])
        self.battery_max = max(int(hero["battery_max"]), 1)

        # Cleaning progress / 清扫进度
        self.last_dirt_cleaned = self.dirt_cleaned
        self.dirt_cleaned = int(hero["dirt_cleaned"])
        self.total_dirt = max(int(env_info["total_dirt"]), 1)

        # Legal actions / 合法动作
        self._legal_act = [int(x) for x in (observation.get("legal_action") or [1] * 8)]

        # Local view map (21×21) / 局部视野地图
        map_info = observation.get("map_info")
        if map_info is not None:
            self._view_map = np.array(map_info, dtype=np.float32)
            hx, hz = self.cur_pos
            self._update_passable(hx, hz)

    def _get_nearest_dir(self):
        """Placeholder for nearest-direction helper.

        最近方向辅助函数预留接口。
        """
        return None

    def _update_passable(self, hx, hz):
        """Write local view into global passable map.

        将局部视野写入全局通行地图。
        """
        view = self._view_map
        vsize = view.shape[0]
        half = vsize // 2

        for ri in range(vsize):
            for ci in range(vsize):
                gx = hx - half + ri
                gz = hz - half + ci
                if 0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE:
                    # 0 = obstacle, 1/2 = passable
                    # 0 = 障碍, 1/2 = 可通行
                    self.passable_map[gx, gz] = 1 if view[ri, ci] != 0 else 0

    def _get_local_view_feature(self):
        """Local view feature (121D): crop center 11×11 from 21×21.

        局部视野特征（121D）：从 21×21 视野中心裁剪 11×11。
        """
        center = self.VIEW_HALF
        h = self.LOCAL_HALF
        crop = self._view_map[center - h : center + h + 1, center - h : center + h + 1]
        return (crop / 2.0).flatten()

    def _get_view_cell(self, x, z, hx, hz):
        """Get cell value from local view by global coordinate.

        根据全局坐标读取局部视野中的格子值。
        """
        if self._view_map is None:
            return 0

        view_x = x - hx + self.VIEW_HALF
        view_z = z - hz + self.VIEW_HALF
        if not (0 <= view_x < 21 and 0 <= view_z < 21):
            return 0

        return int(self._view_map[view_x, view_z])

    def _calc_ray_dirt_features(self, hx, hz, max_ray=30):
        """Find normalized nearest dirt distance along 8 rays.

        计算八方向射线上最近污渍距离，并归一化。
        """
        ray_dirs = [
            (0, -1),
            (1, 0),
            (0, 1),
            (-1, 0),
        ]  # N E S W
        ray_dirt = []

        for dx, dz in ray_dirs:
            x, z = hx, hz
            found = max_ray
            for step in range(1, max_ray + 1):
                x += dx
                z += dz
                if not (0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE):
                    break
                if self._get_view_cell(x, z, hx, hz) == 2:
                    found = step
                    break
            ray_dirt.append(_norm(found, max_ray))

        return ray_dirt

    def _update_nearest_dirt_metrics(self):
        """Update nearest dirt distance and approaching indicator.

        更新最近污渍距离及是否接近污渍标记。
        """
        self.last_nearest_dirt_dist = self.nearest_dirt_dist
        self.nearest_dirt_dist = self._calc_nearest_dirt_dist()
        nearest_dirt_norm = _norm(self.nearest_dirt_dist, 180)
        dirt_delta = 1.0 if self.nearest_dirt_dist < self.last_nearest_dirt_dist else 0.0
        return nearest_dirt_norm, dirt_delta

    def _get_global_state_feature(self):
        """Global state feature (12D).

        全局状态特征（12D）。

        Dimensions / 维度说明：
          [0]  step_norm         step progress / 步数归一化 [0,1]
          [1]  battery_ratio     battery level / 电量比 [0,1]
          [2]  cleaning_progress cleaned ratio / 已清扫比例 [0,1]
          [3]  remaining_dirt    remaining dirt ratio / 剩余污渍比例 [0,1]
          [4]  pos_x_norm        x position / x 坐标归一化 [0,1]
          [5]  pos_z_norm        z position / z 坐标归一化 [0,1]
          [6]  ray_N_dirt        north ray distance / 向上（z-）方向最近污渍距离
          [7]  ray_E_dirt        east ray distance / 向右（x+）方向
          [8]  ray_S_dirt        south ray distance / 向下（z+）方向
          [9]  ray_W_dirt        west ray distance / 向左（x-）方向
          [10] nearest_dirt_norm nearest dirt Euclidean distance / 最近污渍欧氏距离归一化
          [11] dirt_delta        approaching dirt indicator / 是否在接近污渍（1=是, 0=否）
        """
        step_norm = _norm(self.step_no, 2000)
        battery_ratio = _norm(self.battery, self.battery_max)
        cleaning_progress = _norm(self.dirt_cleaned, self.total_dirt)
        remaining_dirt = 1.0 - cleaning_progress

        hx, hz = self.cur_pos
        pos_x_norm = _norm(hx, self.GRID_SIZE)
        pos_z_norm = _norm(hz, self.GRID_SIZE)

        # 4-directional ray to find nearest dirt
        # 四方向射线找最近污渍距离 
        ray_dirt = self._calc_ray_dirt_features(hx, hz)

        # Nearest dirt Euclidean distance (estimated from cropped local view)
        # 最近污渍欧氏距离（基于裁剪后的局部视野粗估）
        nearest_dirt_norm, dirt_delta = self._update_nearest_dirt_metrics()

        return np.array(
            [
                step_norm,
                battery_ratio,
                cleaning_progress,
                remaining_dirt,
                pos_x_norm,
                pos_z_norm,
                ray_dirt[0],
                ray_dirt[1],
                ray_dirt[2],
                ray_dirt[3],
                nearest_dirt_norm,
                dirt_delta,
            ],
            dtype=np.float32,
        )

    def _calc_nearest_dirt_dist(self):
        """Find nearest dirt Euclidean distance from local view.

        从局部视野中找最近污渍的欧氏距离。
        """
        view = self._view_map
        if view is None:
            return 200.0
        dirt_coords = np.argwhere(view == 2)
        if len(dirt_coords) == 0:
            return 200.0
        center = self.VIEW_HALF
        dists = np.sqrt((dirt_coords[:, 0] - center) ** 2 + (dirt_coords[:, 1] - center) ** 2)
        return float(np.min(dists))

    def get_legal_action(self):
        """Return legal action mask (8D list).

        返回合法动作掩码（8D list）。
        """
        return list(self._legal_act)

    def _build_reward_context(self, local_view, global_state, legal_action, legal_arr, feature):
        """Build reward context from current preprocessor state.

        基于当前预处理器状态构建 reward context。
        """
        context = RewardContext()
        context.step_no = self.step_no
        context.battery = self.battery
        context.battery_max = self.battery_max
        context.cur_pos = self.cur_pos
        context.dirt_cleaned = self.dirt_cleaned
        context.last_dirt_cleaned = self.last_dirt_cleaned
        context.total_dirt = self.total_dirt

        context.passable_map = np.array(self.passable_map, copy=True)
        context.nearest_dirt_dist = self.nearest_dirt_dist
        context.last_nearest_dirt_dist = self.last_nearest_dirt_dist
        context.dirt_delta = float(global_state[11])

        context._view_map = np.array(self._view_map, copy=True)
        context._legal_act = list(self._legal_act)

        context.local_view = np.array(local_view, copy=True)
        context.global_state = np.array(global_state, copy=True)
        context.legal_action = list(legal_action)
        context.legal_arr = np.array(legal_arr, copy=True)
        context.feature = np.array(feature, copy=True)

        context.charge_distance = self.charge_distance
        context.charge_dis_delta = self.charge_dis_delta
        context.charge_dir = np.array(self.charge_dir, copy=True)
        return context

    def feature_process(self, env_obs, last_action):
        """Generate 141D feature vector, legal action mask, and scalar reward.

        生成 141D 特征向量、合法动作掩码和标量奖励。
        """
        self.pb2struct(env_obs, last_action)

        local_view = self._get_local_view_feature()  # 121D
        global_state = self._get_global_state_feature()  # 12D
        legal_action = self.get_legal_action()  # 8D
        legal_arr = np.array(legal_action, dtype=np.float32)

        feature = np.concatenate([local_view, global_state, legal_arr])  # 141D

        reward_context = self._build_reward_context(local_view, global_state, legal_action, legal_arr, feature)
        reward = self.reward_process(reward_context)

        return feature, legal_action, reward

    def reward_process(self, context):
        """Compute reward with prepared reward context.

        基于构建好的 reward context 计算奖励。
        """
        total_reward = 0.0
        total_reward += self.reward_charge.get_reward(context)
        total_reward += self.reward_explore.get_reward(context)
        return total_reward
