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

from collections import deque
import numpy as np

from agent_ppo.reward.reward_charge import RewardCharge
from agent_ppo.reward.reward_clean import RewardClean
from agent_ppo.reward.reward_context import RewardContext
from agent_ppo.reward.reward_explore import RewardExplore
from agent_ppo.reward.reward_npc import RewardNpc


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
    LOCAL_HALF = 7  # Cropped view radius (15×15) / 裁剪后的视野半径
    TARGET_MAX_DIST = 180.0
    RAY_DIRS_8 = [
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
    ]

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

        # Global passable map (0=obstacle, 1=passable, 3=charger), used for ray computation
        # 维护全局通行地图（0=障碍, 1=可通行, 3=充电桩），用于射线计算
        self.passable_map = np.ones((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)

        # Nearest dirt distance
        # 最近污渍距离
        self.nearest_dirt_dist = 200.0
        self.last_nearest_dirt_dist = 200.0
        self.nearest_charge_dist = 200.0
        self.last_nearest_charge_dist = 200.0

        self._view_map = np.zeros((21, 21), dtype=np.float32)
        self._legal_act = [1] * 8

        # 自己维护的指标 用于reward构造
        # 最近的充电桩距离
        self.last_charge_distance = 200.0
        self.charge_distance = 200.0
        self.charge_dis_delta = 0.0  # >0 代表更近，<0 代表更远，0 代表不变
        self.charge_dir = np.zeros(8, dtype=np.float32)

        self.charge_count = 0
        self.total_charger = 0
        self.charger_positions = []

        self.charging = False  # 是否正在充电中
        self.last_charging = False  # 上一个时间步是否正在充电


        # Reward components
        # 奖励组件
        self.reward_charge = RewardCharge()
        self.reward_clean = RewardClean()
        self.reward_explore = RewardExplore()
        self.reward_npc = RewardNpc()

        # Reward metrics for monitoring
        # 监控使用的分项奖励指标
        self.last_charge_reward = 0.0
        self.last_clean_reward = 0.0
        self.last_explore_reward = 0.0
        self.last_npc_reward = 0.0
        self.last_total_reward = 0.0
        self.episode_charge_reward = 0.0
        self.episode_clean_reward = 0.0
        self.episode_explore_reward = 0.0
        self.episode_npc_reward = 0.0
        self.episode_total_reward = 0.0

        # 里程碑奖励
        self.first_charge_reward = False

        # 维护最近3个时间段内的位置
        self.last_pos_queue = deque(maxlen=3)
        self.loop_pos = 0.0

        # npc 位置
        self.npc_positions = []
        self.last_npc_distance = 200.0  # 上一个时间步的 npc 距离
        self.close_npc_state = False  # 是否距离 npc 过近了


    def pb2struct(self, env_obs, last_action):
        """Parse and cache essential fields from observation dict.

        从 env_obs 字典中提取并缓存所有需要的状态量。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        hero = frame_state["heroes"]
        organs = frame_state.get("organs") or []

        # 单步事件标记：每一帧开始时先清零，只有本步满足条件时再置为 True
        self.first_charge_reward = False

        self.step_no = int(observation["step_no"])
        self.cur_pos = (int(hero["pos"]["x"]), int(hero["pos"]["z"]))

        # Battery / 电量
        self.battery = int(hero["battery"])
        self.battery_max = max(int(hero["battery_max"]), 1)

        # Cleaning progress / 清扫进度
        self.last_dirt_cleaned = self.dirt_cleaned
        self.dirt_cleaned = int(hero["dirt_cleaned"])
        self.total_dirt = max(int(env_info["total_dirt"]), 1)
        self.total_charger = int(env_info.get("total_charger", self.total_charger))

        # Update loop position
        self._update_loop_pos(self.cur_pos)

        # Legal actions / 合法动作
        self._legal_act = [int(x) for x in (observation.get("legal_action") or [1] * 8)]

        # Local view map (21×21) / 局部视野地图
        map_info = observation.get("map_info")
        if map_info is not None:
            self._view_map = np.array(map_info, dtype=np.float32)
            hx, hz = self.cur_pos
            self._update_passable(hx, hz)

        self._update_charger_state(organs)
        self._update_npc_state(self.cur_pos, frame_state.get("npcs") or [])

    def _update_loop_pos(self, cur_pos):
        """Update loop position.

        更新循环位置。
        """
        # 计算当前位置在最近历史窗口中重复出现的次数
        self.loop_pos = float(sum(1 for pos in self.last_pos_queue if pos == cur_pos))
        self.last_pos_queue.append(cur_pos)


    def _update_charger_state(self, organs):
        """Cache charger positions from observation organs.

        从环境 observation 的 organs 字段中缓存充电桩位置。
        """
        charger_positions = []
        for organ in organs:
            if int(organ.get("sub_type", 0)) != 1:
                continue

            pos = organ.get("pos") or {}
            x = int(pos.get("x", -1))
            z = int(pos.get("z", -1))
            if not (0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE):
                continue

            charger_positions.append((x, z))

        if charger_positions:
            self.charger_positions = charger_positions
            self.total_charger = max(self.total_charger, len(charger_positions))
            for x, z in charger_positions:
                self.passable_map[x, z] = 3

        self.last_charging = self.charging
        self.charging = self.cur_pos in self.charger_positions

    
    def _update_npc_state(self, cur_pos, npcs):
        """Cache npc positions from observation npcs.

        从环境 observation 的 npcs 字段中缓存 npc 位置。
        """
        npc_positions = []
        for npc in npcs:
            if not isinstance(npc, dict):
                continue

            pos = npc.get("pos") or {}
            x = int(pos.get("x", -1))
            z = int(pos.get("z", -1))
            if not (0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE):
                continue

            npc_positions.append((x, z))
        self.npc_positions = npc_positions

        # 计算距离最近的 npc 距离
        if npc_positions:
            npc_distance = float(np.min([
                np.sqrt((x - cur_pos[0])**2 + (z - cur_pos[1])**2) for x, z in npc_positions
            ]))
        else:
            npc_distance = 200.0
        # 只有当距离npc为2个格子的时候，才认为是距离npc过近了 给reward用
        self.close_npc_state = npc_distance < 5.0 and npc_distance < self.last_npc_distance
        self.last_npc_distance = npc_distance


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
                    # 0 = obstacle, 1 = passable, 3 = charger
                    # 0 = 障碍, 1 = 可通行, 3 = 充电桩
                    self.passable_map[gx, gz] = 1 if view[ri, ci] != 0 else 0

    def _get_local_view_feature(self):
        """Local view feature: crop center 15×15 from 21×21.

        局部视野特征：从 21×21 视野中心裁剪 15×15。
        """
        view = np.asarray(self._view_map, dtype=np.float32)
        if view.ndim != 2:
            view = np.zeros((2 * self.LOCAL_HALF + 1, 2 * self.LOCAL_HALF + 1), dtype=np.float32)

        target_size = 2 * self.LOCAL_HALF + 1
        h, w = view.shape
        center_x = h // 2
        center_z = w // 2

        start_x = max(center_x - self.LOCAL_HALF, 0)
        end_x = min(center_x + self.LOCAL_HALF + 1, h)
        start_z = max(center_z - self.LOCAL_HALF, 0)
        end_z = min(center_z + self.LOCAL_HALF + 1, w)

        crop = view[start_x:end_x, start_z:end_z]

        if crop.shape != (target_size, target_size):
            padded = np.zeros((target_size, target_size), dtype=np.float32)
            offset_x = (target_size - crop.shape[0]) // 2
            offset_z = (target_size - crop.shape[1]) // 2
            padded[offset_x : offset_x + crop.shape[0], offset_z : offset_z + crop.shape[1]] = crop
            crop = padded

        return (crop / 2.0).flatten()

    def _get_view_cell(self, x, z, hx, hz):
        """Get cell value from local view by global coordinate.

        根据全局坐标读取局部视野中的格子值。
        """
        if self._view_map is None:
            return 0

        view_h, view_w = self._view_map.shape[:2]
        center_x = view_h // 2
        center_z = view_w // 2

        view_x = x - hx + center_x
        view_z = z - hz + center_z
        if not (0 <= view_x < view_h and 0 <= view_z < view_w):
            return 0

        return int(self._view_map[view_x, view_z])

    def _calc_ray_target_features(self, target_value, hx, hz, max_ray=30):
        """Find normalized nearest target distance along 8 rays.

        计算八方向射线上最近目标距离，并归一化。
        """
        ray_target = []

        for dx, dz in self.RAY_DIRS_8:
            x, z = hx, hz
            found = max_ray
            for step in range(1, max_ray + 1):
                x += dx
                z += dz
                if not (0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE):
                    break
                if self._get_view_cell(x, z, hx, hz) == target_value:
                    found = step
                    break
            ray_target.append(_norm(found, max_ray))

        return ray_target

    def _calc_ray_dirt_features(self, hx, hz, max_ray=30):
        return self._calc_ray_target_features(2, hx, hz, max_ray=max_ray)

    def _calc_ray_charge_features(self, hx, hz, max_ray=60):
        ray_charge = []
        for dx, dz in self.RAY_DIRS_8:
            found = max_ray
            for step in range(1, max_ray + 1):
                x = hx + dx * step
                z = hz + dz * step
                if not (0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE):
                    break
                if self.passable_map[x, z] == 3:
                    found = step
                    break
            ray_charge.append(_norm(found, max_ray))
        return ray_charge

    def _update_nearest_dirt_metrics(self):
        """Update nearest dirt distance and approaching indicator.

        更新最近污渍距离及是否接近污渍标记。
        """
        self.last_nearest_dirt_dist = self.nearest_dirt_dist
        self.nearest_dirt_dist = self._calc_nearest_dirt_dist()
        nearest_dirt_norm = _norm(self.nearest_dirt_dist, self.TARGET_MAX_DIST)
        dirt_delta = 1.0 if self.nearest_dirt_dist < self.last_nearest_dirt_dist else 0.0
        return nearest_dirt_norm, dirt_delta


    def _update_nearest_charge_metrics(self):
        """Update nearest charge distance and approaching indicator.

        更新最近充电桩距离及是否接近充电桩标记。
        """
        self.last_charge_distance = self.charge_distance
        self.last_nearest_charge_dist = self.nearest_charge_dist
        self.charge_distance = self._calc_nearest_charge_dist()
        self.nearest_charge_dist = self.charge_distance
        self.charge_dis_delta = float(self.last_charge_distance - self.charge_distance)

    def _calc_target_direction_feature(self, target_positions):
        """Build soft 8-direction feature for nearest target.

        为最近目标构建 8 方向 soft one-hot 特征。
        """
        direction = np.zeros(8, dtype=np.float32)
        if not target_positions:
            return direction

        cur = np.array(self.cur_pos, dtype=np.float32)
        target = min(target_positions, key=lambda pos: (pos[0] - self.cur_pos[0]) ** 2 + (pos[1] - self.cur_pos[1]) ** 2)
        vec = np.array([target[0] - cur[0], target[1] - cur[1]], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm <= 1e-6:
            direction.fill(1.0 / 8.0)
            return direction

        unit = vec / norm
        basis = [
            np.array([0.0, -1.0], dtype=np.float32),
            np.array([1.0, -1.0], dtype=np.float32) / np.sqrt(2.0),
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([1.0, 1.0], dtype=np.float32) / np.sqrt(2.0),
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([-1.0, 1.0], dtype=np.float32) / np.sqrt(2.0),
            np.array([-1.0, 0.0], dtype=np.float32),
            np.array([-1.0, -1.0], dtype=np.float32) / np.sqrt(2.0),
        ]
        scores = np.maximum(np.array([float(np.dot(unit, b)) for b in basis], dtype=np.float32), 0.0)
        score_sum = float(np.sum(scores))
        if score_sum <= 1e-6:
            direction.fill(1.0 / 8.0)
            return direction
        return scores / score_sum

    def _calc_nearest_charge_dist(self):
        """Find nearest charger Euclidean distance from cached charger positions.

        从已缓存的充电桩坐标中找最近充电桩的欧氏距离。
        """
        if not self.charger_positions:
            return 200.0
        cur = np.array(self.cur_pos, dtype=np.float32)
        charger_coords = np.array(self.charger_positions, dtype=np.float32)
        dists = np.sqrt(np.sum((charger_coords - cur) ** 2, axis=1))
        return float(np.min(dists))
        

    def _get_global_state_feature(self):
        """Global state feature (36D).

        全局状态特征（36D）。

        Dimensions / 维度说明：
          [0]  step_norm         step progress / 步数归一化 [0,1]
          [1]  battery_ratio     battery level / 电量比 [0,1]
          [2]  cleaning_progress cleaned ratio / 已清扫比例 [0,1]
          [3]  remaining_dirt    remaining dirt ratio / 剩余污渍比例 [0,1]
          [4]  pos_x_norm        x position / x 坐标归一化 [0,1]
          [5]  pos_z_norm        z position / z 坐标归一化 [0,1]
          [6]  charging_flag     on charger / 是否正在充电桩上
          [7]  charge_count_norm charge count / 充电次数归一化
          [8:16]   ray_dirt_8      8方向污渍距离
          [16]     nearest_dirt    最近污渍距离归一化
          [17]     dirt_delta      是否接近污渍
          [18:26]  ray_charge_8    8方向充电桩距离
          [26:34]  charge_dir_8    最近充电桩相对方向
          [34]     nearest_charge  最近充电桩距离归一化
          [35]     charge_delta    是否接近充电桩
        """
        step_norm = _norm(self.step_no, 2000)
        battery_ratio = _norm(self.battery, self.battery_max)
        cleaning_progress = _norm(self.dirt_cleaned, self.total_dirt)
        remaining_dirt = 1.0 - cleaning_progress

        hx, hz = self.cur_pos
        pos_x_norm = _norm(hx, self.GRID_SIZE)
        pos_z_norm = _norm(hz, self.GRID_SIZE)

        charging_flag = 1.0 if self.charging else 0.0
        charge_count_norm = _norm(self.charge_count, max(self.total_charger, 1) * 4)

        # 8-directional rays for dirt / 污渍八方向射线距离
        ray_dirt = self._calc_ray_dirt_features(hx, hz)

        # Nearest dirt Euclidean distance (estimated from cropped local view)
        # 最近污渍欧氏距离（基于裁剪后的局部视野粗估）
        nearest_dirt_norm, dirt_delta = self._update_nearest_dirt_metrics()


        # Nearest charge Euclidean distance (estimated from global map)
        # 最近充电桩欧氏距离（基于全局地图粗估）
        self._update_nearest_charge_metrics()
        nearest_charge_norm = _norm(self.charge_distance, self.TARGET_MAX_DIST)
        charge_delta = 1.0 if self.charge_dis_delta > 0 else 0.0
        ray_charge = self._calc_ray_charge_features(hx, hz)
        self.charge_dir = self._calc_target_direction_feature(self.charger_positions)


        ## 计算充电指标
        if self.charging and not self.last_charging:
            self.charge_count += 1
            self.first_charge_reward = True
        
        return np.array(
            [
                step_norm,
                battery_ratio,
                cleaning_progress,
                remaining_dirt,
                pos_x_norm,
                pos_z_norm,
                charging_flag,
                charge_count_norm,
                *ray_dirt,
                nearest_dirt_norm,
                dirt_delta,
                *ray_charge,
                *self.charge_dir.tolist(),
                nearest_charge_norm,
                charge_delta,
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
        center_x = view.shape[0] // 2
        center_z = view.shape[1] // 2
        dists = np.sqrt((dirt_coords[:, 0] - center_x) ** 2 + (dirt_coords[:, 1] - center_z) ** 2)
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
        context.dirt_delta = float(global_state[17])

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
        context.charging = self.charging
        context.first_charge_reward = self.first_charge_reward
        context.charge_count = self.charge_count
        context.loop_pos = self.loop_pos
        context.close_npc_state = self.close_npc_state

        return context

    def feature_process(self, env_obs, last_action):
        """Generate feature vector, legal action mask, and scalar reward.

        生成特征向量、合法动作掩码和标量奖励。
        """
        self.pb2struct(env_obs, last_action)

        local_view = self._get_local_view_feature()  # 225D
        global_state = self._get_global_state_feature()  # 36D
        legal_action = self.get_legal_action()  # 8D
        legal_arr = np.array(legal_action, dtype=np.float32)

        feature = np.concatenate([local_view, global_state, legal_arr])
        expected_dim = local_view.shape[0] + global_state.shape[0] + legal_arr.shape[0]
        if feature.shape[0] != expected_dim:
            raise ValueError(
                f"feature_process produced invalid dim={feature.shape[0]}, "
                f"local_view_shape={self._view_map.shape if self._view_map is not None else None}, "
                f"local_feature_dim={local_view.shape[0]}, global_dim={global_state.shape[0]}, legal_dim={legal_arr.shape[0]}"
            )

        reward_context = self._build_reward_context(local_view, global_state, legal_action, legal_arr, feature)
        reward = self.reward_process(reward_context)

        return feature, legal_action, reward

    def reward_process(self, context):
        """Compute reward with prepared reward context.

        基于构建好的 reward context 计算奖励。
        """
        charge_reward = self.reward_charge.get_reward(context)
        clean_reward = self.reward_clean.get_reward(context)
        explore_reward = self.reward_explore.get_reward(context)
        npc_reward = self.reward_npc.get_reward(context)
        
        total_reward = charge_reward + clean_reward + explore_reward + npc_reward

        self.last_charge_reward = float(charge_reward)
        self.last_clean_reward = float(clean_reward)
        self.last_explore_reward = float(explore_reward)
        self.last_npc_reward = float(npc_reward)
        self.last_total_reward = float(total_reward)

        self.episode_charge_reward += self.last_charge_reward
        self.episode_clean_reward += self.last_clean_reward
        self.episode_explore_reward += self.last_explore_reward
        self.episode_npc_reward += self.last_npc_reward
        self.episode_total_reward += self.last_total_reward

        return total_reward
