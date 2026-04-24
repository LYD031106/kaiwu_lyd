import numpy as np


class RewardContext:
    GRID_SIZE = 128
    VIEW_HALF = 10
    LOCAL_HALF = 5

    def __init__(self):
        self.reset()

    def reset(self):
        # 全局状态
        self.step_no = 0
        self.battery = 600
        self.battery_max = 600

        self.cur_pos = (0, 0)

        self.dirt_cleaned = 0
        self.last_dirt_cleaned = 0
        self.total_dirt = 1

        # 全局通行地图（0=障碍, 1=可通行）
        self.passable_map = np.ones((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)

        # 最近污渍距离
        self.nearest_dirt_dist = 200.0
        self.last_nearest_dirt_dist = 200.0

        # 局部视野与合法动作
        self._view_map = np.zeros((21, 21), dtype=np.float32)
        self._legal_act = [1] * 8

        # 特征处理结果缓存
        self.local_view = np.zeros(121, dtype=np.float32)
        self.global_state = np.zeros(12, dtype=np.float32)
        self.legal_action = [1] * 8
        self.legal_arr = np.ones(8, dtype=np.float32)
        self.feature = np.zeros(141, dtype=np.float32)
        self.reward = 0.0

        # 探索字段
        self.visit = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        self.dur_visit = np.zeros((3, 3), dtype=np.int8)
        self.dirt_delta = 0.0

        # 充电字段
        self.charge_distance = 200.0
        self.charge_dis_delta = 0
        self.charge_dir = np.zeros(8, dtype=np.float32)
        self.charging = False  # 是否正在充电中

        # 里程碑奖励
        self.first_charge_reward = False
        self.charge_count = 0

        # 循环次数
        self.loop_pos = 0.0

        # npc 靠近状态
        self.close_npc_state = False
