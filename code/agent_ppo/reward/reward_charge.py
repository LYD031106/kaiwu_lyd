from agent_ppo.reward.reward_context import RewardContext


class RewardCharge:
    def __init__(self):
        self.reward_context = RewardContext()

    @staticmethod
    def _safe_margin(distance: float):
        """Estimate battery safety margin for returning to charger.

        基于最短路距离估计安全回桩所需的最低电量。
        """
        distance = max(float(distance), 0.0)
        return distance * 1.5 + 8.0

    def get_reward(self, context: RewardContext):
        """Get reward from observation dict.

        基于安全余量的充电奖励。
        """
        reward = 0.0

        battery = float(context.battery)
        distance = float(context.charge_distance)
        delta = float(context.charge_dis_delta)
        moving_towards_charge = delta > 0.0
        moving_away_from_charge = delta < 0.0
        safe_margin = self._safe_margin(distance)
        margin = battery - safe_margin

        # 安全余量很大：基本不打扰清扫，只轻微抑制高电量站桩
        if margin >= 30.0:
            if context.charging:
                reward -= 0.02
            return reward

        # 余量尚可：轻微提醒，可以开始往充电桩方向收敛
        if margin >= 20.0:
            if moving_towards_charge:
                reward += min(delta, 2.0) * 0.04
            elif moving_away_from_charge and distance < 40.0:
                reward += max(delta, -2.0) * 0.04

        # 已接近安全边界：明显鼓励回桩
        elif margin >= 10.0:
            if moving_towards_charge:
                reward += 0.04 + min(delta, 3.0) * 0.05
            elif moving_away_from_charge:
                reward -= 0.04 + min(abs(delta), 3.0) * 0.05
            else:
                reward -= 0.03

        # 最后纠偏区：已经非常接近安全边界，但仍有机会抢救回来
        elif margin >= 0.0:
            if moving_towards_charge:
                reward += 0.10 + min(delta, 4.0) * 0.06
            elif moving_away_from_charge:
                reward -= 0.10 + min(abs(delta), 4.0) * 0.06
            else:
                reward -= 0.05

        # 已跌破安全边界：进入补救区，只奖励正确补救动作，不再重罚
        else:
            if moving_towards_charge:
                reward += 0.04 + min(delta, 3.0) * 0.04
            elif moving_away_from_charge:
                reward -= 0.01

        # 成功上桩时给明确里程碑奖励
        if context.first_charge_reward:
            reward += 2.0 if margin < 20.0 else 1.5

        return reward
