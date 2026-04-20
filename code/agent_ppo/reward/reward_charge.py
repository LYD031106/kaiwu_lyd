from agent_ppo.reward.reward_context import RewardContext


class RewardCharge:
    def __init__(self):
        self.reward_context = RewardContext()

    def get_reward(self, context: RewardContext):
        """Get reward from observation dict.

        计算 充电奖励
        """
        reward = 0.0
        # 分阶段当电量在 50 以上 不会施加任何奖励
        if context.battery >= 30:
            pass
        elif context.battery < 30 and context.battery > 15:
            if context.charge_dis_delta == 0:
                reward += 0.5
            else:
                reward -= 0.2
        elif context.battery < 15:
            if context.charge_dis_delta == 0:
                reward += 0.7
            else:
                reward -= 0.5

        return reward
        return reward
