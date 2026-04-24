from agent_ppo.reward.reward_context import RewardContext


class RewardCharge:
    def __init__(self):
        self.reward_context = RewardContext()

    def get_reward(self, context: RewardContext):
        """Get reward from observation dict.

        计算 充电奖励 总量级为[-6, 6]
        """
        reward = 0.0
        # 分阶段当电量在 30 以上 不会施加任何奖励 适当进行惩罚
        charge_ratio = context.battery / context.battery_max
        moving_towards_charge = context.charge_dis_delta > 0
        if context.battery >= 30:
            # 适当进行惩罚
            if context.charging:
                reward -= 0.2
        elif context.battery < 30 and context.battery > 15:
            if moving_towards_charge:
                reward += 0.2
            else :
                reward -= 0.2
        elif context.battery < 15:
            if moving_towards_charge:
                reward += 0.4
            else :
                reward -= 0.4


        # 里程碑奖励
        if context.charge_count == 1 and context.first_charge_reward:
            reward += 3

        return reward
