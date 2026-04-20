class RewardCharge:
    def __init__(self):
        self.reward_context = RewardContext()
        
    def get_reward(self, context: RewardContext):
        """Get reward from observation dict.
        
        计算 充电奖励
        """
        reward = 0.0
        # 分阶段当电量在 50 以上 不会施加任何奖励
        if context.battery >= 50:
            pass
        elif context.battery < 50 and context.battery > 30:
            if context.charge_dis_delta == 0:
                reward += 0.3
            else:
                reward -= 0.1
        elif context.battery < 30:
            if context.charge_dis_delta == 0:
                reward += 0.5
            else:
                reward -= 0.3
            