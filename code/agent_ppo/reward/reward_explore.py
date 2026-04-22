from agent_ppo.reward.reward_context import RewardContext


class RewardCharge:
    def __init__(self):
        self.reward_context = RewardContext()
    
    def get_reward(self, context: RewardContext):
        """Get reward from observation dict.
        
        计算探索 shaping 奖励。
        """
        reward = 0.0
        
        # 首先计算打扫奖励
        if context.dirt_cleaned - context.last_dirt_cleaned > 0:
            reward += (context.dirt_cleaned - context.last_dirt_cleaned) * 0.1
        
        return reward
