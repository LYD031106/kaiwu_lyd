from agent_ppo.reward.reward_context import RewardContext


class RewardExplore:
    def __init__(self):
        self.reward_context = RewardContext()
    
    def get_reward(self, context: RewardContext):
        """Get reward from observation dict.
        
        计算探索 shaping 奖励。
        """
        reward = 0.0
        
        # 首先计算探索奖励
        # 直接污渍来，因为污渍肯定就是没去过的地方
        if context.dirt_cleaned - context.last_dirt_cleaned > 0:
            reward += (context.dirt_cleaned - context.last_dirt_cleaned) * 0.01
        
        if context.loop_pos > 0:
            reward -= context.loop_pos * 0.1
        return reward
