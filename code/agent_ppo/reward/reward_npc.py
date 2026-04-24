from agent_ppo.reward.reward_context import RewardContext


class RewardNpc:
    def __init__(self):
        self.reward_context = RewardContext()
    
    def get_reward(self, context: RewardContext):
        """Get reward from observation dict.

        计算 npc 近近奖励。
        """
        reward = 0.0
        if context.close_npc_state:
            reward -= 0.5
        return reward
