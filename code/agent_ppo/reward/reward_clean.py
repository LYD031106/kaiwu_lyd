from agent_ppo.reward.reward_context import RewardContext


class RewardClean:
    def __init__(self):
        self.reward_context = RewardContext()

    def get_reward(self, context: RewardContext):
        """Get cleaning reward from observation context.

        仅计算清扫增量奖励。
        """
        cleaned_delta = max(int(context.dirt_cleaned) - int(context.last_dirt_cleaned), 0)
        return cleaned_delta * 0.1