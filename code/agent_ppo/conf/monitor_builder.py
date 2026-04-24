#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Monitor panel configuration builder for Robot Vacuum.
清扫大作战监控面板配置构建器。
"""


from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("清扫大作战")
        .add_group(
            group_name="算法指标",
            group_name_en="algorithm",
        )
        .add_panel(
            name="训练回报",
            name_en="training_reward",
            type="line",
        )
        .add_metric(
            metrics_name="reward",
            expr="avg(reward{})",
        )
        .end_panel()
        .add_panel(
            name="单局累计奖励",
            name_en="episode_reward_metrics",
            type="line",
        )
        .add_metric(
            metrics_name="单局总奖励",
            expr="avg(reward_total{})",
        )
        .add_metric(
            metrics_name="清扫奖励",
            expr="avg(reward_clean{})",
        )
        .add_metric(
            metrics_name="探索奖励",
            expr="avg(reward_explore{})",
        )
        .add_metric(
            metrics_name="NPC奖励",
            expr="avg(reward_npc{})",
        )
        .add_metric(
            metrics_name="充电奖励",
            expr="avg(reward_charge{})",
        )
        .add_metric(
            metrics_name="终局奖励",
            expr="avg(reward_final{})",
        )
        .end_panel()
        .add_panel(
            name="单步平均奖励",
            name_en="step_avg_reward_metrics",
            type="line",
        )
        .add_metric(
            metrics_name="步均清扫奖励",
            expr="avg(step_avg_reward_clean{})",
        )
        .add_metric(
            metrics_name="步均探索奖励",
            expr="avg(step_avg_reward_explore{})",
        )
        .add_metric(
            metrics_name="步均NPC奖励",
            expr="avg(step_avg_reward_npc{})",
        )
        .add_metric(
            metrics_name="步均充电奖励",
            expr="avg(step_avg_reward_charge{})",
        )
        .end_panel()
        .add_panel(
            name="对局指标",
            name_en="episode_metrics",
            type="line",
        )
        .add_metric(
            metrics_name="每局充电数",
            expr="avg(charge_count{})",
        )
        .add_metric(
            metrics_name="对局数",
            expr="avg(episode_cnt{})",
        )
        .end_panel()
        .add_panel(
            name="总损失",
            name_en="total_loss",
            type="line",
        )
        .add_metric(
            metrics_name="total_loss",
            expr="avg(total_loss{})",
        )
        .end_panel()
        .add_panel(
            name="价值损失",
            name_en="value_loss",
            type="line",
        )
        .add_metric(
            metrics_name="value_loss",
            expr="avg(value_loss{})",
        )
        .end_panel()
        .add_panel(
            name="策略损失",
            name_en="policy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="policy_loss",
            expr="avg(policy_loss{})",
        )
        .end_panel()
        .add_panel(
            name="熵损失",
            name_en="entropy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="entropy_loss",
            expr="avg(entropy_loss{})",
        )
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
