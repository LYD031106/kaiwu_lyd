#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Robot Vacuum.
"""

import os
import time
from collections import deque

import numpy as np

from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import SampleData, sample_process
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf


EPISODE_METRICS_WINDOW_SIZE = 100
MIN_STAGE_WINDOW_EPISODES = 20

TRAINING_STAGE_EARLY = "early"
TRAINING_STAGE_MIDDLE = "middle"
TRAINING_STAGE_LATE = "late"

TRAINING_STAGE_LABELS = {
    TRAINING_STAGE_EARLY: "早期",
    TRAINING_STAGE_MIDDLE: "中期",
    TRAINING_STAGE_LATE: "后期",
}

TRAINING_STAGE_IDS = {
    TRAINING_STAGE_EARLY: 0,
    TRAINING_STAGE_MIDDLE: 1,
    TRAINING_STAGE_LATE: 2,
}

TRAINING_STAGE_THRESHOLDS = {
    "early": {
        "first_charge_success_rate_max": 0.55,
        "battery_depleted_rate_min": 0.45,
        "charge_route_found_rate_weighted_max": 0.55,
        "dock_success_given_contact_rate_max": 0.60,
        "dock_contact_without_charge_mean_min": 2.5,
        "dock_stall_steps_mean_min": 3.0,
    },
    "late": {
        "first_charge_success_rate_min": 0.75,
        "battery_depleted_rate_max": 0.08,
        "charge_route_found_rate_weighted_min": 0.70,
        "dock_contact_without_charge_mean_max": 2.0,
        "route_stall_steps_total_mean_max": 8.0,
        "dock_success_given_contact_rate_min": 0.82,
        "dock_stall_steps_mean_max": 1.5,
        "dock_regress_count_mean_max": 1.0,
    },
}

FINAL_REWARD_STAGE_PROFILES = {
    TRAINING_STAGE_EARLY: {
        "completed_base": 3.2,
        "completed_clean_scale": 4.8,
        "completed_charge_scale": 0.8,
        "completed_first_charge_bonus": 1.0,
        "battery_depleted_base": -5.0,
        "battery_depleted_no_charge_extra": -1.2,
        "near_charger_extra": -0.5,
        "npc_collision_penalty": -2.7,
        "other_failure_penalty": -2.4,
    },
    TRAINING_STAGE_MIDDLE: {
        "completed_base": 4.0,
        "completed_clean_scale": 6.0,
        "completed_charge_scale": 0.4,
        "completed_first_charge_bonus": 0.4,
        "battery_depleted_base": -4.0,
        "battery_depleted_no_charge_extra": -1.0,
        "near_charger_extra": -0.4,
        "npc_collision_penalty": -2.5,
        "other_failure_penalty": -2.0,
    },
    TRAINING_STAGE_LATE: {
        "completed_base": 4.8,
        "completed_clean_scale": 7.2,
        "completed_charge_scale": 0.2,
        "completed_first_charge_bonus": 0.1,
        "battery_depleted_base": -4.2,
        "battery_depleted_no_charge_extra": -0.8,
        "near_charger_extra": -0.3,
        "npc_collision_penalty": -2.2,
        "other_failure_penalty": -1.8,
    },
}


def _normalize_training_stage_name(stage_name):
    normalized_name = str(stage_name or TRAINING_STAGE_EARLY).strip().lower()
    if normalized_name not in TRAINING_STAGE_IDS:
        return TRAINING_STAGE_EARLY
    return normalized_name


def _get_final_reward_profile(stage_name):
    return FINAL_REWARD_STAGE_PROFILES[_normalize_training_stage_name(stage_name)]


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    """
    训练流程入口。

    这里负责做训练侧的最外层编排：
    1. 读取环境配置。
    2. 构造 `EpisodeRunner`。
    3. 持续消费 runner 产出的样本并发送给 learner。
    4. 按固定时间间隔自动保存模型。
    """
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


def _extract_extra_info(env_obs):
    """
    兼容不同观测结构，提取环境返回的 `extra_info`。

    有些环境字段直接挂在最外层，有些包在 `observation` 下面，这里统一做一次兜底。
    """
    observation = env_obs.get("observation", {})
    return env_obs.get("extra_info") or observation.get("extra_info") or {}


def _extract_result_details(env_obs, fm, truncated, step):
    """
    从终局观测中提取结果标签和核心统计。

    这里不再强调规则排障细节，而是更关注当前 baseline 是否学会：
    1. 存活到终局。
    2. 成功完成至少一次充电。
    3. 在保证存活的前提下尽可能提高清扫比例。

    返回结果会同时保留：
    1. 环境直接给出的结果字段，如 `result_code`、`result_message`。
    2. 训练关心的行为统计，如 `charge_count`、`clean_ratio`。
    3. `Preprocessor` 导出的调试快照，便于 GAMEOVER 时定位问题到底出在回桩、贴桩还是撞 NPC。
    """
    env_info = env_obs.get("observation", {}).get("env_info", {})
    extra_info = _extract_extra_info(env_obs)
    snapshot = fm.get_debug_snapshot()

    # 先统一收集环境原始统计，避免后面的 fail_reason 判断混进太多取字段逻辑。
    result_code = extra_info.get("result_code")
    result_message = extra_info.get("result_message")
    total_score = int(env_info.get("total_score", 0))
    clean_score = int(env_info.get("clean_score", total_score))
    remaining_charge = int(env_info.get("remaining_charge", snapshot["remaining_charge"]))
    charge_count = int(env_info.get("charge_count", snapshot["charge_count"]))
    max_step = max(int(env_info.get("max_step", fm.max_step)), 1)
    frame_no = int(env_obs.get("frame_no", step))
    clean_ratio = float(fm.dirt_cleaned) / float(max(fm.total_dirt, 1))

    completed_by_max_step = bool(
        truncated
        and result_code in (None, 0)
        and not result_message
        and max(step, frame_no) >= max_step
    )

    # fail_reason 是训练侧自己的统一终局标签。
    # 它的目的不是完全复刻环境结果码，而是把奖励、监控和排查真正关心的失败类型压成少数几类。
    if completed_by_max_step:
        fail_reason = "completed_max_step"
    elif truncated:
        fail_reason = "abnormal_truncated"
    elif result_message:
        fail_reason = str(result_message)
    elif result_code not in (None, 0):
        fail_reason = f"result_code_{result_code}"
    elif remaining_charge <= 0:
        fail_reason = "battery_depleted"
    elif snapshot["nearest_npc_dist"] is not None and snapshot["nearest_npc_dist"] <= 1:
        fail_reason = "npc_collision"
    else:
        fail_reason = "unknown_failure"

    return {
        "result_code": result_code,
        "result_message": result_message,
        "fail_reason": fail_reason,
        "total_score": total_score,
        "clean_score": clean_score,
        "remaining_charge": remaining_charge,
        "charge_count": charge_count,
        "frame_no": frame_no,
        "max_step": max_step,
        "is_completed": completed_by_max_step,
        "clean_ratio": clean_ratio,
        "first_charge_success": float(charge_count > 0),
        "battery_depleted": float(fail_reason == "battery_depleted"),
        "training_stage_name": _normalize_training_stage_name(snapshot.get("training_stage_name")),
        "training_stage_id": int(snapshot.get("training_stage_id", 0)),
        "snapshot": snapshot,
    }


def _compute_final_reward(result_details):
    """
    计算 episode 终局奖励。

    设计原则：
    1. 存活并完成清扫是主目标。
    2. 电量耗尽要有显著惩罚，避免策略把回充当成可有可无。
    3. 终局奖励尽量简单，主要学习压力交给 step-level reward shaping。

    也就是说，这里只负责在 episode 结束时做一次“方向性校正”：
    完成任务给正奖励，明显失败给负奖励；真正细粒度的回充学习信号仍来自 step reward。
    """
    reward_profile = _get_final_reward_profile(result_details.get("training_stage_name"))

    if result_details["is_completed"]:
        # 正常跑满并完成清扫时，清扫比例是主项，充电次数只给很轻的附加项，
        # 避免策略为了刷充电次数而牺牲主任务。
        final_reward = (
            reward_profile["completed_base"]
            + reward_profile["completed_clean_scale"] * result_details["clean_ratio"]
            + reward_profile["completed_charge_scale"] * min(result_details["charge_count"], 3)
        )
        if result_details["first_charge_success"] > 0:
            final_reward += reward_profile["completed_first_charge_bonus"]
        return final_reward, "WIN"

    if result_details["fail_reason"] == "battery_depleted":
        # 电量耗尽是当前阶段最希望优先压下去的失败类型，所以这里惩罚最重。
        final_reward = reward_profile["battery_depleted_base"]
        if result_details["charge_count"] <= 0:
            final_reward += reward_profile["battery_depleted_no_charge_extra"]
        nearest_charger = result_details["snapshot"]["nearest_charger_dist"]
        if nearest_charger is not None and nearest_charger <= 3:
            final_reward += reward_profile["near_charger_extra"]
        return final_reward, "FAIL"

    if result_details["fail_reason"] == "npc_collision":
        return reward_profile["npc_collision_penalty"], "FAIL"

    return reward_profile["other_failure_penalty"], "FAIL"


def _build_episode_metrics(result_details):
    """
    整理一组更适合观察回充学习效果的核心指标。

    这些指标主要用于监控和日志，不直接参与训练计算。重点是把“是否学会首充、
    是否还会电量耗尽、最终清扫得怎么样”这几个维度稳定暴露出来。
    """
    snapshot = result_details["snapshot"]
    low_battery_route_progress_mean = snapshot.get("low_battery_route_progress_mean")
    return {
        "clean_ratio": float(result_details["clean_ratio"]),
        "charge_count": int(result_details["charge_count"]),
        "first_charge_success": float(result_details["first_charge_success"]),
        "battery_depleted": float(result_details["battery_depleted"]),
        "remaining_charge": int(result_details["remaining_charge"]),
        "completed": float(result_details["is_completed"]),
        "explored_ratio": float(snapshot["explored_ratio"]),
        "nearest_charger_dist": (
            -1 if snapshot["nearest_charger_dist"] is None else int(snapshot["nearest_charger_dist"])
        ),
        "first_charge_step": -1 if snapshot["first_charge_step"] is None else int(snapshot["first_charge_step"]),
        "return_trigger_step": -1 if snapshot["return_trigger_step"] is None else int(snapshot["return_trigger_step"]),
        "return_trigger_margin": (
            -999 if snapshot["return_trigger_margin"] is None else int(snapshot["return_trigger_margin"])
        ),
        "first_return_trigger_step": (
            -1 if snapshot["first_return_trigger_step"] is None else int(snapshot["first_return_trigger_step"])
        ),
        "min_battery_margin": (
            -999 if snapshot["min_battery_margin"] is None else int(snapshot["min_battery_margin"])
        ),
        "battery_margin_at_first_charge": (
            -999
            if snapshot["battery_margin_at_first_charge"] is None
            else int(snapshot["battery_margin_at_first_charge"])
        ),
        "low_battery_steps": int(snapshot["low_battery_steps"]),
        "low_battery_route_progress_mean": (
            0.0 if low_battery_route_progress_mean is None else float(low_battery_route_progress_mean)
        ),
        "charge_route_found_rate": float(snapshot["charge_route_found_rate"]),
        "route_stall_steps_total": int(snapshot["route_stall_steps_total"]),
        "max_charge_stall_steps": int(snapshot["max_charge_stall_steps"]),
        "dock_contact_without_charge": int(snapshot["dock_contact_without_charge"]),
        "near_dock_entries": int(snapshot.get("near_dock_entries", 0)),
        "dock_contact_entries": int(snapshot.get("dock_contact_entries", 0)),
        "dock_success_after_contact_count": int(snapshot.get("dock_success_after_contact_count", 0)),
        "charge_success_after_dock_step_sum": int(snapshot.get("charge_success_after_dock_step_sum", 0)),
        "dock_stall_steps_total": int(snapshot.get("dock_stall_steps_total", 0)),
        "dock_regress_count": int(snapshot.get("dock_regress_count", 0)),
        "charge_guidance_steps": int(snapshot["charge_guidance_steps"]),
        "charge_route_found_steps": int(snapshot["charge_route_found_steps"]),
    }


def _classify_training_stage(window_metrics):
    """
    根据最近窗口指标，将训练状态分为早期 / 中期 / 后期。

    判定原则遵循当前项目的训练目标优先级：
    1. 先建立“活下来 + 回得去 + 充得上”的生存链路。
    2. 再追求稳定性。
    3. 最后优化清扫效率与完成质量。
    """
    episode_count = int(window_metrics.get("window_episode_count", 0))
    first_charge_success_rate = float(window_metrics.get("first_charge_success_rate", 0.0))
    battery_depleted_rate = float(window_metrics.get("battery_depleted_rate", 0.0))
    charge_route_found_rate_weighted = float(window_metrics.get("charge_route_found_rate_weighted", 0.0))
    dock_contact_without_charge_mean = float(window_metrics.get("dock_contact_without_charge_mean", 0.0))
    route_stall_steps_total_mean = float(window_metrics.get("route_stall_steps_total_mean", 0.0))
    near_dock_entry_rate = float(window_metrics.get("near_dock_entry_rate", 0.0))
    dock_success_given_contact_rate = float(window_metrics.get("dock_success_given_contact_rate", 0.0))
    dock_stall_steps_mean = float(window_metrics.get("dock_stall_steps_mean", 0.0))
    dock_regress_count_mean = float(window_metrics.get("dock_regress_count_mean", 0.0))
    completed_rate = float(window_metrics.get("completed_rate", 0.0))
    clean_ratio_mean = float(window_metrics.get("clean_ratio_mean", 0.0))

    early_thresholds = TRAINING_STAGE_THRESHOLDS["early"]
    late_thresholds = TRAINING_STAGE_THRESHOLDS["late"]
    reasons = []

    if episode_count < MIN_STAGE_WINDOW_EPISODES:
        stage_name = TRAINING_STAGE_EARLY
        reasons.append(
            f"窗口局数不足({episode_count}<{MIN_STAGE_WINDOW_EPISODES})，默认按早期观察"
        )
    else:
        early_hits = []
        if first_charge_success_rate < early_thresholds["first_charge_success_rate_max"]:
            early_hits.append(
                "首充成功率偏低"
                f"({first_charge_success_rate:.4f}<{early_thresholds['first_charge_success_rate_max']:.2f})"
            )
        if battery_depleted_rate > early_thresholds["battery_depleted_rate_min"]:
            early_hits.append(
                "没电率偏高"
                f"({battery_depleted_rate:.4f}>{early_thresholds['battery_depleted_rate_min']:.2f})"
            )
        if charge_route_found_rate_weighted < early_thresholds["charge_route_found_rate_weighted_max"]:
            early_hits.append(
                "回桩路径命中率偏低"
                f"({charge_route_found_rate_weighted:.4f}<{early_thresholds['charge_route_found_rate_weighted_max']:.2f})"
            )
        if dock_success_given_contact_rate < early_thresholds["dock_success_given_contact_rate_max"]:
            early_hits.append(
                "贴桩后充上成功率偏低"
                f"({dock_success_given_contact_rate:.4f}<{early_thresholds['dock_success_given_contact_rate_max']:.2f})"
            )
        if dock_contact_without_charge_mean > early_thresholds["dock_contact_without_charge_mean_min"]:
            early_hits.append(
                "贴桩未充仍明显偏高"
                f"({dock_contact_without_charge_mean:.4f}>{early_thresholds['dock_contact_without_charge_mean_min']:.2f})"
            )
        if dock_stall_steps_mean > early_thresholds["dock_stall_steps_mean_min"]:
            early_hits.append(
                "近桩停滞步数偏高"
                f"({dock_stall_steps_mean:.4f}>{early_thresholds['dock_stall_steps_mean_min']:.2f})"
            )

        if early_hits:
            stage_name = TRAINING_STAGE_EARLY
            reasons.extend(early_hits)
        else:
            late_checks = [
                (
                    first_charge_success_rate >= late_thresholds["first_charge_success_rate_min"],
                    "首充成功率达标",
                    f"{first_charge_success_rate:.4f}>={late_thresholds['first_charge_success_rate_min']:.2f}",
                ),
                (
                    battery_depleted_rate <= late_thresholds["battery_depleted_rate_max"],
                    "没电率已压低",
                    f"{battery_depleted_rate:.4f}<={late_thresholds['battery_depleted_rate_max']:.2f}",
                ),
                (
                    charge_route_found_rate_weighted >= late_thresholds["charge_route_found_rate_weighted_min"],
                    "回桩路径命中率达标",
                    f"{charge_route_found_rate_weighted:.4f}>={late_thresholds['charge_route_found_rate_weighted_min']:.2f}",
                ),
                (
                    dock_contact_without_charge_mean <= late_thresholds["dock_contact_without_charge_mean_max"],
                    "贴桩未充次数较低",
                    f"{dock_contact_without_charge_mean:.4f}<={late_thresholds['dock_contact_without_charge_mean_max']:.2f}",
                ),
                (
                    route_stall_steps_total_mean <= late_thresholds["route_stall_steps_total_mean_max"],
                    "回桩卡顿步数较低",
                    f"{route_stall_steps_total_mean:.4f}<={late_thresholds['route_stall_steps_total_mean_max']:.2f}",
                ),
                (
                    dock_success_given_contact_rate >= late_thresholds["dock_success_given_contact_rate_min"],
                    "贴桩后对接成功率达标",
                    f"{dock_success_given_contact_rate:.4f}>={late_thresholds['dock_success_given_contact_rate_min']:.2f}",
                ),
                (
                    dock_stall_steps_mean <= late_thresholds["dock_stall_steps_mean_max"],
                    "近桩停滞步数较低",
                    f"{dock_stall_steps_mean:.4f}<={late_thresholds['dock_stall_steps_mean_max']:.2f}",
                ),
                (
                    dock_regress_count_mean <= late_thresholds["dock_regress_count_mean_max"],
                    "近桩回退次数较低",
                    f"{dock_regress_count_mean:.4f}<={late_thresholds['dock_regress_count_mean_max']:.2f}",
                ),
            ]
            late_hits = [f"{desc}({detail})" for ok, desc, detail in late_checks if ok]
            late_ready = len(late_hits) == len(late_checks)

            if late_ready:
                stage_name = TRAINING_STAGE_LATE
                reasons.extend(late_hits)
                reasons.append(
                    f"当前重点转向效率优化(completed_rate={completed_rate:.4f}, clean_ratio_mean={clean_ratio_mean:.4f})"
                )
            else:
                stage_name = TRAINING_STAGE_MIDDLE
                reasons.append(
                    "基础生存链路已脱离早期，但稳定性/效率指标尚未全部达到后期阈值"
                )
                reasons.append(
                    f"首充成功率={first_charge_success_rate:.4f}, 没电率={battery_depleted_rate:.4f}, "
                    f"近桩进入率={near_dock_entry_rate:.4f}, 贴桩成功率={dock_success_given_contact_rate:.4f}, "
                    f"完成率={completed_rate:.4f}, 清扫比例={clean_ratio_mean:.4f}"
                )

    stage_id = TRAINING_STAGE_IDS[stage_name]
    monitor_metrics = {
        "training_stage_id": float(stage_id),
        "training_stage_early": 1.0 if stage_name == TRAINING_STAGE_EARLY else 0.0,
        "training_stage_middle": 1.0 if stage_name == TRAINING_STAGE_MIDDLE else 0.0,
        "training_stage_late": 1.0 if stage_name == TRAINING_STAGE_LATE else 0.0,
    }
    return {
        "stage_name": stage_name,
        "stage_label": TRAINING_STAGE_LABELS[stage_name],
        "stage_id": stage_id,
        "monitor_metrics": monitor_metrics,
        "reasons": reasons,
    }


class EpisodeRunner:
    """按 episode 驱动环境交互、样本收集、终局奖励和监控上报。"""

    def __init__(self, env, agent, usr_conf, logger, monitor):
        """
        保存训练循环需要的环境、agent、配置和监控对象。

        `EpisodeRunner` 本身不做策略学习，它只负责跑环境、收样本、补终局奖励
        并把关键指标上报给日志和监控系统。
        """
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0
        self.recent_episode_records = deque(maxlen=EPISODE_METRICS_WINDOW_SIZE)

    @staticmethod
    def _mean(values):
        return float(sum(values) / len(values)) if values else 0.0

    def _mean_valid(self, key, invalid_values):
        values = [float(record[key]) for record in self.recent_episode_records if record[key] not in invalid_values]
        return self._mean(values)

    def _build_episode_record(self, episode_return, result_details, metrics):
        return {
            "episode_return": float(episode_return),
            "completed": float(metrics["completed"]),
            "battery_depleted": float(metrics["battery_depleted"]),
            "first_charge_success": float(metrics["first_charge_success"]),
            "clean_ratio": float(metrics["clean_ratio"]),
            "first_charge_step": int(metrics["first_charge_step"]),
            "return_trigger_margin": int(metrics["return_trigger_margin"]),
            "dock_contact_without_charge": int(metrics["dock_contact_without_charge"]),
            "near_dock_entries": int(metrics["near_dock_entries"]),
            "dock_contact_entries": int(metrics["dock_contact_entries"]),
            "dock_success_after_contact_count": int(metrics["dock_success_after_contact_count"]),
            "charge_success_after_dock_step_sum": int(metrics["charge_success_after_dock_step_sum"]),
            "dock_stall_steps_total": int(metrics["dock_stall_steps_total"]),
            "dock_regress_count": int(metrics["dock_regress_count"]),
            "route_stall_steps_total": int(metrics["route_stall_steps_total"]),
            "charge_guidance_steps": int(metrics["charge_guidance_steps"]),
            "charge_route_found_steps": int(metrics["charge_route_found_steps"]),
            "fail_reason": str(result_details["fail_reason"]),
        }

    def _build_window_metrics(self):
        records = list(self.recent_episode_records)
        if not records:
            return {
                "window_episode_count": 0,
                "episode_return_mean": 0.0,
                "episode_return_max": 0.0,
                "episode_return_min": 0.0,
                "completed_rate": 0.0,
                "battery_depleted_rate": 0.0,
                "first_charge_success_rate": 0.0,
                "clean_ratio_mean": 0.0,
                "first_charge_step_mean": 0.0,
                "return_trigger_margin_mean": 0.0,
                "dock_contact_without_charge_mean": 0.0,
                "near_dock_entry_rate": 0.0,
                "dock_success_given_contact_rate": 0.0,
                "dock_stall_steps_mean": 0.0,
                "dock_regress_count_mean": 0.0,
                "charge_success_after_dock_step_mean": 0.0,
                "route_stall_steps_total_mean": 0.0,
                "charge_route_found_rate_weighted": 0.0,
                "npc_collision_rate": 0.0,
                "abnormal_truncated_rate": 0.0,
                "unknown_failure_rate": 0.0,
                "other_failure_rate": 0.0,
            }

        episode_returns = [record["episode_return"] for record in records]
        episode_count = len(records)
        charge_guidance_steps_sum = sum(record["charge_guidance_steps"] for record in records)
        charge_route_found_steps_sum = sum(record["charge_route_found_steps"] for record in records)
        near_dock_entries_sum = sum(record["near_dock_entries"] for record in records)
        dock_contact_entries_sum = sum(record["dock_contact_entries"] for record in records)
        dock_success_after_contact_sum = sum(record["dock_success_after_contact_count"] for record in records)
        charge_success_after_dock_step_sum = sum(record["charge_success_after_dock_step_sum"] for record in records)

        known_failure_reasons = {
            "completed_max_step",
            "battery_depleted",
            "npc_collision",
            "abnormal_truncated",
            "unknown_failure",
        }
        other_failure_count = sum(
            1
            for record in records
            if not record["completed"] and record["fail_reason"] not in known_failure_reasons
        )

        return {
            "window_episode_count": episode_count,
            "episode_return_mean": round(self._mean(episode_returns), 4),
            "episode_return_max": round(max(episode_returns), 4),
            "episode_return_min": round(min(episode_returns), 4),
            "completed_rate": round(self._mean([record["completed"] for record in records]), 4),
            "battery_depleted_rate": round(self._mean([record["battery_depleted"] for record in records]), 4),
            "first_charge_success_rate": round(self._mean([record["first_charge_success"] for record in records]), 4),
            "clean_ratio_mean": round(self._mean([record["clean_ratio"] for record in records]), 4),
            "first_charge_step_mean": round(self._mean_valid("first_charge_step", {-1}), 4),
            "return_trigger_margin_mean": round(self._mean_valid("return_trigger_margin", {-999}), 4),
            "dock_contact_without_charge_mean": round(
                self._mean([record["dock_contact_without_charge"] for record in records]), 4
            ),
            "near_dock_entry_rate": round(
                sum(1 for record in records if record["near_dock_entries"] > 0) / float(episode_count), 4
            ),
            "dock_success_given_contact_rate": round(
                float(dock_success_after_contact_sum) / float(dock_contact_entries_sum)
                if dock_contact_entries_sum > 0
                else 0.0,
                4,
            ),
            "dock_stall_steps_mean": round(self._mean([record["dock_stall_steps_total"] for record in records]), 4),
            "dock_regress_count_mean": round(self._mean([record["dock_regress_count"] for record in records]), 4),
            "charge_success_after_dock_step_mean": round(
                float(charge_success_after_dock_step_sum) / float(dock_success_after_contact_sum)
                if dock_success_after_contact_sum > 0
                else 0.0,
                4,
            ),
            "route_stall_steps_total_mean": round(
                self._mean([record["route_stall_steps_total"] for record in records]), 4
            ),
            "charge_route_found_rate_weighted": round(
                float(charge_route_found_steps_sum) / float(charge_guidance_steps_sum)
                if charge_guidance_steps_sum > 0
                else 0.0,
                4,
            ),
            "npc_collision_rate": round(
                sum(1 for record in records if record["fail_reason"] == "npc_collision") / float(episode_count),
                4,
            ),
            "abnormal_truncated_rate": round(
                sum(1 for record in records if record["fail_reason"] == "abnormal_truncated") / float(episode_count),
                4,
            ),
            "unknown_failure_rate": round(
                sum(1 for record in records if record["fail_reason"] == "unknown_failure") / float(episode_count),
                4,
            ),
            "other_failure_rate": round(float(other_failure_count) / float(episode_count), 4),
        }

    @staticmethod
    def _format_optional_metric(value, digits=3):
        if value is None:
            return "None"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    def _log_episode_result(self, step, result_str, total_reward, final_reward, episode_return, result_details, metrics):
        self.logger.info(
            "[EPISODE_RESULT] "
            f"ep:{self.episode_cnt} steps:{step} result:{result_str} reason:{result_details['fail_reason']} "
            f"training_stage:{result_details['training_stage_name']} "
            f"episode_return:{episode_return:.3f} step_reward_sum:{total_reward:.3f} final_reward:{final_reward:.3f} "
            f"completed:{int(metrics['completed'])} clean_ratio:{metrics['clean_ratio']:.4f} "
            f"first_charge_success:{int(metrics['first_charge_success'])} battery_depleted:{int(metrics['battery_depleted'])} "
            f"charge_count:{metrics['charge_count']} remaining_charge:{metrics['remaining_charge']} "
            f"explored_ratio:{metrics['explored_ratio']:.4f} total_score:{result_details['total_score']} "
            f"clean_score:{result_details['clean_score']}"
        )

    def _log_episode_diagnostics(self, snapshot, action_debug, result_details):
        self.logger.info(
            "[EPISODE_DIAG] "
            f"ep:{self.episode_cnt} pos:{snapshot['pos']} visit:{snapshot['current_visit']} "
            f"training_stage:{snapshot.get('training_stage_name')} "
            f"nearest_charger:{snapshot['nearest_charger_dist']} nearest_npc:{snapshot['nearest_npc_dist']} "
            f"return_mode:{int(snapshot['return_mode'])} return_reason:{snapshot['return_reason']} "
            f"return_trigger_step:{snapshot['return_trigger_step']} "
            f"return_trigger_margin:{snapshot['return_trigger_margin']} "
            f"first_return_trigger_step:{snapshot['first_return_trigger_step']} "
            f"first_charge_step:{snapshot['first_charge_step']} first_charge_stage:{snapshot['first_charge_stage']} "
            f"battery_margin:{snapshot['battery_margin']} min_battery_margin:{snapshot['min_battery_margin']} "
            f"battery_margin_at_first_charge:{snapshot['battery_margin_at_first_charge']} "
            f"low_battery_steps:{snapshot['low_battery_steps']} "
            f"low_battery_route_progress_mean:{self._format_optional_metric(snapshot.get('low_battery_route_progress_mean'))} "
            f"charge_route_found:{int(snapshot['charge_route_found'])} "
            f"charge_route_found_rate:{snapshot['charge_route_found_rate']:.3f} "
            f"charge_guidance_steps:{snapshot['charge_guidance_steps']} "
            f"charge_route_found_steps:{snapshot['charge_route_found_steps']} "
            f"near_dock_entries:{snapshot.get('near_dock_entries', 0)} "
            f"dock_contact_entries:{snapshot.get('dock_contact_entries', 0)} "
            f"dock_success_after_contact_count:{snapshot.get('dock_success_after_contact_count', 0)} "
            f"charge_success_after_dock_step_sum:{snapshot.get('charge_success_after_dock_step_sum', 0)} "
            f"dock_stall_steps_total:{snapshot.get('dock_stall_steps_total', 0)} "
            f"dock_regress_count:{snapshot.get('dock_regress_count', 0)} "
            f"charge_progress_delta:{self._format_optional_metric(snapshot.get('charge_route_progress_delta'), digits=2)} "
            f"route_stall_steps_total:{snapshot['route_stall_steps_total']} "
            f"max_charge_stall_steps:{snapshot['max_charge_stall_steps']} "
            f"dock_contact_without_charge:{snapshot['dock_contact_without_charge']} "
            f"action_source:{action_debug['source']} action_sampled:{action_debug['sampled_action']} "
            f"action_selected:{action_debug['selected_action']} result_code:{result_details['result_code']} "
            f"result_message:{result_details['result_message']}"
        )

    def _log_window_metrics(self, window_metrics, active_stage_info, stage_info):
        self.logger.info(
            "[EVAL_WINDOW] "
            f"episodes:{window_metrics['window_episode_count']} "
            f"episode_return_mean:{window_metrics['episode_return_mean']:.4f} "
            f"completed_rate:{window_metrics['completed_rate']:.4f} "
            f"battery_depleted_rate:{window_metrics['battery_depleted_rate']:.4f} "
            f"first_charge_success_rate:{window_metrics['first_charge_success_rate']:.4f} "
            f"clean_ratio_mean:{window_metrics['clean_ratio_mean']:.4f} "
            f"first_charge_step_mean:{window_metrics['first_charge_step_mean']:.4f} "
            f"return_trigger_margin_mean:{window_metrics['return_trigger_margin_mean']:.4f} "
            f"charge_route_found_rate_weighted:{window_metrics['charge_route_found_rate_weighted']:.4f} "
            f"near_dock_entry_rate:{window_metrics['near_dock_entry_rate']:.4f} "
            f"dock_success_given_contact_rate:{window_metrics['dock_success_given_contact_rate']:.4f} "
            f"dock_stall_steps_mean:{window_metrics['dock_stall_steps_mean']:.4f} "
            f"dock_regress_count_mean:{window_metrics['dock_regress_count_mean']:.4f} "
            f"charge_success_after_dock_step_mean:{window_metrics['charge_success_after_dock_step_mean']:.4f} "
            f"dock_contact_without_charge_mean:{window_metrics['dock_contact_without_charge_mean']:.4f} "
            f"route_stall_steps_total_mean:{window_metrics['route_stall_steps_total_mean']:.4f}"
        )
        self.logger.info(
            "[TRAIN_STAGE] "
            f"episodes:{window_metrics['window_episode_count']} "
            f"active_stage:{active_stage_info['stage_name']}({active_stage_info['stage_label']}) "
            f"next_stage:{stage_info['stage_name']}({stage_info['stage_label']}) "
            f"next_stage_id:{stage_info['stage_id']} "
            f"reasons:{' | '.join(stage_info['reasons'])}"
        )
        self.logger.info(
            "[EVAL_FAILURE] "
            f"episodes:{window_metrics['window_episode_count']} npc_collision_rate:{window_metrics['npc_collision_rate']:.4f} "
            f"abnormal_truncated_rate:{window_metrics['abnormal_truncated_rate']:.4f} "
            f"unknown_failure_rate:{window_metrics['unknown_failure_rate']:.4f} "
            f"other_failure_rate:{window_metrics['other_failure_rate']:.4f}"
        )

    def run_episodes(self):
        """
        持续运行 episode，并在每局结束后产出一批训练样本。

        这是训练主循环：
        1. 重置环境，载入最新模型。
        2. 持续执行 `obs -> act -> env.step -> next_obs`。
        3. 收集 shaped reward 和 value/prob 等训练字段。
        4. 在终局时补发 final reward，并输出核心监控指标。

        这里会把 step reward 和 final reward 分开处理：
        - step reward 来自 `Preprocessor.reward_process()`，负责逐步塑造行为。
        - final reward 只在终局加到最后一帧，负责把整局成败明确反馈给 PPO。
        """
        while True:
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics: {training_metrics}")

            env_obs = self.env.reset(self.usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            # 每局开始都重新加载一次最新模型，确保 actor 侧尽快跟上 learner 已更新的参数。
            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")
            active_stage_info = self.agent.get_training_stage()
            active_stage_info["stage_label"] = TRAINING_STAGE_LABELS.get(
                active_stage_info["stage_name"], active_stage_info["stage_name"]
            )

            obs_data, _ = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0

            self.logger.info(
                f"Episode {self.episode_cnt} start, feature_dim={Config.DIM_OF_OBSERVATION}, "
                f"local_view={Config.LOCAL_VIEW_SIZE}, global_state={Config.GLOBAL_FEATURE_SIZE}, "
                f"training_stage={active_stage_info['stage_name']}"
            )

            while not done:
                # 先基于当前观测产出动作，再推动环境执行一步。
                act_data_list = self.agent.predict([obs_data])
                if not act_data_list:
                    self.logger.error(
                        f"Episode {self.episode_cnt} predict returned no action data at step {step}, abort episode"
                    )
                    break

                act_data = act_data_list[0]
                act = self.agent.action_process(act_data)

                env_reward, env_obs = self.env.step(act)
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                frame_no = env_obs["frame_no"]
                step += 1
                done = terminated or truncated

                next_obs_data, _ = self.agent.observation_process(env_obs)
                next_obs_data.frame_no = frame_no

                # 训练写样本时使用的是 preprocessor 给出的 shaped reward，而不是环境原始 reward。
                # 这样奖励设计可以完全围绕“清扫-探索-回充”这套学习目标来做。
                reward_scalar = float(self.agent.last_reward)
                total_reward += reward_scalar

                final_reward = 0.0
                if done:
                    fm = self.agent.preprocessor
                    result_details = _extract_result_details(env_obs, fm, truncated, step)
                    final_reward, result_str = _compute_final_reward(result_details)
                    metrics = _build_episode_metrics(result_details)
                    episode_return = total_reward + final_reward
                    self.recent_episode_records.append(
                        self._build_episode_record(episode_return, result_details, metrics)
                    )
                    window_metrics = self._build_window_metrics()
                    stage_info = _classify_training_stage(window_metrics)
                    self.agent.set_training_stage(stage_info["stage_name"], stage_info["stage_id"])
                    snapshot = result_details["snapshot"]
                    action_debug = self.agent.get_action_debug_snapshot()

                    self._log_episode_result(
                        step=step,
                        result_str=result_str,
                        total_reward=total_reward,
                        final_reward=final_reward,
                        episode_return=episode_return,
                        result_details=result_details,
                        metrics=metrics,
                    )
                    self._log_episode_diagnostics(snapshot, action_debug, result_details)

                # 单步样本先按 step reward 落盘；若终局，再把 final reward 叠到最后一帧。
                reward_arr = np.array([reward_scalar], dtype=np.float32)
                value_arr = act_data.value.flatten()[: Config.VALUE_NUM]

                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array(act_data.action),
                    reward=reward_arr,
                    done=np.array([float(done)]),
                    reward_sum=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    value=value_arr,
                    next_value=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    advantage=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                )
                collector.append(frame)

                if done:
                    collector[-1].reward = collector[-1].reward + np.array([final_reward], dtype=np.float32)

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        # 监控侧只上报高价值的整局指标，避免面板里被大量中间字段淹没。
                        monitor_payload = {
                            "episode_return": episode_return,
                            "episode_cnt": self.episode_cnt,
                            "clean_ratio": metrics["clean_ratio"],
                            "charge_count": metrics["charge_count"],
                            "first_charge_success": metrics["first_charge_success"],
                            "battery_depleted": metrics["battery_depleted"],
                            "remaining_charge": metrics["remaining_charge"],
                            "completed": metrics["completed"],
                            "explored_ratio": metrics["explored_ratio"],
                            "nearest_charger_dist": metrics["nearest_charger_dist"],
                            "first_charge_step": metrics["first_charge_step"],
                            "return_trigger_step": metrics["return_trigger_step"],
                            "return_trigger_margin": metrics["return_trigger_margin"],
                            "first_return_trigger_step": metrics["first_return_trigger_step"],
                            "min_battery_margin": metrics["min_battery_margin"],
                            "battery_margin_at_first_charge": metrics["battery_margin_at_first_charge"],
                            "low_battery_steps": metrics["low_battery_steps"],
                            "low_battery_route_progress_mean": metrics["low_battery_route_progress_mean"],
                            "charge_route_found_rate": metrics["charge_route_found_rate"],
                            "route_stall_steps_total": metrics["route_stall_steps_total"],
                            "max_charge_stall_steps": metrics["max_charge_stall_steps"],
                            "dock_contact_without_charge": metrics["dock_contact_without_charge"],
                            "charge_guidance_steps": metrics["charge_guidance_steps"],
                            "charge_route_found_steps": metrics["charge_route_found_steps"],
                        }
                        monitor_payload.update(window_metrics)
                        monitor_payload.update(stage_info["monitor_metrics"])
                        monitor_payload.update(
                            {
                                "active_training_stage_id": float(active_stage_info["stage_id"]),
                                "active_training_stage_early": 1.0 if active_stage_info["stage_name"] == TRAINING_STAGE_EARLY else 0.0,
                                "active_training_stage_middle": 1.0 if active_stage_info["stage_name"] == TRAINING_STAGE_MIDDLE else 0.0,
                                "active_training_stage_late": 1.0 if active_stage_info["stage_name"] == TRAINING_STAGE_LATE else 0.0,
                            }
                        )
                        self.monitor.put_data(
                            {
                                os.getpid(): monitor_payload
                            }
                        )
                        self._log_window_metrics(window_metrics, active_stage_info, stage_info)
                        self.last_report_monitor_time = now

                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                obs_data = next_obs_data
