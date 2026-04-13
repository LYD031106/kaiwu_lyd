# Kaiwu Robot Vacuum PPO Training

这个仓库用于共享腾讯 Kaiwu `robot_vacuum` 项目的 PPO 训练代码，目标是让别人拉取代码后，在相同训练环境下直接启动训练，并得到接近的训练趋势。

当前共享代码基于本地验证过的主版本，训练逻辑入口在：

- `code/agent_ppo/agent.py`
- `code/agent_ppo/feature/preprocessor.py`
- `code/agent_ppo/workflow/train_workflow.py`

## 运行前提

- Windows 主机
- Docker Desktop / Docker Compose
- NVIDIA GPU，并且 Docker 已开启 GPU 支持
- 能拉取 `kaiwu-pub.tencentcloudcr.com` 镜像
- 自己的 `license.dat`
- 自己的 Kaiwu 平台配置：`KAIWU_PLAYER_ID`、`KAIWU_TASK_UUID`、`TRACKER_ID`、`MONITOR_ID` 等

## 目录说明

- `code/`：训练代码与配置
- `train/`：训练容器编排、训练日志、模型导出目录
- `dev/`：开发容器编排

## 当前训练配置

当前环境配置见 `code/agent_ppo/conf/train_env_conf.toml`，主要参数如下：

- `algorithm = ppo`
- `map = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
- `robot_count = 4`
- `charger_count = 4`
- `max_step = 1000`
- `battery_max = 200`

