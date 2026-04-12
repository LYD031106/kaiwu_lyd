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

## 快速开始

### 1. 准备训练环境变量

复制环境样板：

```powershell
Copy-Item train/.env.example train/.env
```

然后修改 `train/.env`，至少替换这些字段：

- `KAIWU_LICENSE_PATH`
- `KAIWU_TASK_UUID`
- `KAIWU_PLAYER_ID`
- `KAIWU_PUBLIC_KEY`
- `USER_ID`
- `MONITOR_ID`
- `TRACKER_ID`

如果你的 `license.dat` 放在仓库根目录，`KAIWU_LICENSE_PATH=../license.dat` 可以保持不变。

### 2. 启动训练

```powershell
Set-Location train
docker compose --env-file .env -f .docker-compose.yaml --profile distributed up -d
```

### 3. 查看训练日志

```powershell
docker compose --env-file .env -f .docker-compose.yaml logs -f learner
```

训练日志会落在 `train/log/`。

### 4. 查看监控页面

默认监控端口来自 `train/.env` 中的 `KAIWU_MONITOR_PORT`，默认值是 `11000`：

```text
http://127.0.0.1:11000
```

### 5. 停止训练

```powershell
docker compose --env-file .env -f .docker-compose.yaml --profile distributed down
```

## 模型与权重

- 训练过程中的导出包默认在 `train/backup_model/`
- `code/agent_ppo/workflow/train_workflow.py` 中默认每 1800 秒触发一次 `save_model`
- 模型文件命名格式为 `model.ckpt-<step>.pkl`

如果你要从已有权重继续训练，可以启用 `code/conf/configure_app.toml` 里的 `preload_model=true`，然后把权重放到 `code/agent_ppo/ckpt/`。

## 结果预期

这个项目的训练结果不能保证逐局一致。影响因素包括：

- 地图与环境随机性
- PPO 采样随机性
- GPU 与容器环境带来的非严格确定性

合理预期是训练趋势接近，而不是每一轮都和原始实验完全相同。建议至少观察几个小时，再对比以下指标走势：

- `charge_count`
- `finished_steps`
- `total_score`

## 可选：启动开发环境

如果只是想进入 Kaiwu 开发容器，可以先复制：

```powershell
Copy-Item dev/.env.example dev/.env
```

再启动：

```powershell
Set-Location dev
docker compose --env-file .env -f .docker-compose.yaml up -d
```
