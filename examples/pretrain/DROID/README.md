# DROID → LIBERO Plus

默认在 `neo-bj4` 单机 8 卡运行，使用 CRA 环境。所有入口默认 `DRY_RUN=1`，显式设置 `DRY_RUN=0` 才执行。配置集中在 `train_files/experiment.env`，可通过环境变量覆盖。

## 已确认协议

- Qwen3-VL-4B-Instruct + QwenPI DiT，decoder LayerNorm，canonical 90 维，horizon 8，`include_state=false`，seed 42。
- DROID 预训练 30,000 步，LIBERO Plus 后训练 30,000 步；8 卡 × 每卡 batch 8 = global batch 64；warmup 分别为 2,500 / 5,000 步，每 5,000 步保存。
- VLM LR `1e-5`，动作头 LR `1e-4`。W&B 离线记录，目标项目为 `jasperz-georgia-institute-of-technology/pretrain`。
- 预训练从原始 Qwen3-VL 加新动作头开始；后训练严格加载预训练最终完整模型，重置优化器及调度器，并使用 LIBERO 自己的统计量。
- 测评使用后训练 `final_model/pytorch_model.pt`，4 suites、10,030 episodes，8 个 policy server，每卡 4 个仿真进程，共 32 个。沿用已有环境 seed 7 和 policy noise 行为。

## 数据与动作语义

数据目录：`/mnt/zhaoshijie/dataset/oxe_franka_lerobot/droid_1.0.1/stage3_unmasked_20261021_142915/job/full/dataset`。

LeRobot v2.1，15 Hz，64,260 episodes、16,771,924 帧。磁盘上的 state/action 均为 17 维：

| 原始下标（从 0 开始） | 内容 | 单位/表示 |
| --- | --- | --- |
| `[0:3]` | 末端位置 xyz | 米，机器人基座坐标系中的绝对坐标 |
| `[3:9]` | 末端姿态 rot6d | 旋转矩阵前两行按行展开 |
| `[9:10]` | 夹爪 | 1 开、0 合 |
| `[10:17]` | 7 个关节角 | rad，绝对角度 |

state 是观测状态，action 是控制目标，并非 `state[t+1]`。末端参考 `panda_link8`；数据未声明物理夹爪尖端 TCP 标定。

训练监督映射至 canonical **右臂**：关节 `[32:39]`、末端 `[39:48]`、夹爪 `[48:49]`，共 17 维有效，其余 73 维不参与监督。EEF 目标在 loader 中转为相对 chunk 起始末端坐标系：`p_rel = R0.T @ (p_goal - p0)`、`R_rel = R0.T @ R_goal`。关节角和夹爪保持绝对目标。LIBERO 后训练只监督其自身有效的 10 维。

三路图像按 `front, exterior_2, wrist` 排列，全帧缩放至 256×256。uint8 缓存约 9 TiB，共 192,780 个 episode-camera 文件。

## 运行入口

以下命令在 `neo-bj4` 执行：

```bash
cd /mnt/zhaoshijie/workspace/starVLA
S=examples/pretrain/DROID/train_files

# 只预览完整命令。
PREPARE_DATA=1 DRY_RUN=1 bash "$S/run_pipeline.sh"

# 一键：统计量 → 缓存 → 预训练 → 后训练 → 完整测评。
PIPELINE_ID="droid_ln_8gpu_30k_$(date +%Y%m%d_%H%M%S)" \
  PREPARE_DATA=1 DRY_RUN=0 bash "$S/run_pipeline.sh"

# 数据已准备时，一键运行三个训练/测评阶段。
PIPELINE_ID="droid_ln_$(date +%Y%m%d_%H%M%S)" \
  DRY_RUN=0 bash "$S/run_pipeline.sh"

# 两个准备阶段也可以单独运行，均可重复执行并复用已验证的结果。
STATS_WORKERS=160 STATS_SUMMARY_WORKERS=160 DRY_RUN=0 bash "$S/precompute_droid_stats.sh"
CACHE_WORKERS=160 DRY_RUN=0 bash "$S/precompute_droid_frame_cache.sh"

# 各训练阶段的独立入口。
RUN_ID=droid_pretrain_unique DRY_RUN=0 bash "$S/run_droid_pretrain.sh"
PRETRAIN_RUN_DIR=/mnt/zhaoshijie/checkpoints/starVLA/droid_pretrain_unique \
  RUN_ID=droid_libero_unique DRY_RUN=0 bash "$S/run_libero_plus_posttrain.sh"
POSTTRAIN_RUN_DIR=/mnt/zhaoshijie/checkpoints/starVLA/droid_libero_unique \
  DRY_RUN=0 bash "$S/run_libero_plus_eval.sh"
```

## 并行准备与检查

统计量使用 160 个 episode 扫描 worker 和最多 160 个坐标归约 worker，每个 worker 单线程；两个阶段依次运行。仅扫描浮点数值列，以不重叠的 mmap 区间存储，临时空间约 10.8 GiB。绝对统计覆盖 18 个浮点字段 / 103 维；末端相对统计保留 `[8,9]` 形状。分位数基于全部样本精确计算，不取 episode 分位数平均。统计 fingerprint 与现有 loader 兼容，写入原子化且有锁。

缓存复用已有 `starVLA.dataloader.precompute_galaxea_frame_cache`，160 worker、单线程 CPU 解码、episode-camera 并行、每块 32 帧、`INTER_AREA` 缩放。完整缓存约 9 TiB，运行前应保证可用空间。完成全部文件后才发布 `droid_cache_complete.json`；重跑会复用已有完整文件。缓存根目录默认 `/mnt/zhaoshijie/frame_cache/droid_stage3_unmasked_256`。

任何阶段退出失败都停止后续阶段。训练前检查统计 fingerprint、缓存和 8 卡空闲；训练后检查有限 loss、最终步数、样本曝光数、统计 mask 和最终模型；后训练严格核对来源模型；测评核对全部 10,030 条唯一任务记录及汇总结果。训练目录不覆盖、不会自动恢复 optimizer。

`RUN_ROOT_DIR/launcher_logs/PIPELINE_ID/` 保存阶段日志、状态和退出码。两个模型目录默认分别为 `RUN_ROOT_DIR/PIPELINE_ID_pretrain`、`RUN_ROOT_DIR/PIPELINE_ID_libero_plus`；测评目录默认 `EVAL_ROOT_DIR/PIPELINE_ID_libero_plus_final`。训练完成标记为 `pretrain_success.json` / `posttrain_success.json`，测评完整校验后写 `eval_success.json`。长任务建议用 screen 后台运行。

缓存完整标记绑定元数据 hash。训练前 DROID 额外检查首尾缓存文件头，LIBERO 检查全部文件头；标记生成后若人工删除中间 DROID 缓存，需重新执行缓存脚本，不能只保留旧标记。

## W&B 同步

neo-bj4 的 W&B API 在线检查返回 HTTP 403，经用户确认改为 `WANDB_MODE=offline`。训练与评测不中断，网络恢复后对各 run 的离线目录执行 `wandb sync <offline-run-directory>`。本地 `metrics.jsonl` 和 `train.log` 同时保留。

## 主线归档与短程验证

本目录从 2026-11-08 修复后的实际 DROID 运行源码归档；包括 ANSI/Rich 日志换行兼容的完成校验，以及从已完成预训练继续后训练/测评的独立入口。数据定义、canonical 映射、正式训练步数与测评协议保持原样。所有模型、缓存、数据集、运行日志与离线 W&B 产物均位于仓库外。

脚本从自身位置定位当前 worktree，不要求 checkout 位于历史 `/mnt/zhaoshijie/workspace/starVLA`。`experiment.env` 保存当时机群默认路径，可通过环境变量或 `EXPERIMENT_ENV=/absolute/path/to/local.env` 覆写；切换机器时核对 `PYTHON_BIN`、`EVAL_PYTHON`、`BASE_VLM`、数据/缓存目录、`LIBERO_HOME` 和 `SIM_ENV_OVERLAY`。YAML 内的默认资源路径会被入口脚本的命令行参数覆盖。

`SMOKE_TEST=1` 显式选择独立验证预算：3 步 DROID → 3 步 LIBERO → `libero_goal` 第 0 个任务的一次 rollout。它仍使用 8 GPU、每卡 batch 8、严格继承完整预训练模型；warmup=1、每步记录、最终步保存。所有 run ID 自动带 `_smoke`，显式指定的评测输出目录也必须以 `_smoke` 结尾。smoke 标记带 `smoke=true`，正式后训练和评测拒绝把它作为正式来源。`SMOKE_TEST=0` 保持原来的 30k → 30k → 10,030 例。

```bash
# 在所选新 worktree 的根目录运行，使用新 PIPELINE_ID。
SMOKE_TEST=1 DRY_RUN=1 PIPELINE_ID=droid_integration_check \
  bash examples/pretrain/DROID/train_files/run_pipeline.sh
# 确认资源路径后，将 DRY_RUN=0 执行完整短程检查。
```
