# Xperience decoder LayerNorm 诊断

模型补丁在 action decoder 第一层 Linear 前增加可选的
`LayerNorm(input_dim, elementwise_affine=False, eps=1e-6)`。
`framework.action_model.decoder_input_norm` 缺省为 false，Xperience 训练 YAML 显式设为 true。
LayerNorm 没有新增可训练参数；两层 Linear、ReLU、DiT block 和 Flow Matching loss 保持原计算结构。

## 脚本

- `fixed_batch/data_prepare.py`：通过生产 processor 采集固定真实小批次。
- `fixed_batch/cache_vlm.py`：冻结原始 Qwen3-VL，缓存各层特征。
- `fixed_batch/train_ab.py`：新初始化整个动作头，进行无 LN / 有 LN 对照。
- `fixed_batch/run_fixed_batch_validation.sh`：带时间戳和进程号输出目录的复现入口。
- `fixed_batch/precision_probe.py`：相同权重和输入的 BF16 / FP32 只读比较。
- `layer_probe/layer_probe.py`：逐层配对噪声追踪及特征导出。
- `layer_probe/prepare_readout.py`：保留 48 个有效输出维，并重建输入差分正对照。
- `layer_probe/ridge_readout.py`：在冻结特征上拟合独立线性读出器。
- `layer_probe/merge_confirmation_cache.py`：保留原训练/验证轮次，替换为新噪声确认集。
- `layer_probe/analyze_layer_probe.py`：汇总已有探针输出并绘图。

这些脚本面向本轮 Xperience 设置：8 个观测、动作窗口 8、canonical 宽度 90、
48 个有效维、36 层 VLM/DiT。逐层读出默认采集 t=0.5 的 96 轮，
ridge 按 64/16/16 轮划分训练/验证/测试；confirmation 工具要求另有 16 轮不重叠的新噪声。

## 固定小批次验证

在 neo-bj1 运行：

```bash
GPU=0 RUN_EXTENDED=1 RUN_LOW_LR=1 bash /mnt/zhaoshijie/workspace/starVLA/examples/pretrain/Xperience/diagnostics/fixed_batch/run_fixed_batch_validation.sh
```

默认只运行原 decoder 和 LN 各 600 步；RUN_EXTENDED=1 增加 LN 2000 步；
RUN_LOW_LR=1 同时运行 1e-4 和 3e-5 两个学习率的 2000 步对照。
每组从同一 seed 重新初始化。固定观测和真实动作，但每步重新采样噪声及时间。
VLM 冻结；参与前向的整个动作头训练；使用原生 masked Flow Matching loss。
参数及 AdamW 状态为 FP32，前向 BF16 autocast，warmup 100 后恒定 LR。
连续 20 步 ReLU 全失活且 decoder 第一层梯度为零时停止。

入口默认使用原仓库及 CRA Python，可通过 REPO、PYTHON_BIN、SOURCE_CONFIG、
BASE_VLM、RUN_ROOT 和 GPU 覆盖路径或设备。
SOURCE_CONFIG 默认读取历史 80k 实验的 config.full.yaml，以保留已验证实验的完整配置；
它只提供配置，动作头不加载该实验权重。train_ab.py 根据 --variant 显式设置 LN 开关。
BASE_VLM 默认从 REPO 下的 playground/Pretrained_models 读取。

输出目录在 /mnt/zhaoshijie/checkpoints/starVLA 下，格式为
xperience_fixed_batch_validation_年月日_时分秒_进程号，已有目录不会被复用。
结果目录中的脚本副本也支持显式设置 REPO 后重跑。
LN 组保存模型权重，不保存优化器状态；这些文件不能用于精确恢复优化器训练轨迹。

## 已有权重的逐层探针

以下例子读取已完成的 LN 2000 步实验，输出到一个新的时间戳目录：

```bash
REPO=/mnt/zhaoshijie/workspace/starVLA
DIAG="$REPO/examples/pretrain/Xperience/diagnostics"
RUN=/mnt/zhaoshijie/checkpoints/starVLA/xperience_ln_validation_20261104_134354
OUT="/mnt/zhaoshijie/checkpoints/starVLA/xperience_layer_probe_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/CRA/bin/python "$DIAG/layer_probe/layer_probe.py" \
  --repo "$REPO" --helper-dir "$DIAG/fixed_batch" \
  --config "$RUN/ln_2000/config.yaml" --cache "$RUN/features.pt" \
  --checkpoint "$RUN/ln_2000/action_head_model.pt" \
  --output "$OUT" --name ln2000
```

探针必须传入与权重配套的训练组 config.yaml。LN 无 affine 参数，因此
strict=True 加载权重也无法发现错误的 decoder_input_norm 设置；
探测 LN 权重时不要传缺少此开关的历史 source_config.full.yaml。

precision_probe.py 同样使用对应组的 config.yaml，并与 train_ab.py 保持同目录。
layer_probe.py 通过 --helper-dir 指定 fixed_batch 目录。
其余工具使用 --help 查看输入输出参数。逐层与精度探针不更新模型；
ridge 仅拟合独立读出器，不将其写回动作头。

## 已完成实验的范围

原实验补丁基于 c7f713a881db955bd1ac81b41812719b537247da，
来自 /mnt/zhaoshijie/workspace/starVLA_ln_validation_20261104_134354。
本目录保留已执行版本的计算逻辑；仅调整复现 shell 的默认仓库和 VLM 路径。

历史训练结果：
/mnt/zhaoshijie/checkpoints/starVLA/xperience_ln_validation_20261104_134354

历史逐层探针结果：
/mnt/zhaoshijie/checkpoints/starVLA/xperience_layer_probe_20261104_162029

本次对照中 LN 避免了完全 ReLU 失活，并恢复部分学习，尚未通过可靠去噪与动作重建验证。
2000 步是否不足尚未排除；10000 步对照、直接线性 decoder 和完整 Ego 重训均未执行。
旧 smoke 脚本、初始 ridge 网格版本、图表、报告、缓存与模型权重保留在历史结果目录。
