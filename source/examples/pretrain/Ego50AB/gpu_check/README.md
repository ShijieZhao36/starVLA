# 三源真实混合 batch GPU 检查

`check_three_source_gpu.py` 读取 CPU audit 生成的可信 pickle 样本，在**同一个 batch**放入 Galaxea、LIBERO、Xperience。默认每源 1 个真实样本，不补图、不造动作、不复制样本。保留三源各自 3／2／1 视角、41／10／48 有效动作维度。

脚本从 config 指定的原始 VLM 初始化完整模型，执行真实 forward、backward 和 prediction；不创建优化器、不更新权重、不保存 checkpoint。它验证混合输入兼容性和有限梯度，不作为生产 batch=8 的显存／吞吐承诺。

从远端实验 checkout 内运行，使用 root 确认空闲的单张 GPU 和 MRA Python：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 \
  /root/miniconda3/envs/MRA/bin/python \
  examples/pretrain/Ego50AB/train_files/check_three_source_gpu.py \
  --config /absolute/path/to/arm_config.yaml \
  --samples /absolute/path/to/three_source_samples.pkl \
  --output /absolute/path/to/three_source_gpu.json
```

脚本位置由主 agent 部署位置决定；以上路径仅为示例。可用 `--samples-per-source 2` 执行 6 个不同真实样本，但不必为启动检查扩大量。默认同时检查输出 `[3,8,90]`、有限值和 masked action 预测为零。`--skip-prediction` 可只运行训练路径。JSON 分阶段原子写入，失败时保留错误和 traceback；另记录模型/可训练参数数目、VLM 与动作头梯度覆盖、CUDA 峰值内存、版本、输入哈希与配置哈希。
