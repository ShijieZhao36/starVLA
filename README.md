# StarVLA PI05：35 袜子子集、100k 实验代码归档

这是服务器 `starvla_pi05` 环境下本次实际运行的独立 PI05 实验源码快照。
实验：8 卡 × 单卡 batch 8（全局 64），从 PyTorch PI05 base 重新训练，目标 100,000 步。
预测 50 步绝对动作；14 个真实动作维参与损失，补齐到 32 维；启用 state 分词。
三路图像从原视频生成 224×224 等比例缩放补边缓存。VLM/Action 学习率均为 5e-5，
warmup 1,000 步后恒定，每 5,000 步保存，使用仓库 gr00t_sharded dataloader。

- `source/`：完整仓库工作区源码，包含此次未提交改动、35 子集注册和 gr00t 代码。
- `source/starVLA/model/framework/VLM4A/PI05.py`：PI05 入口；其继承的 PI0.py、
  OpenPI_ActionHead、openpi_transformers、训练器、数据加载器均在同一源码目录。
- `launch/train_100k_bs8.yaml`：本次正式输入配置；`config.full.yaml` 为展开后的配置。
- `launch/formal_command.json`、`launch_original_server.sh`：原服务器实际启动参数及命令。
- `launch/dataset_statistics.json`：本次训练采用的归一化统计。
- `source/examples/realRobots/ARX_X5_Socks35/train_files/data_registry/`：35 个子集名单、混合权重和字段配置。
- `preparation/`：相关审计、14 维掩码验证、224 缓存生成、batch 8 检查和绘图脚本。
  其中带 30k/40k 字样的配置和流水线属于历史准备阶段，本次正式配置以 `launch/` 的 100k 为准。
- `environment/`：原 `starvla_pi05` 的 Conda/Python 依赖版本和非敏感启动环境变量。
- `assets/`：实际使用的 PaliGemma tokenizer。
- `provenance/`：commit、工作区补丁、运行身份、外部资产路径和基础权重哈希。
- `manifest.json`：逐文件 SHA-256、大小和权限，以及原文件身份记录。

保留原始服务器绝对路径，未为了本地路径修改训练代码或配置。没有复制基础权重、
训练 checkpoint、视频数据、帧缓存或整个 Conda 环境。启动回执保留其原始采集时间，
不代表当前训练进度；快照时进度记录在 `provenance/capture.json`。
复制过程不启动训练或机器人，也不修改正在运行的实验。
