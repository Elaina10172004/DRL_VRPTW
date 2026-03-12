# VRPTW-DRL

中文：这是一个面向 VRPTW（Vehicle Routing Problem with Time Windows，带时间窗车辆路径问题）的深度强化学习项目，包含训练、推理、局部搜索和单实例适配相关代码。

English: This repository is a deep reinforcement learning project for VRPTW (Vehicle Routing Problem with Time Windows), including training, inference, local search, and single-instance adaptation code.

## Highlights / 项目特点

- `HeadGatedMultiheadAttention` encoder with a single-path FFN.
- REINFORCE training with `group_mean_only` leave-one-out advantage.
- Main decoding modes: `greedy` and `lookahead`.
- Supports local search, batch solving, and Solomon-format instances.

## Repository Layout / 仓库结构

- `train.py`: training entry / 训练入口
- `solve.py`: solving entry / 求解入口
- `neural_policy.py`: model, environment, and training loop / 模型、环境与训练主循环
- `vrptw_data.py`: Solomon parser and feasibility rules / Solomon 数据读取与可行性规则
- `local_search.py`: local search operators / 局部搜索算子
- `cost_utils.py`: cost utilities / 成本函数
- `config.py`: configuration defaults / 默认配置
- `plot_loss.py`: plotting utilities / 绘图工具
- `evaluate.py`: evaluation helpers / 评估辅助脚本
- `export_results.py`: result export helpers / 结果导出脚本
- `solo.py`: Solomon-format conversion helpers / Solomon 格式转换辅助脚本

## Included Data / 仓库内数据

This repository is published with data files included.

- `solomon_data/`: Solomon benchmark instances
- `VRPTW.csv`: project data file used by the codebase
- `solomon56_parallel_aug8_latent32_lookahead_benchmark.csv`: benchmark output example

Local checkpoints and temporary artifacts are not tracked by default.

- `pt/`
- `*.pt`
- `*.pth`
- training logs and result JSON files matched by `.gitignore`

## Environment / 环境准备

Recommended: Python 3.10+.

Install dependencies / 安装依赖:

```bash
pip install -r requirements.txt
```

## Quick Start / 快速开始

Train a model / 训练模型:

```bash
python train.py --epochs 10 --batch_size 64 --n_customers 100
```

Note / 说明:

- It is usually better to generate the dataset ahead of training instead of generating it on the critical path of every epoch.
- 更推荐先生成训练数据，再开始正式训练，否则数据生成速度会拖慢训练过程。

Solve one instance / 单实例求解:

```bash
python solve.py --instance solomon_data/C101.txt --pt pt/best.pt
```

Solve a folder of instances / 批量求解:

```bash
python solve.py --instances_dir solomon_data --pt pt/best.pt --out_all results.json
```

If you do not have a checkpoint yet, train one first or replace `pt/best.pt` with your own model path.

## Open Source Notes / 开源说明

- The repository is released under the MIT License.
- Data files currently present in the repository are intended to be distributed together with the code.
- `.gitignore` excludes local weights, caches, and generated experiment artifacts so the public repository stays clean.

## License / 许可证

This project is licensed under the MIT License. See `LICENSE` for details.
