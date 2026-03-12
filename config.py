from dataclasses import dataclass


@dataclass
class TrainDefaults:
    """训练默认参数 / Training defaults."""

    use_train_folder: bool = False  # 是否读取 train_folder 训练 / Use train_folder instances during training
    gen_instances: int = 6400  # 每轮动态生成实例数 / Number of generated instances per epoch
    n_customers: int = 100  # 每个实例的最大客户数 / Max customers per instance
    epochs: int = 150  # 训练轮数 / Number of training epochs
    train_folder: str | None = "solomon_data"  # 训练实例目录 / Training instance folder
    solve_folder: str | None = "solomon_data"  # 训练后评估目录 / Evaluation folder after training

    gen_dist_type: str = "auto"  # 动态实例空间分布 / Generated spatial type: auto/C/R/RC
    gen_tw_type: str = "auto"  # 动态实例时间窗类型 / Generated time-window type: auto/1/2
    gen_seed: int = 42  # 动态生成随机种子 / Random seed for generated instances
    gen_customers: int | None = None  # 动态实例客户数覆盖 / Override customer count for generated instances
    gen_async: bool = True  # 是否异步预取下一轮实例 / Asynchronously prefetch next epoch instances

    train_bin: str | None = None  # 预制训练集路径 / Prebuilt training dataset path
    save_bin: str | None = None  # 保存生成数据集路径 / Save generated dataset path
    use_train_shards: bool = True  # 是否优先使用 shards / Prefer shard-based training data
    train_shards_dir: str | None = "pt/der640k_shards"  # shard 目录 / Shard directory
    train_shards_count: int = 100  # 最多加载多少个 shard / Max number of shards to load
    train_shard_shuffle: bool = True  # 每轮是否打乱 shard 内样本 / Shuffle samples inside each shard per epoch

    batch_size: int = 64  # 训练批大小 / Training batch size

    device: str = "cuda"  # 训练设备 / Training device: cuda or cpu
    use_bf16: bool = True  # 是否启用 BF16 自动混合精度 / Enable CUDA BF16 autocast when available
    out: str = "neural_solution.json"  # 默认输出路径 / Default output path
    log: str = "train_log.csv"  # 训练日志路径 / Training log CSV path

    lr: float = 3e-4  # 初始学习率 / Initial learning rate
    lr_schedule: str = "cosine"  # 学习率调度 / LR schedule: fixed/cosine/linear/multistep
    lr_milestones: tuple[float, ...] = (0.8, 0.95)  # multistep 里程碑 / Milestones for multistep LR
    lr_gamma: float = 0.1  # multistep 衰减系数 / Decay factor for multistep LR
    entropy_coef: float = 0.0003  # 熵正则系数 / Entropy regularization coefficient
    resume: str | None = None  # 断点恢复路径 / Resume checkpoint path
    latent_multi_k: int = 32  # 并行 latent rollout 数 / Number of parallel latent rollouts
    max_grad_norm: float = 15000.0  # 梯度裁剪阈值 / Gradient clipping threshold

    model_embed_dim: int = 128  # 模型嵌入维度 / Model embedding dimension
    model_n_heads: int = 8  # 注意力头数 / Number of attention heads
    model_n_layers: int = 3  # 编码器层数 / Number of encoder layers
    model_ff_dim: int = 0  # FFN 隐藏维度，0 表示 4x embed / FFN hidden dim, 0 means 4x embed_dim

    latent_dim: int = 16  # 解码 latent 维度 / Decoder latent dimension
    use_greedy_start_pool: bool = False  # 是否启用 greedy 起点池 / Enable greedy start-pool rollout
    use_dynamic_key_aug: bool = True  # 是否启用动态 key 增强 / Enable dynamic key augmentation
    use_raw_feature_bias: bool = False  # 是否启用原始特征偏置 / Enable raw static-feature bias
    dyn_dim: int | None = None  # 动态特征维度覆盖 / Override dynamic feature dimension
    raw_feat_dim: int | None = None  # 原始特征偏置维度覆盖 / Override raw-feature bias dimension
    cand_phi_dim: int = 9  # 候选动态特征维度 / Candidate dynamic feature dimension
    cand_phi_hidden_dim: int = 0  # 候选特征投影隐藏层维度 / Hidden dim of candidate feature projection
    depot_due_sentinel_threshold: float = 1e5  # 回仓截止时间哨兵阈值 / Sentinel threshold for depot due time
    latent_max_sigma: float = 0.7  # latent 噪声初始标准差 / Initial latent noise sigma
    latent_sigma_decay: float = 0.97  # latent 噪声衰减系数 / Latent noise decay factor
    latent_noise_mode: str = "adaptive"  # latent 噪声模式 / Latent noise mode: schedule or adaptive
    latent_adaptive_sigma_init: float = 0.0  # adaptive 初始 sigma / Initial sigma for adaptive mode
    latent_adaptive_sigma_min: float = 0.1  # adaptive sigma 下界 / Lower sigma bound for adaptive mode
    latent_adaptive_sigma_max: float = 1.0  # adaptive sigma 上界 / Upper sigma bound for adaptive mode
    latent_adaptive_improve_tol: float = 0.002  # mean_cost 改进阈值 / Relative mean_cost improvement threshold
    latent_adaptive_downscale: float = 0.92  # 改进时缩小倍率 / Sigma multiplier after improvement
    latent_adaptive_upscale: float = 1.10  # 停滞时放大倍率 / Sigma multiplier after stagnation
    latent_adaptive_patience: int = 1  # adaptive 停滞容忍轮数 / Patience before sigma upscale
    gate_init_bias: float = 2.0  # 解码门控初始偏置 / Initial bias of the decoder gate
    diag_log_every: int = 100  # 训练诊断输出间隔 / Diagnostic print interval in updates

    vehicle_max: int | None = 25  # 车辆上限 / Vehicle upper bound


@dataclass
class SolveDefaults:
    """求解默认参数 / Inference and solve defaults."""

    instance: str = "solomon_data/c103.txt"  # 单实例路径 / Single-instance path
    instances_dir: str = "solomon_data"  # 批量实例目录 / Directory for batch solving
    pt: str = "neural_solution3676.pt"  # 模型权重路径 / Model checkpoint path
    out_all: str = "results_solomon56_full_no_ls.json"  # 批量结果输出 / Batch output JSON
    n_customers: int | None = None  # 求解时客户截断 / Optional customer truncation at solve time

    solve_use_latent_multi: bool = False  # 是否启用 latent 多样化 / Enable latent-multi decoding
    solve_decode_mode: str = "greedy"  # 解码模式 / Decode mode: greedy or lookahead
    solve_use_aug8: bool = False  # 是否启用 aug8 / Enable aug8 inference
    solve_use_local_search: bool = False  # 是否启用局部搜索 / Enable local search
    solve_use_eas: bool = True  # 是否启用 EAS / Enable EAS adaptation

    ls_budget_sec: float = 0.5  # 局部搜索时间预算 / Local-search time budget in seconds
    ls_parallel: bool = True  # 局部搜索并行 / Enable parallel local search
    ls_parallel_workers: int | None = 6  # 局部搜索 worker 数 / Number of local-search workers
    decode_max_steps: int | None = 150  # 最大解码步数 / Max decoding steps
    lookahead_confident_prob: float = 0.95  # 高置信阈值 / Confidence threshold for direct greedy
    lookahead_top_k: int = 3  # 前瞻展开 top-k / Top-k branches for lookahead
    solve_parallel_aug8_workers: int = 3  # aug8 并发 worker / Worker count for aug8
    solve_parallel_instance_workers: int = 1  # 实例级并发 worker / Parallel workers across instances

    latent_multi_k: int = 32  # 求解阶段 latent rollout 数 / Number of latent rollouts at solve time
    device: str = "cuda"  # 求解设备 / Solve device: cuda or cpu
    use_bf16: bool = True  # 是否启用 BF16 / Enable CUDA BF16 autocast when available
    latent_dim: int | None = None  # latent 维度覆盖 / Override latent dimension
    use_dynamic_key_aug: bool = False  # 是否启用动态 key 增强 / Enable dynamic key augmentation at solve time
    use_raw_feature_bias: bool = False  # 是否启用原始特征偏置 / Enable raw-feature bias at solve time
    cand_phi_dim: int = 9  # 候选动态特征维度 / Candidate dynamic feature dimension
    cand_phi_hidden_dim: int = 0  # 候选特征隐藏层维度 / Candidate feature hidden dimension

    latent_search_samples: int = 1  # latent 随机采样次数 / Number of random latent samples

    eas_steps: int = 40  # EAS 更新步数 / Number of EAS optimization steps
    eas_rollout_batch: int = 1536  # 每步 EAS rollout 数 / Number of rollouts per EAS step
    eas_lr: float = 5e-5  # EAS 学习率 / Learning rate for EAS
    eas_entropy_coef: float = 0.001  # EAS 熵正则 / Entropy regularization for EAS


@dataclass
class PlotDefaults:
    """绘图默认参数 / Plotting defaults."""

    log: str = "train_log.csv"  # 训练日志路径 / Training log CSV path
    instance: str = "solomon_data/c101.txt"  # 默认绘图实例 / Default plotting instance
    solution: str = "all_results.json"  # 默认结果 JSON / Default solution JSON
    solution_name: str | None = "c101"  # 结果中的实例名 / Instance name inside the result JSON
    n_customers: int | None = 100  # 绘图时客户截断 / Optional customer truncation for plotting


@dataclass
class ExportDefaults:
    """导出默认参数 / Export defaults."""

    json: str = "results_solomon56_full_no_ls.json"  # 待导出 JSON / Input JSON to export
    out: str = "greedy.xlsx"  # 导出目标文件 / Output spreadsheet path


train_defaults = TrainDefaults()
solve_defaults = SolveDefaults()
plot_defaults = PlotDefaults()
export_defaults = ExportDefaults()
