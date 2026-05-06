This package contains:
- benchmark_common.py: shared benchmark engine
- yolo26_benchmark_master.py: benchmark all benchmark + proposed optimizers
- run_<optimizer>_benchmark.py: compare one proposed optimizer against all benchmark optimizers

Key features:
- all parameters are defined before training
- optimizer-specific parameters are configurable in OPTIMIZER_CONFIGS
- benchmark optimizers are compared against each proposed optimizer
- seeds = [0,1,2] with mean ± std reporting
- validation and test metrics
- plots: mAP, precision, recall, loss curves, accuracy vs training time
- hyperparameter sensitivity runs and plots
