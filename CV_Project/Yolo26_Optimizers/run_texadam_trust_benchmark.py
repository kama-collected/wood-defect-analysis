
from yolo26_benchmark_master import MODEL_NAME, DATASET_PATH, CONFIG, BENCHMARK_OPTIMIZERS_TO_RUN, OPTIMIZER_CONFIGS
from benchmark_common import run_benchmark_package

CUSTOM_OPTIMIZERS_TO_RUN = ["texadam_trust"]
PROJECT_DIR = r"runs/texadam_trust_vs_benchmarks"
SENSITIVITY_PLAN = {"texadam_trust": {"texture_gamma": [0.25, 0.50, 0.75]}}

if __name__ == "__main__":
    run_benchmark_package(
        model_name=MODEL_NAME,
        dataset_path=DATASET_PATH,
        project_dir=PROJECT_DIR,
        config=CONFIG,
        benchmark_optimizers_to_run=BENCHMARK_OPTIMIZERS_TO_RUN,
        custom_optimizers_to_run=CUSTOM_OPTIMIZERS_TO_RUN,
        optimizer_configs=OPTIMIZER_CONFIGS,
        run_sensitivity_flag=True,
        sensitivity_plan=SENSITIVITY_PLAN,
    )
