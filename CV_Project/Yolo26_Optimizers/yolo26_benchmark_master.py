
from benchmark_common import run_benchmark_package

MODEL_NAME = "yolo26n.pt"
DATASET_PATH = r"../wood-2/data.yaml"
PROJECT_DIR = r"runs/final_optimizer_benchmark_master"

CONFIG = {
    "epochs": 30,
    "imgsz": 416,
    "batch": 32,
    "device": 0,
    "workers": 0,
    "patience": 20,
    "seed": [0,1,2],
    "deterministic": True,
    "verbose": True,
    "exist_ok": True,
}

BENCHMARK_OPTIMIZERS_TO_RUN = ["Adam", "Adamax", "AdamW", "NAdam"]

CUSTOM_OPTIMIZERS_TO_RUN = ["consensusdrift_adam"]

OPTIMIZER_CONFIGS = {

    "Adam": {"optimizer": "Adam", "lr0": 0.001, "lrf": 0.01, "momentum": 0.9, "weight_decay": 5e-4},
    "Adamax": {"optimizer": "Adamax", "lr0": 0.002, "lrf": 0.01, "momentum": 0.9, "weight_decay": 5e-4},
    "AdamW": {"optimizer": "AdamW", "lr0": 0.001, "lrf": 0.01, "momentum": 0.9, "weight_decay": 1e-3},
    "NAdam": {"optimizer": "NAdam", "lr0": 0.001, "lrf": 0.01, "momentum": 0.9, "weight_decay": 5e-4},
    "texadam_trust": {"optimizer": "texadam_trust", "lr0": 0.001, "lrf": 0.01, "momentum": 0.9, "weight_decay": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8, "texture_gamma": 0.5, "trust_clip": (0.1, 10.0), "agc_clip": 0.01},
    "consensusdrift_adam": {"optimizer": "consensusdrift_adam", "lr0": 0.001, "lrf": 0.01, "momentum": 0.9, "weight_decay": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8, "rho": 0.05, "consensus_beta": 0.9, "drift_eps": 1e-8},
}

SENSITIVITY_PLAN = {
    "gradshift_muadam": {"shift_scale": [0.10, 0.15, 0.20]},
    "texadam_trust": {"texture_gamma": [0.25, 0.50, 0.75]},
    "consensusdrift_adam": {"rho": [0.01, 0.05, 0.10]},
}

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
