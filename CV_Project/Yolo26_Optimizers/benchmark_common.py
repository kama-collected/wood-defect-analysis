
from __future__ import annotations

import gc
import json
import math
import os
import random
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import ultralytics
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from yolo26_trainer_patch import patch_ultralytics_custom_optimizers


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def infer_task_from_model_name(model_name: str) -> str:
    n = model_name.lower()
    return "seg" if "-seg" in n or "seg" in n else "detect"


def flatten_metrics(prefix: str, obj: Any, task: str = "detect") -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    if obj is None:
        return out
    if task == "seg":
        candidates = {
            f"{prefix}_precision": ["seg.p", "metrics/precision(M)", "precision", "P", "p"],
            f"{prefix}_recall": ["seg.r", "metrics/recall(M)", "recall", "R", "r"],
            f"{prefix}_map50": ["seg.map50", "metrics/mAP50(M)", "map50", "mAP50"],
            f"{prefix}_map50_95": ["seg.map", "metrics/mAP50-95(M)", "map", "mAP50-95"],
            f"{prefix}_mask_map": ["seg.map", "metrics/mAP50-95(M)", "map", "mAP50-95"],
            f"{prefix}_iou": ["iou", "metrics/IoU(M)", "IoU"],
            f"{prefix}_fps": ["fps", "speed/fps", "FPS"],
        }
    else:
        candidates = {
            f"{prefix}_precision": ["box.p", "metrics/precision(B)", "precision", "P", "p"],
            f"{prefix}_recall": ["box.r", "metrics/recall(B)", "recall", "R", "r"],
            f"{prefix}_map50": ["box.map50", "metrics/mAP50(B)", "map50", "mAP50"],
            f"{prefix}_map50_95": ["box.map", "metrics/mAP50-95(B)", "map", "mAP50-95"],
            f"{prefix}_fps": ["fps", "speed/fps", "FPS"],
        }
    for out_key, keys in candidates.items():
        value = None
        if isinstance(obj, dict):
            for k in keys:
                if k in obj:
                    value = safe_float(obj.get(k))
                    break
        out[out_key] = value
    for attr_name in ("box", "seg"):
        source = getattr(obj, attr_name, None)
        if source is not None:
            if out.get(f"{prefix}_precision") is None and hasattr(source, "p"):
                out[f"{prefix}_precision"] = safe_float(source.p)
            if out.get(f"{prefix}_recall") is None and hasattr(source, "r"):
                out[f"{prefix}_recall"] = safe_float(source.r)
            if out.get(f"{prefix}_map50") is None and hasattr(source, "map50"):
                out[f"{prefix}_map50"] = safe_float(source.map50)
            if out.get(f"{prefix}_map50_95") is None and hasattr(source, "map"):
                out[f"{prefix}_map50_95"] = safe_float(source.map)
            if task == "seg" and out.get(f"{prefix}_mask_map") is None and hasattr(source, "map"):
                out[f"{prefix}_mask_map"] = safe_float(source.map)
    speed = getattr(obj, "speed", None)
    if isinstance(speed, dict):
        total_ms = sum(v for v in speed.values() if isinstance(v, (int, float)))
        if total_ms > 0:
            out[f"{prefix}_fps"] = 1000.0 / total_ms
            out[f"{prefix}_time_s"] = total_ms / 1000.0
    return out


def find_results_csv(run_dir: str | Path) -> Optional[Path]:
    run_dir = Path(run_dir)
    candidate = run_dir / "results.csv"
    if candidate.exists():
        return candidate
    csvs = list(run_dir.glob("**/results.csv"))
    return csvs[0] if csvs else None


def extract_best_epoch_row(results_csv: str | Path, task: str = "detect") -> Dict[str, Any]:
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    best_cols = ["metrics/mAP50-95(B)", "metrics/mAP50(B)", "metrics/precision(B)", "fitness"] if task == "detect" else ["metrics/mAP50-95(M)", "metrics/mAP50(M)", "metrics/precision(M)", "fitness"]
    best_col = next((c for c in best_cols if c in df.columns), None)
    row = df.loc[df[best_col].idxmax()].to_dict() if best_col else df.iloc[-1].to_dict()
    row["results_csv"] = str(results_csv)
    return row


def normalize_epoch_row(row: Dict[str, Any], task: str = "detect") -> Dict[str, Optional[float]]:
    out = {
        "best_epoch": safe_float(row.get("epoch")),
        "train_box_loss": safe_float(row.get("train/box_loss")),
        "train_cls_loss": safe_float(row.get("train/cls_loss")),
        "train_dfl_loss": safe_float(row.get("train/dfl_loss")),
        "val_box_loss": safe_float(row.get("val/box_loss")),
        "val_cls_loss": safe_float(row.get("val/cls_loss")),
        "val_dfl_loss": safe_float(row.get("val/dfl_loss")),
        "time_s": safe_float(row.get("time")),
    }
    if task == "seg":
        out.update({
            "val_precision": safe_float(row.get("metrics/precision(M)")),
            "val_recall": safe_float(row.get("metrics/recall(M)")),
            "val_map50": safe_float(row.get("metrics/mAP50(M)")),
            "val_map50_95": safe_float(row.get("metrics/mAP50-95(M)")),
            "val_mask_map": safe_float(row.get("metrics/mAP50-95(M)")),
            "val_iou": safe_float(row.get("metrics/IoU(M)")),
        })
    else:
        out.update({
            "val_precision": safe_float(row.get("metrics/precision(B)")),
            "val_recall": safe_float(row.get("metrics/recall(B)")),
            "val_map50": safe_float(row.get("metrics/mAP50(B)")),
            "val_map50_95": safe_float(row.get("metrics/mAP50-95(B)")),
        })
    return out


def format_mean_std(mean: Optional[float], std: Optional[float], digits: int = 4) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return ""
    if std is None or (isinstance(std, float) and math.isnan(std)):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    agg = df.groupby("optimizer_name", as_index=False)[numeric_cols].agg(["mean", "std", "max"])
    agg.columns = ["_".join([x for x in col if x]).strip("_") for col in agg.columns.to_flat_index()]
    return agg


def make_summary_table(df: pd.DataFrame, task: str = "detect") -> pd.DataFrame:
    rows = []
    for opt, g in df.groupby("optimizer_name"):
        row = {"optimizer": opt, "runs": len(g)}
        metrics = ["val_eval_map50_95", "val_eval_map50", "val_eval_precision", "val_eval_recall",
                   "test_map50_95", "test_map50", "test_precision", "test_recall",
                   "val_eval_fps", "test_fps", "elapsed_train_hours", "epoch_time_seconds_mean"]
        if task == "seg":
            metrics += ["val_eval_mask_map", "test_mask_map", "val_eval_iou", "test_iou"]
        for m in metrics:
            if m in g.columns:
                row[m] = g[m].mean()
                row[f"{m}_std"] = g[m].std()
                row[f"{m}_mean±std"] = format_mean_std(g[m].mean(), g[m].std())
        rows.append(row)
    out = pd.DataFrame(rows)
    sort_col = "test_map50_95" if "test_map50_95" in out.columns else "val_eval_map50_95"
    if sort_col in out.columns:
        out = out.sort_values(sort_col, ascending=False)
    return out


def plot_bar(summary_df: pd.DataFrame, metric: str, out_path: str | Path, title: str) -> None:
    if metric not in summary_df.columns:
        return
    plt.figure(figsize=(12, 6))
    plt.bar(summary_df["optimizer"].astype(str), summary_df[metric])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_learning_curves(all_runs_df: pd.DataFrame, plot_dir: str | Path, task: str = "detect") -> None:
    plot_dir = Path(plot_dir)
    metric_candidates = ["metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"] if task == "detect" else ["metrics/mAP50(M)", "metrics/mAP50-95(M)", "metrics/precision(M)", "metrics/recall(M)"]
    loss_candidates = ["train/box_loss", "val/box_loss", "train/cls_loss", "val/cls_loss"]
    for _, row in all_runs_df.iterrows():
        results_csv = find_results_csv(row.get("run_dir", ""))
        if not results_csv:
            continue
        try:
            df = pd.read_csv(results_csv)
            df.columns = [c.strip() for c in df.columns]
            metrics = [m for m in metric_candidates + loss_candidates if m in df.columns]
            for metric in metrics:
                plt.figure(figsize=(8, 5))
                plt.plot(df["epoch"], df[metric])
                plt.xlabel("Epoch")
                plt.ylabel(metric)
                plt.title(f"{row['optimizer_name']} - {metric}")
                plt.tight_layout()
                safe_name = metric.replace("/", "_").replace("(", "").replace(")", "")
                plt.savefig(plot_dir / f"curve_{row['run_name']}_{safe_name}.png", dpi=300)
                plt.close()
        except Exception:
            continue


def plot_accuracy_vs_time(summary_df: pd.DataFrame, out_path: str | Path, map_col: str = "test_map50_95", time_col: str = "elapsed_train_hours") -> None:
    if map_col not in summary_df.columns or time_col not in summary_df.columns:
        return
    plt.figure(figsize=(8, 6))
    plt.scatter(summary_df[time_col], summary_df[map_col])
    for _, r in summary_df.iterrows():
        plt.annotate(str(r["optimizer"]), (r[time_col], r[map_col]))
    plt.xlabel("Training Time")
    plt.ylabel("mAP50-95")
    plt.title("Accuracy vs Training Time")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_sensitivity(df: pd.DataFrame, out_dir: str | Path, sensitivity_param: str) -> None:
    out_dir = Path(out_dir)
    metric = "test_map50_95" if "test_map50_95" in df.columns else "val_eval_map50_95"
    if metric not in df.columns or "sensitivity_value" not in df.columns:
        return
    for opt, g in df.groupby("optimizer_name"):
        gg = g.dropna(subset=[metric, "sensitivity_value"]).sort_values("sensitivity_value")
        if gg.empty:
            continue
        plt.figure(figsize=(7, 5))
        plt.plot(gg["sensitivity_value"], gg[metric], marker="o")
        plt.xlabel(sensitivity_param)
        plt.ylabel(metric)
        plt.title(f"Sensitivity: {opt}")
        plt.tight_layout()
        plt.savefig(out_dir / f"sensitivity_{opt}_{sensitivity_param}.png", dpi=300)
        plt.close()


def save_observations(out_path: str | Path) -> None:
    lines = [
        "Observation template:",
        "- Proposed optimizers converge faster in early epochs.",
        "- Adaptive mechanisms stabilize late-stage training.",
        "- Reduced oscillations compared to Adam and RMSProp.",
        "- Some optimizers provide better accuracy at moderate cost increase.",
        "- Others maintain competitive performance with lower training time."
    ]
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def get_patch_overrides(optimizer_configs: Dict[str, Dict[str, Any]], optimizer_names: List[str]) -> Dict[str, Dict[str, Any]]:
    common_exclude = {"optimizer", "lr0", "lrf", "momentum", "weight_decay"}
    overrides = {}
    for name in optimizer_names:
        cfg = optimizer_configs.get(name, {})
        overrides[name] = {k: v for k, v in cfg.items() if k not in common_exclude}
    return overrides


def train_one(model_name: str, dataset_path: str, project_dir: str | Path, config: Dict[str, Any], optimizer_name: str, optimizer_cfg: Dict[str, Any], task: str = "detect", seed: int = 42, run_suffix: Optional[str] = None, do_val: bool = True, do_test: bool = True) -> Dict[str, Any]:
    set_seed(seed)
    run_name = f"{Path(model_name).stem}_{optimizer_name}_seed{seed}"
    if run_suffix:
        run_name += f"_{run_suffix}"
    run_dir = ensure_dir(Path(project_dir) / run_name)
    model = YOLO(model_name)
    t0 = time.time()
    model.train(
        data=dataset_path,
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        device=config["device"],
        workers=config["workers"],
        patience=config["patience"],
        seed=seed,
        deterministic=config["deterministic"],
        verbose=config["verbose"],
        exist_ok=config["exist_ok"],
        project=str(project_dir),
        name=run_name,
        pretrained=True,
        val=True,
        plots=True,
        save=True,
        save_period=-1,
        **optimizer_cfg,
    )
    elapsed_train_hours = (time.time() - t0) / 3600.0
    results_csv = find_results_csv(run_dir)
    epoch_info = {}
    epoch_time_seconds_mean = None
    if results_csv:
        best_row = extract_best_epoch_row(results_csv, task=task)
        epoch_info = normalize_epoch_row(best_row, task=task)
        try:
            rdf = pd.read_csv(results_csv)
            rdf.columns = [c.strip() for c in rdf.columns]
            if "time" in rdf.columns and len(rdf) > 1:
                diffs = pd.to_numeric(rdf["time"], errors="coerce").diff().dropna()
                if len(diffs):
                    epoch_time_seconds_mean = float(diffs.mean())
        except Exception:
            pass
    val_metrics, test_metrics = {}, {}
    weights_path = run_dir / "weights" / "best.pt"
    eval_model = YOLO(str(weights_path) if weights_path.exists() else model_name)
    if do_val:
        try:
            val_res = eval_model.val(data=dataset_path, imgsz=config["imgsz"], batch=config["batch"], device=config["device"], split="val")
            val_metrics = flatten_metrics("val_eval", val_res, task=task)
        except Exception:
            val_metrics = {"val_eval_error": 1}
    if do_test:
        try:
            test_res = eval_model.val(data=dataset_path, imgsz=config["imgsz"], batch=config["batch"], device=config["device"], split="test")
            test_metrics = flatten_metrics("test", test_res, task=task)
        except Exception:
            test_metrics = {"test_error": 1}
    return {"model": model_name, "dataset": dataset_path, "optimizer_name": optimizer_name, "seed": seed, "run_name": run_name, "run_dir": str(run_dir), "elapsed_train_hours": elapsed_train_hours, "epoch_time_seconds_mean": epoch_time_seconds_mean, **optimizer_cfg, **epoch_info, **val_metrics, **test_metrics}


def run_sensitivity(model_name: str, dataset_path: str, project_dir: str | Path, config: Dict[str, Any], optimizer_configs: Dict[str, Dict[str, Any]], sensitivity_plan: Dict[str, Dict[str, List[Any]]], task: str = "detect") -> pd.DataFrame:
    records = []
    base_seed = config["seed"][0] if isinstance(config["seed"], list) else config["seed"]
    for opt, plan in sensitivity_plan.items():
        if opt not in optimizer_configs:
            continue
        base_cfg = optimizer_configs[opt]
        for param_name, values in plan.items():
            for value in values:
                override = base_cfg.copy()
                override[param_name] = value
                try:
                    rec = train_one(model_name, dataset_path, project_dir, config, opt, override, task=task, seed=base_seed, run_suffix=f"sens_{param_name}_{str(value).replace('.', 'p')}")
                    rec["sensitivity_param"] = param_name
                    rec["sensitivity_value"] = value
                    records.append(rec)
                except Exception as e:
                    records.append({"optimizer_name": opt, "sensitivity_param": param_name, "sensitivity_value": value, "error": str(e)})
    return pd.DataFrame(records)


def save_json(path: str | Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def run_benchmark_package(model_name: str, dataset_path: str, project_dir: str, config: Dict[str, Any], benchmark_optimizers_to_run: List[str], custom_optimizers_to_run: List[str], optimizer_configs: Dict[str, Dict[str, Any]], task: Optional[str] = None, run_sensitivity_flag: bool = False, sensitivity_plan: Optional[Dict[str, Dict[str, List[Any]]]] = None) -> None:
    task = task or infer_task_from_model_name(model_name)
    ultralytics.checks()
    ensure_dir(project_dir)
    plot_dir = ensure_dir(Path(project_dir) / "plots")
    all_optimizers = benchmark_optimizers_to_run + custom_optimizers_to_run
    patch_ultralytics_custom_optimizers(get_patch_overrides(optimizer_configs, all_optimizers))
    seeds = config["seed"] if isinstance(config["seed"], list) else [config["seed"]]
    records = []
    for opt in all_optimizers:
        if opt not in optimizer_configs:
            records.append({"optimizer_name": opt, "error": "Missing config"})
            continue
        for seed in seeds:
            try:
                rec = train_one(model_name, dataset_path, project_dir, config, opt, optimizer_configs[opt], task=task, seed=seed, do_val=True, do_test=True)
                records.append(rec)
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                records.append({"model": model_name, "dataset": dataset_path, "optimizer_name": opt, "seed": seed, "error": str(e), "traceback": traceback.format_exc()})
    all_runs = pd.DataFrame(records)
    all_runs.to_csv(Path(project_dir) / "all_runs.csv", index=False)
    clean_runs = all_runs[~all_runs.get("error", pd.Series([None] * len(all_runs))).notna()].copy()
    if clean_runs.empty:
        return
    agg = aggregate_results(clean_runs)
    agg.to_csv(Path(project_dir) / "aggregated_runs.csv", index=False)
    summary = make_summary_table(clean_runs, task=task)
    summary.to_csv(Path(project_dir) / "summary_table.csv", index=False)
    summary.to_markdown(Path(project_dir) / "summary_table.md", index=False)
    for metric, fname, title in [("val_eval_map50_95", "bar_val_map50_95.png", "Validation mAP50-95 by Optimizer"), ("val_eval_map50", "bar_val_map50.png", "Validation mAP50 by Optimizer"), ("val_eval_precision", "bar_val_precision.png", "Validation Precision by Optimizer"), ("val_eval_recall", "bar_val_recall.png", "Validation Recall by Optimizer"), ("test_map50_95", "bar_test_map50_95.png", "Test mAP50-95 by Optimizer"), ("test_map50", "bar_test_map50.png", "Test mAP50 by Optimizer"), ("test_precision", "bar_test_precision.png", "Test Precision by Optimizer"), ("test_recall", "bar_test_recall.png", "Test Recall by Optimizer"), ("val_eval_fps", "bar_val_fps.png", "Validation FPS by Optimizer"), ("test_fps", "bar_test_fps.png", "Test FPS by Optimizer"), ("elapsed_train_hours", "bar_train_time_hours.png", "Training Time (hours) by Optimizer"), ("epoch_time_seconds_mean", "bar_epoch_time_seconds.png", "Training Time per Epoch (seconds) by Optimizer")]:
        plot_bar(summary, metric, Path(plot_dir) / fname, title)
    plot_learning_curves(clean_runs, plot_dir, task=task)
    plot_accuracy_vs_time(summary, Path(plot_dir) / "accuracy_vs_training_time.png")
    save_observations(Path(project_dir) / "observation_template.txt")
    if run_sensitivity_flag and sensitivity_plan:
        sens_df = run_sensitivity(model_name, dataset_path, project_dir, config, optimizer_configs, sensitivity_plan, task=task)
        sens_df.to_csv(Path(project_dir) / "hyperparameter_sensitivity.csv", index=False)
        for param_name in {k for plan in sensitivity_plan.values() for k in plan.keys()}:
            plot_sensitivity(sens_df[sens_df.get("sensitivity_param") == param_name].copy(), plot_dir, param_name)
    save_json(Path(project_dir) / "experiment_config.json", {"model_name": model_name, "dataset_path": dataset_path, "project_dir": project_dir, "config": config, "benchmark_optimizers_to_run": benchmark_optimizers_to_run, "custom_optimizers_to_run": custom_optimizers_to_run, "optimizer_configs": optimizer_configs, "task": task, "metrics": ["mAP50", "mAP50-95", "Precision", "Recall", "Mask mAP", "IoU", "Training time per epoch", "Total training time", "Inference speed (FPS)", "Time (s)"], "reporting": "mean ± standard deviation across seeds"})
