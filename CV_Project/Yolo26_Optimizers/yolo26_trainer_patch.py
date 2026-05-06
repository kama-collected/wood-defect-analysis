"""
Patch helper for using custom optimizers with Ultralytics YOLO26.

What it does
------------
- imports all custom optimizers from yolo26_custom_optimizers_all_unified.py
- monkey-patches BaseTrainer.build_optimizer so custom optimizers can be selected
  directly from `model.train(optimizer="...")`
- supports optimizer-specific hyperparameter overrides set at the beginning of training
- filters and forwards only constructor-supported keyword arguments for each optimizer

Recommended usage
-----------------
from yolo26_trainer_patch import patch_ultralytics_custom_optimizers

CUSTOM_OPTIMIZER_OVERRIDES = {
    "gradshift_muadam": {
        "betas": (0.9, 0.999),
        "shift_scale": 0.15,
        "muon_ratio": 0.10,
        "ns_steps": 5,
        "weight_decay": 1e-4,
    },
    "orthomuonw": {
        "muon_ratio": 0.20,
        "ns_steps": 5,
        "nesterov": True,
    },
}

patch_ultralytics_custom_optimizers(CUSTOM_OPTIMIZER_OVERRIDES)

from ultralytics import YOLO
model = YOLO("yolo26n.pt")
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    optimizer="gradshift_muadam",
)

Notes
-----
- Custom optimizers are selected directly through `model.train(optimizer="name")`.
- Extra optimizer parameters should be configured at the start of training using
  `patch_ultralytics_custom_optimizers(CUSTOM_OPTIMIZER_OVERRIDES)`.
- This file was syntax-tested here. Full YOLO26 end-to-end training should be
  validated in the target local environment.
"""

from __future__ import annotations

from copy import deepcopy
import inspect
from typing import Any, Dict, Mapping

from yolo26_custom_optimizers_all_unified import OPTIMIZER_REGISTRY

_CUSTOM_OPTIMIZER_OVERRIDES: Dict[str, dict] = {}


def _match_name(name: str) -> str:
    return str(name).lower().replace("-", "").replace("_", "")


def _normalize_overrides(overrides: Mapping[str, dict] | None) -> Dict[str, dict]:
    if not overrides:
        return {}
    return {_match_name(k): deepcopy(v) for k, v in overrides.items()}


def set_custom_optimizer_overrides(overrides: Mapping[str, dict] | None = None) -> None:
    """Replace the global custom optimizer override mapping."""
    global _CUSTOM_OPTIMIZER_OVERRIDES
    _CUSTOM_OPTIMIZER_OVERRIDES = _normalize_overrides(overrides)


def get_custom_optimizer_overrides() -> Dict[str, dict]:
    """Return a deep copy of the active override mapping."""
    return deepcopy(_CUSTOM_OPTIMIZER_OVERRIDES)


def update_custom_optimizer_overrides(overrides: Mapping[str, dict] | None = None) -> None:
    """Update the global override mapping without clearing existing entries."""
    global _CUSTOM_OPTIMIZER_OVERRIDES
    if not overrides:
        return
    normalized = _normalize_overrides(overrides)
    for key, value in normalized.items():
        if key not in _CUSTOM_OPTIMIZER_OVERRIDES:
            _CUSTOM_OPTIMIZER_OVERRIDES[key] = {}
        _CUSTOM_OPTIMIZER_OVERRIDES[key].update(deepcopy(value))


def _extract_trainer_overrides(trainer: Any, requested: str) -> dict:
    """
    Try to read runtime overrides from trainer args if they are available.

    Supported optional attributes on `trainer.args`:
    - custom_optimizer_overrides: {optimizer_name: {...}}
    - optimizer_overrides: {optimizer_name: {...}}
    - custom_optimizer_params: {...}  # direct mapping for the current optimizer

    If the local Ultralytics config rejects unknown train() kwargs, use the
    patch_ultralytics_custom_optimizers({...}) entry point instead.
    """
    runtime: dict = {}
    args = getattr(trainer, "args", None)
    if args is None:
        return runtime

    # named mappings: {"gradshift_muadam": {...}}
    for attr in ("custom_optimizer_overrides", "optimizer_overrides"):
        maybe = getattr(args, attr, None)
        if isinstance(maybe, Mapping):
            maybe_norm = _normalize_overrides(maybe)
            runtime.update(deepcopy(maybe_norm.get(requested, {})))

    # direct current-optimizer mapping: {...}
    maybe_direct = getattr(args, "custom_optimizer_params", None)
    if isinstance(maybe_direct, Mapping):
        runtime.update(deepcopy(maybe_direct))

    return runtime


def _filter_kwargs_for_optimizer(opt_cls: type, raw_kwargs: Mapping[str, Any]) -> dict:
    """Forward only constructor-supported kwargs to the optimizer class."""
    sig = inspect.signature(opt_cls.__init__)
    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kw:
        return dict(raw_kwargs)

    allowed = set(params.keys()) - {"self", "params"}
    filtered = {k: v for k, v in raw_kwargs.items() if k in allowed}
    return filtered


def patch_ultralytics_custom_optimizers(overrides: Mapping[str, dict] | None = None) -> None:
    """
    Patch Ultralytics BaseTrainer so custom optimizers can be selected via
    `model.train(optimizer="...")`.

    Parameters
    ----------
    overrides:
        Optional mapping like:
        {
            "gradshift_muadam": {"shift_scale": 0.15, "muon_ratio": 0.1},
            "orthomuonw": {"muon_ratio": 0.2, "ns_steps": 5},
        }
    """
    try:
        from ultralytics.engine.trainer import BaseTrainer
    except Exception as e:
        raise RuntimeError(
            "Ultralytics is not importable in this environment. Install ultralytics in your local environment first."
        ) from e

    if overrides is not None:
        set_custom_optimizer_overrides(overrides)

    registry = {_match_name(k): v for k, v in OPTIMIZER_REGISTRY.items()}

    # If already patched, refresh registry/overrides and exit.
    if hasattr(BaseTrainer, "_custom_optimizer_patch_applied"):
        BaseTrainer._custom_optimizer_registry = registry
        BaseTrainer._custom_optimizer_overrides = get_custom_optimizer_overrides()
        return

    original_build_optimizer = getattr(BaseTrainer, "build_optimizer", None)
    if original_build_optimizer is None:
        raise RuntimeError(
            "This Ultralytics version does not expose BaseTrainer.build_optimizer. "
            "Patch the local trainer source manually using the logic in this file."
        )

    def patched_build_optimizer(
        self: Any,
        model,
        name: str = "auto",
        lr: float = 0.001,
        momentum: float = 0.9,
        decay: float = 1e-5,
        iterations: float = 1e5,
    ):
        requested = _match_name(name)
        local_registry = getattr(type(self), "_custom_optimizer_registry", registry)
        if requested in local_registry:
            params = [p for p in model.parameters() if p.requires_grad]
            opt_cls = local_registry[requested]

            # Base kwargs from Ultralytics trainer arguments.
            raw_kwargs: dict[str, Any] = {
                "lr": lr,
                "momentum": momentum,
                "weight_decay": decay,
                "iterations": iterations,
            }

            # Global/user-configured overrides from the start of training.
            raw_kwargs.update(deepcopy(getattr(type(self), "_custom_optimizer_overrides", {})).get(requested, {}))

            # Optional runtime overrides if the local trainer args expose them.
            raw_kwargs.update(_extract_trainer_overrides(self, requested))

            # Only pass constructor-supported kwargs.
            kwargs = _filter_kwargs_for_optimizer(opt_cls, raw_kwargs)
            return opt_cls(params, **kwargs)

        return original_build_optimizer(
            self,
            model,
            name=name,
            lr=lr,
            momentum=momentum,
            decay=decay,
            iterations=iterations,
        )

    BaseTrainer.build_optimizer = patched_build_optimizer
    BaseTrainer._custom_optimizer_patch_applied = True
    BaseTrainer._custom_optimizer_registry = registry
    BaseTrainer._custom_optimizer_overrides = get_custom_optimizer_overrides()


def patch_source_snippet() -> str:
    return (
        """
from yolo26_trainer_patch import patch_ultralytics_custom_optimizers

CUSTOM_OPTIMIZER_OVERRIDES = {
    "orthomuonw": {
        "muon_ratio": 0.20,
        "ns_steps": 5,
        "nesterov": True,
    },
    "gradshift_muadam": {
        "betas": (0.9, 0.999),
        "shift_scale": 0.15,
        "muon_ratio": 0.10,
        "ns_steps": 5,
        "weight_decay": 1e-4,
    },
    "texadam_trust": {
        "betas": (0.9, 0.999),
        "trust_clip": (0.1, 10.0),
        "texture_alpha": 0.25,
    },
    "medstable_adam": {
        "betas": (0.9, 0.999),
        "stability_beta": 0.95,
        "agc_clip": 0.01,
    },
    "lossswitch_muadam": {
        "betas": (0.9, 0.999),
        "switch_threshold": 0.01,
        "muon_ratio": 0.10,
    },
    "layertrust_scaleadam": {
        "betas": (0.9, 0.999),
        "trust_clip": (0.1, 10.0),
        "gamma": 0.5,
    },
    "consensusdrift_adam": {
        "betas": (0.9, 0.999),
        "drift_beta": 0.95,
        "consensus_beta": 0.90,
    },
}

patch_ultralytics_custom_optimizers(CUSTOM_OPTIMIZER_OVERRIDES)

from ultralytics import YOLO
model = YOLO("yolo26n.pt")
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    optimizer="gradshift_muadam",
)
        """.strip()
    )


if __name__ == "__main__":
    # Syntax/import sanity check only. Full YOLO validation should be performed locally.
    print("Available custom optimizers:")
    for key in sorted(OPTIMIZER_REGISTRY):
        print(f" - {key}")
