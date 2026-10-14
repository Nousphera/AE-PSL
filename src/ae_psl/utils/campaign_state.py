"""Evidence required to skip a finished job, never an optimizer-state resume."""

import json
import math
from pathlib import Path


def completed_result(path, args):
    try:
        result = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(result, dict):
        return None
    completion = result.get("completion")
    if completion is not None and not isinstance(completion, dict):
        return None
    if completion and args.get("ae_pretrain_only"):
        return (
            completion
            if (
                completion.get("state") == "completed"
                and completion.get("stage") == "ga-only"
                and completion.get("requested_epochs") == args["ae_pretrain_epochs"]
                and isinstance(result.get("ae_pretrain_results"), dict)
            )
            else None
        )
    metrics = result.get("finetune_results", {})
    requested = args["nr_of_epochs"]
    if requested < 1 or not isinstance(metrics, dict):
        return None
    for key in ("final_test_loss", "final_test_accuracy"):
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            return None
    if metrics["final_test_accuracy"] > 1:
        return None
    if completion:
        last = completion.get("last_completed_epoch", -1)
        valid_epoch = isinstance(last, int) and 0 <= last < requested
        if (
            completion.get("state") == "completed"
            and completion.get("stage") == "dft"
            and completion.get("requested_epochs") == requested
            and valid_epoch
            and metrics.get("last_completed_epoch") == last
            and (last == requested - 1 or completion.get("stopped_early") is True)
        ):
            return completion
        return None
    # Historical combined results were written only after final evaluation.
    # Require that structure, valid final metrics and the full requested history;
    # an epoch-level JSON or an ambiguous early-stopped legacy run is insufficient.
    epochs = metrics.get("epochs", [])
    history = metrics.get("train_metric", [])
    if (
        isinstance(result.get("ae_pretrain_results"), dict)
        and isinstance(epochs, list)
        and isinstance(history, list)
        and all(epoch in epochs for epoch in range(requested))
        and len(history) >= requested
    ):
        return {
            "state": "completed",
            "stage": "dft",
            "requested_epochs": requested,
            "last_completed_epoch": requested - 1,
            "legacy_completion_evidence": True,
        }
    return None


def last_recorded_epoch(path):
    try:
        result = json.loads(Path(path).read_text())
        return result.get("finetune_results", {}).get("last_completed_epoch", -1)
    except (OSError, ValueError, AttributeError):
        return -1
