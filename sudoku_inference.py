"""
Step-by-step Sudoku inference for the HRM model.

This script:
 - Loads a trained checkpoint (local directory or Hugging Face repo)
 - Loads one random Sudoku from the test set
 - Runs the model step-by-step (ACT) and prints internal states and predictions

You can copy cells into a Jupyter notebook and execute incrementally.
"""

import os
import json
import random
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch

try:
    import yaml
except Exception:
    yaml = None  # Only needed when loading local checkpoints with YAML config

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None

from utils.functions import load_model_class


# ---------------------------
# Pretty printing utilities
# ---------------------------
def pretty_print_sudoku(grid: np.ndarray):
    if isinstance(grid, torch.Tensor):
        grid = grid.detach().cpu().numpy()
    grid = grid.astype(int)
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - -")
        row = []
        for j in range(9):
            if j % 3 == 0 and j != 0:
                row.append("|")
            row.append(str(grid[i, j]-1))
        print(" ".join(row))


# ---------------------------
# Dataset loading
# ---------------------------
def load_sudoku_metadata(data_path: str) -> Dict[str, Any]:
    test_dir = os.path.join(data_path, "test")
    with open(os.path.join(test_dir, "dataset.json"), "r") as f:
        return json.load(f)


def load_random_sudoku_example(data_path: str, puzzle_index: Optional[int] = None) -> Tuple[Dict[str, torch.Tensor], int]:
    """Loads a single example from the Sudoku test set as a batch of size 1."""
    test_dir = os.path.join(data_path, "test")
    with open(os.path.join(test_dir, "dataset.json"), "r") as f:
        meta = json.load(f)

    # Sudoku builder uses a single set named "all"
    assert len(meta["sets"]) == 1, "Sudoku test set is expected to have a single subset"
    set_name = meta["sets"][0]

    inputs = np.load(os.path.join(test_dir, f"{set_name}__inputs.npy"))
    labels = np.load(os.path.join(test_dir, f"{set_name}__labels.npy"))
    # For Sudoku, puzzle_identifiers is all zeros
    puzzle_identifiers = np.zeros((inputs.shape[0],), dtype=np.int32)

    assert inputs.shape[1] == 81 and labels.shape[1] == 81, "Sudoku expects 9x9 (seq_len=81)"

    n = inputs.shape[0]
    if puzzle_index is None:
        puzzle_index = random.randrange(n)

    batch = {
        "inputs": torch.from_numpy(inputs[puzzle_index:puzzle_index + 1].astype(np.int32)),
        "labels": torch.from_numpy(labels[puzzle_index:puzzle_index + 1].astype(np.int32)),
        "puzzle_identifiers": torch.from_numpy(puzzle_identifiers[puzzle_index:puzzle_index + 1]),
    }
    return batch, puzzle_index


# ---------------------------
# Checkpoint/model loading
# ---------------------------
def load_model_from_checkpoint(
    *,
    device: torch.device,
    data_meta: Dict[str, Any],
    checkpoint_dir: Optional[str] = None,
    ckpt_path: Optional[str] = None,
    hf_repo: str = "sapientinc/HRM-checkpoint-sudoku-extreme",
    hf_filename: str = "model.pt",
):
    """Builds the model and loads weights from either a local training directory or HF."""
    model = None

    def _normalize_state_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Handle torch.compile wrapper and loss-head wrapper prefixes
        def strip_prefix(k: str, p: str) -> str:
            return k[len(p):] if k.startswith(p) else k

        out = {}
        for k, v in state.items():
            k2 = strip_prefix(k, "_orig_mod.")
            k2 = strip_prefix(k2, "model.")
            out[k2] = v
        return out

    # Case A: explicit checkpoint file path
    if ckpt_path is not None:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        obj = torch.load(ckpt_path, map_location=device)

        # Packaged format: contains cfg + model_state_dict
        if isinstance(obj, dict) and ("cfg" in obj) and ("model_state_dict" in obj):
            cfg = obj["cfg"]
            if hasattr(cfg, "to_container"):
                arch_cfg = dict(cfg.arch)  # type: ignore
            else:
                arch_cfg = dict(cfg["arch"])  # type: ignore

            model_cls = load_model_class(arch_cfg["name"])  # type: ignore
            model = model_cls(arch_cfg).to(device)
            # Be lenient about prefixes in packaged files
            state = obj["model_state_dict"]  # type: ignore
            if isinstance(state, dict):
                state = _normalize_state_keys(state)
            model.load_state_dict(state)  # type: ignore
            model.eval()
            return model, arch_cfg

        # Raw state_dict format: need YAML config nearby (in same dir or checkpoint_dir)
        cfg_dir = os.path.dirname(ckpt_path)
        cfg_path = os.path.join(cfg_dir, "all_config.yaml")
        if (checkpoint_dir is not None) and (not os.path.exists(cfg_path)):
            cfg_path = os.path.join(checkpoint_dir, "all_config.yaml")
        if yaml is None:
            raise RuntimeError("PyYAML is required to load local checkpoints. Please install pyyaml.")
        assert os.path.exists(cfg_path), f"Missing config: {cfg_path}"

        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        arch_cfg = cfg["arch"]

        model_cfg = dict(
            **{k: v for k, v in arch_cfg.items() if k != "name"},
            batch_size=1,
            seq_len=data_meta["seq_len"],
            vocab_size=data_meta["vocab_size"],
            num_puzzle_identifiers=data_meta["num_puzzle_identifiers"],
        )
        model_cls = load_model_class(arch_cfg["name"])  # e.g., "hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1"
        model = model_cls(model_cfg).to(device)

        state = obj if isinstance(obj, dict) else obj.state_dict()
        state = _normalize_state_keys(state)
        model.load_state_dict(state, assign=True)
        model.eval()
        return model, arch_cfg

    if checkpoint_dir is not None:
        # Expect a training directory with all_config.yaml and a step_* file
        if yaml is None:
            raise RuntimeError("PyYAML is required to load local checkpoints. Please install pyyaml.")

        cfg_path = os.path.join(checkpoint_dir, "all_config.yaml")
        assert os.path.exists(cfg_path), f"Missing config: {cfg_path}"

        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        arch_cfg = cfg["arch"]

        # Build model config using dataset-derived fields
        model_cfg = dict(
            **{k: v for k, v in arch_cfg.items() if k != "name"},
            batch_size=1,
            seq_len=data_meta["seq_len"],
            vocab_size=data_meta["vocab_size"],
            num_puzzle_identifiers=data_meta["num_puzzle_identifiers"],
        )

        model_cls = load_model_class(arch_cfg["name"])  # e.g., "hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1"
        model = model_cls(model_cfg).to(device)

        # Pick the latest step_* file if not specified
        step_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("step_") and "." not in f]
        assert len(step_files), f"No step_* files found in {checkpoint_dir}"
        step_files.sort(key=lambda s: int(s.split("_")[1]))
        ckpt_path = os.path.join(checkpoint_dir, step_files[-1])

        state = torch.load(ckpt_path, map_location=device)
        state = _normalize_state_keys(state)
        model.load_state_dict(state, assign=True)

        model.eval()
        return model, arch_cfg

    # Otherwise load from Hugging Face packaged checkpoint (contains cfg + model_state_dict)
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub is required for HF checkpoints. Please install huggingface_hub.")

    artifact_path = hf_hub_download(repo_id=hf_repo, filename=hf_filename)
    blob = torch.load(artifact_path, map_location=device)

    # Expect keys: 'cfg' (OmegaConf/dict-like) and 'model_state_dict'
    cfg = blob.get("cfg")
    if hasattr(cfg, "to_container"):
        arch_cfg = cfg.arch  # OmegaConf-like
        arch_cfg = dict(arch_cfg)
    else:
        # Assume dict nesting
        arch_cfg = dict(cfg["arch"])  # type: ignore

    model_cls = load_model_class(arch_cfg["name"])  # type: ignore
    model = model_cls(arch_cfg).to(device)
    state = blob["model_state_dict"]  # type: ignore
    if isinstance(state, dict):
        # Normalize any prefixes just in case
        state = { (k[len("model."):] if isinstance(k, str) and k.startswith("model.") else k): v for k, v in state.items() }
    model.load_state_dict(state)  # type: ignore
    model.eval()
    return model, arch_cfg


# ---------------------------
# Step-by-step inference
# ---------------------------
def run_step_by_step_inference(
    data_path: str = "data/sudoku-extreme-1k-aug-1000",
    checkpoint_dir: Optional[str] = None,
    ckpt_path: Optional[str] = None,
    hf_repo: str = "sapientinc/HRM-checkpoint-sudoku-extreme",
    hf_filename: str = "model.pt",
    puzzle_index: Optional[int] = None,
    seed: int = 0,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if not os.path.exists(os.path.join(data_path, "test", "dataset.json")):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Build it via dataset/build_sudoku_dataset.py or adjust data_path."
        )

    meta = load_sudoku_metadata(data_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, arch_cfg = load_model_from_checkpoint(
        device=device,
        data_meta=meta,
        checkpoint_dir=checkpoint_dir,
        ckpt_path=ckpt_path,
        hf_repo=hf_repo,
        hf_filename=hf_filename,
    )

    print("Loaded model on", device)
    print(model)
    print("halt_max_steps:", arch_cfg.get("halt_max_steps"))
    model.eval()

    batch, chosen_index = load_random_sudoku_example(data_path, puzzle_index=puzzle_index)
    # Move tensors to device
    batch = {k: v.to(device) for k, v in batch.items()}

    print(f"Selected puzzle index: {chosen_index}")
    print("Input puzzle (0 means blank):")
    # Tokens are 1..10 (PAD=0). Subtract 1 for human-readable 0..9.
    pretty_print_sudoku((batch["inputs"].view(9, 9)))

    # Step-by-step execution
    # Ensure carry tensors are allocated on the same device as the model
    with torch.device("cuda" if device.type == "cuda" else "cpu"):
        carry = model.initial_carry(batch)
    final_outputs = None

    with torch.inference_mode():
        for step in range(int(arch_cfg.get("halt_max_steps", 1))):
            print(f"\n=== Reasoning Step {step + 1} ===")
            carry, outputs = model(carry, batch)

            z_H = carry.inner_carry.z_H
            z_L = carry.inner_carry.z_L
            print("z_H shape:", tuple(z_H.shape), "mean:", float(z_H.mean()))
            print("z_L shape:", tuple(z_L.shape), "mean:", float(z_L.mean()))

            # Q logits
            qh = outputs["q_halt_logits"].detach().float().cpu().item()
            qc = outputs["q_continue_logits"].detach().float().cpu().item()
            print(f"q_halt_logits={qh:.3f}, q_continue_logits={qc:.3f}")

            # Current prediction
            logits = outputs["logits"]  # [1, 81, vocab]
            pred_tokens = torch.argmax(logits, dim=-1).view(9, 9)
            print("Current prediction (0..9):")
            pretty_print_sudoku((pred_tokens))

            final_outputs = outputs

            if carry.halted.item():
                print("Halted.")
                break
            else:
                print("Continuing...")

    # Final results
    assert final_outputs is not None
    final_pred = torch.argmax(final_outputs["logits"], dim=-1).view(9, 9)
    gt = batch["labels"].view(9, 9)

    print("\n=== Final Prediction ===")
    pretty_print_sudoku((final_pred))
    print("\nGround truth:")
    pretty_print_sudoku((gt))

    is_correct = bool(torch.all(final_pred == gt).item())
    print("\nExact match:", is_correct)


if __name__ == "__main__":
    # Example usage (edit paths as needed):
    # - If you trained locally: set checkpoint_dir to your checkpoints folder
    # - If you want to use the public HF checkpoint: leave checkpoint_dir=None
    run_step_by_step_inference(
        data_path = "data/sudoku-extreme-1k-aug-1000",
        checkpoint_dir="checkpoints/Sudoku-extreme-1k-aug-1000 ACT-torch/HierarchicalReasoningModel_ACTV1 judicious-dinosaur",  # or a directory containing all_config.yaml
        ckpt_path=None, # "checkpoints/sudoku/checkpoint",       # or a direct path like "checkpoints/sudoku/checkpoint"
        hf_repo="sapientinc/HRM-checkpoint-sudoku-extreme",
        hf_filename="model.pt",
        puzzle_index=None,
        seed=0,
    )
