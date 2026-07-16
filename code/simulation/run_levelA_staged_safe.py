#!/usr/bin/env python3
"""
run_levelA_staged_safe.py
=========================
Safe, SEQUENTIAL, staged runner for Level-A helical 300 nm GLAD jobs.

One alpha/seed per invocation. Each job is grown in small target-height stages
(default 150 -> 200 -> 250 -> 275 -> 300 nm) instead of one shot, so the
active-slab-only GPU path and host RAM stay bounded on the 4 GB GTX 1650.

After every stage: validate the best checkpoint with h5py, copy it to
checkpoint_v3_latest.h5, record status, and STOP on any failure. Voxelization +
descriptor extraction run ONLY after the final 300 nm stage.

This is NOT the industrial wrapper and launches NO parallel jobs. It calls the
validated simulator directly:  simulation MD/src/glad_v3_core.py

Default mode is DRY-RUN. Pass --run to actually execute.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py

# --------------------------------------------------------------------------- #
# Fixed locations (WSL only)
# --------------------------------------------------------------------------- #
def find_project_root(start: Path = Path(__file__).resolve()) -> Path:
    for p in [start, *start.parents]:
        if (p / "00_PHD_KNOWLEDGE_HUB").exists() and (p / "README_PROJECT_STRUCTURE.md").exists():
            return p
    raise RuntimeError("Cannot locate /mnt/d/GLAD_PROJECT root")


PROJECT_ROOT = find_project_root()
ROOT = PROJECT_ROOT
GLAD_ROOT = PROJECT_ROOT / "01_GLAD_SIMULATION"
PINN_NIF_ROOT = PROJECT_ROOT / "02_PINN_NIF"
VOXEL_DESC_ROOT = PROJECT_ROOT / "03_VOXEL_DESCRIPTOR_ANALYSIS"
LIT_ROOT = PROJECT_ROOT / "04_LITERATURE_AND_PARAMETER_WORKFLOW"
THESIS_ROOT = PROJECT_ROOT / "05_THESIS_PAPER_ASSETS"
EXPERIMENTAL_ROOT = PROJECT_ROOT / "06_EXPERIMENTAL_VALIDATION"
UTIL_ROOT = PROJECT_ROOT / "07_UTILITIES_GENERATORS"
LEGACY_ROOT = PROJECT_ROOT / "08_LEGACY_REVIEW"

PYTHON = "/home/administrateur/miniforge3/envs/gladwsl/bin/python"
SIM_CORE = GLAD_ROOT / "simulation MD" / "src" / "glad_v3_core.py"
VOXELIZER = VOXEL_DESC_ROOT / "tools" / "voxelize_glad_v3_run_to_nif.py"
DESCRIPTOR = VOXEL_DESC_ROOT / "tools" / "extract_descriptors_for_voxel_job.py"

LEVELA_STAGE = "levelA_helical_alpha_300nm"
RUNS_DIR = GLAD_ROOT / "simulation_batch" / "runs" / LEVELA_STAGE
STAGED_DIR = GLAD_ROOT / "simulation_batch" / "runs" / "staged_levelA"
LOG_DIR = GLAD_ROOT / "simulation_batch" / "runs" / "wsl_manual_launch_logs"
VOXEL_DIR = PINN_NIF_ROOT / "pinn_training" / "datasets" / "universal_glad_pipeline" / "voxels"
DESC_DIR = VOXEL_DESC_ROOT / "descriptor_results_hybrid" / "universal_glad_pipeline"
DESC_CSV = DESC_DIR / "descriptors.csv"
DESC_JSONL = DESC_DIR / "descriptors.jsonl"

# Jobs already completed + accepted -> never touched by this runner.
# Keys are (fmt_alpha, seed_str); fmt_alpha(60) == "060".
ACCEPTED_JOBS = {("060", "0")}  # alpha60 seed0

EXPECTED_CD = 0.2
HEIGHT_TOL = 0.5            # nm; a stage target is "reached" within this tolerance
DEFAULT_STAGES = [150.0, 200.0, 250.0, 275.0, 300.0]
FINAL_TARGET = 300.0
FATAL_LOG_MARKERS = ("std::bad_alloc", "MemoryError", "Traceback (most recent call last)",
                     "[FATAL]", "[CRITICAL]", "CUDA_ERROR", "out of memory")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(p)


# --------------------------------------------------------------------------- #
# Job identity (mirrors master_run_universal_glad_pipeline.py)
# --------------------------------------------------------------------------- #
def fmt_alpha(alpha: float) -> str:
    return f"{int(alpha):03d}" if float(alpha).is_integer() else str(alpha).replace(".", "p")


def make_job_id(alpha: float, seed: int, height: float, mode: str, pitch: float) -> str:
    parts = [LEVELA_STAGE.replace("level", "L"), f"alpha{fmt_alpha(alpha)}",
             f"seed{seed:03d}", f"{int(height)}nm"]
    if mode:
        parts.append(mode)
    if pitch is not None and pitch < 1e11:
        parts.append(f"pitch{int(pitch)}")
    return "_".join(parts)


def job_dir(job_id: str) -> Path:
    return RUNS_DIR / job_id


def seeds_from_policy(alpha: float, policy: dict[str, Any]) -> list[int]:
    seeds = [int(s) for s in policy.get("default", [])]
    extra = policy.get("extra", {})
    for key in (str(int(alpha)), str(float(alpha))):
        if key in extra:
            seeds.extend(int(s) for s in extra[key])
    return sorted(set(seeds))


def levelA_job_specs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    st = cfg["stages"][LEVELA_STAGE]
    defaults = cfg.get("defaults", {})
    pitch = float(st.get("pitch_nm", defaults.get("pitch_nm", 150)))
    height = float(st.get("height_nm", 300))
    mode = st.get("rotation_mode", "helical")
    policy = st.get("seed_policy", {"default": [0]})
    awh = float(st.get("active_window_height", defaults.get("active_window_height", 20.0)) or 20.0)
    rawh = float(st.get("resume_active_window_height",
                        defaults.get("resume_active_window_height", 20.0)) or 20.0)
    batch = int(defaults.get("batch_size", 512))
    specs = []
    for a in st["alphas"]:
        for s in seeds_from_policy(float(a), policy):
            specs.append({
                "alpha": float(a), "seed": int(s), "pitch": pitch, "height": height,
                "mode": mode, "active_window_height": awh,
                "resume_active_window_height": rawh, "batch_size": batch,
                "job_id": make_job_id(float(a), int(s), height, mode, pitch),
            })
    return specs


# --------------------------------------------------------------------------- #
# Checkpoint validation / selection
# --------------------------------------------------------------------------- #
def validate_checkpoint(p: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"file": str(p), "name": p.name, "valid": False, "reason": ""}
    try:
        info["file_size_bytes"] = p.stat().st_size
        with h5py.File(p, "r") as f:
            if "positions" not in f or not isinstance(f["positions"], h5py.Dataset):
                info["reason"] = "no positions dataset"
                return info
            d = f["positions"]
            shape = tuple(int(x) for x in d.shape)
            ch = float(f.attrs.get("current_height", float("nan")))
            schema = str(f.attrs.get("schema", ""))
            cg = str(f.attrs.get("contact_geometry", ""))
            cd = float(f.attrs.get("collision_diameter", -1.0))
            alpha = float(f.attrs.get("alpha_deg", -1.0))
            pitch = float(f.attrs.get("pitch", -1.0))
            info.update(positions_shape=list(shape), current_height=ch, schema=schema,
                        contact_geometry=cg, collision_diameter=cd, alpha_deg=alpha, pitch=pitch)
            problems = []
            if not (len(shape) == 2 and shape[1] in (3, 4) and shape[0] > 0):
                problems.append(f"bad shape {shape}")
            if not math.isfinite(ch):
                problems.append("non-finite current_height")
            if not schema.endswith("v3.2"):
                problems.append(f"schema {schema!r} not v3.2")
            if cg != "2R_physical_contact":
                problems.append(f"contact_geometry {cg!r}")
            if not (abs(cd - EXPECTED_CD) < 1e-3):
                problems.append(f"collision_diameter {cd}")
            info["valid"] = not problems
            info["reason"] = "OK" if not problems else "; ".join(problems)
    except Exception as e:  # noqa: BLE001
        info["reason"] = f"open error: {e}"
    return info


def find_best_checkpoint(ckpt_dir: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Return (best_valid_info_or_None, all_infos). Ignores *_BAD* files. Best =
    valid checkpoint with the highest finite current_height."""
    all_infos: list[dict[str, Any]] = []
    if not ckpt_dir.exists():
        return None, all_infos
    for p in sorted(ckpt_dir.glob("*.h5")):
        if "_BAD" in p.name:
            continue
        all_infos.append(validate_checkpoint(p))
    valid = [i for i in all_infos if i.get("valid")]
    if not valid:
        return None, all_infos
    best = max(valid, key=lambda i: float(i.get("current_height", -1.0)))
    return best, all_infos


def print_checkpoint_info(info: dict[str, Any]) -> None:
    print(f"    file              : {info.get('name')}")
    print(f"    current_height    : {info.get('current_height')}")
    print(f"    positions_shape   : {info.get('positions_shape')}")
    print(f"    schema            : {info.get('schema')}")
    print(f"    contact_geometry  : {info.get('contact_geometry')}")
    print(f"    collision_diameter: {info.get('collision_diameter')}")
    print(f"    alpha_deg         : {info.get('alpha_deg')}")
    print(f"    pitch             : {info.get('pitch')}")
    print(f"    file_size_bytes   : {info.get('file_size_bytes')}")
    print(f"    valid             : {info.get('valid')}  ({info.get('reason')})")


# --------------------------------------------------------------------------- #
# Per-stage YAML config
# --------------------------------------------------------------------------- #
def stage_config_path(jd: Path, target: float) -> Path:
    return jd / f"case_config_stage{int(target)}_window20.yaml"


def write_stage_config(spec: dict[str, Any], jd: Path, target: float) -> Path:
    ckpt_dir = jd / "checkpoints"
    log_file = jd / f"simulation_stage{int(target)}.log"
    text = f"""# Auto-generated by run_levelA_staged_safe.py  ({now_iso()})
# Staged Level-A config: one target-height stage of a sequential resume.
# Active-slab-only memory controls are pinned to 20 nm for 4 GB-GPU safety.
simulation:
  target_height: {float(target)}
  alpha: {spec['alpha']}
  random_seed: {spec['seed']}
  material: Cu
  substrate_spacing: 1.0
performance:
  use_gpu: true
  batch_size: {spec['batch_size']}
box:
  width: 100
  depth: 100
physics:
  pitch: {spec['pitch']}
  author_radius_nm: 0.1
  growth_rate: 0.3
  enable_surface_diffusion: false
  use_arrhenius: false
  active_window_height: {spec['active_window_height']}
  resume_active_window_height: {spec['resume_active_window_height']}
checkpoint:
  output_dir: "{ckpt_dir.as_posix()}"
  height_interval: 25
  particle_interval: 20000000
  time_interval: 600
  allow_legacy_resume: false
monitoring:
  log_file: "{log_file.as_posix()}"
  write_interval: 1000000
  verbose: false
  auto_postprocess: false
"""
    sp = stage_config_path(jd, target)
    sp.write_text(text, encoding="utf-8")
    return sp


# --------------------------------------------------------------------------- #
# Stage planning
# --------------------------------------------------------------------------- #
def plan_stages(stages: list[float], current_h: float) -> list[float]:
    """Stages still to run: those strictly above the current best height."""
    return [s for s in stages if s > current_h + HEIGHT_TOL]


# --------------------------------------------------------------------------- #
# Running a single stage
# --------------------------------------------------------------------------- #
def copy_to_latest(best: dict[str, Any], ckpt_dir: Path, do: bool) -> Path | None:
    latest = ckpt_dir / "checkpoint_v3_latest.h5"
    src = Path(best["file"])
    if src.resolve() == latest.resolve():
        return latest
    if do:
        shutil.copy2(src, latest)
    return latest


def run_one_stage(spec: dict[str, Any], jd: Path, target: float, use_load: bool,
                  do_run: bool) -> dict[str, Any]:
    ckpt_dir = jd / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = write_stage_config(spec, jd, target)
    latest = ckpt_dir / "checkpoint_v3_latest.h5"

    cmd = [PYTHON, "-u", str(SIM_CORE),
           "--config", str(cfg_path),
           "--target_height", str(float(target)),
           "--random_seed", str(spec["seed"])]
    if use_load:
        cmd += ["--load_v3", str(latest)]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{spec['job_id']}_stage{int(target)}_{ts}.log"
    result = {"target": target, "config": rel(cfg_path), "load_v3": rel(latest) if use_load else None,
              "stdout_log": rel(log_path), "command": cmd}

    if not do_run:
        result["status"] = "DRY_RUN"
        return result

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONUNBUFFERED"] = "1"

    print(f"\n[STAGE {int(target)}] launching simulator ...")
    print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    print(f"  tee -> {rel(log_path)}")

    t0 = time.time()
    fatal_seen = False
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env, cwd=str(ROOT))
        for line in proc.stdout:  # type: ignore[union-attr]
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
            logf.flush()
            if any(m in line for m in FATAL_LOG_MARKERS):
                fatal_seen = True
        proc.wait()
    rc = proc.returncode
    elapsed = time.time() - t0
    result.update(returncode=rc, elapsed_s=round(elapsed, 1), fatal_marker_seen=fatal_seen)

    # Re-scan checkpoints to confirm the stage actually advanced the film.
    best, _ = find_best_checkpoint(ckpt_dir)
    reached = float(best["current_height"]) if best else -1.0
    result["post_stage_best_height"] = reached
    ok = (rc == 0 and not fatal_seen and best is not None and reached >= target - HEIGHT_TOL)
    result["status"] = "STAGE_OK" if ok else "STAGE_FAILED"
    if ok:
        copy_to_latest(best, ckpt_dir, do=True)
        result["latest_updated_from"] = best["name"]
    return result


# --------------------------------------------------------------------------- #
# Post-processing (final 300 nm only)
# --------------------------------------------------------------------------- #
def run_voxelize(spec: dict[str, Any], jd: Path, do_run: bool) -> dict[str, Any]:
    latest = jd / "checkpoints" / "checkpoint_v3_latest.h5"
    vox = VOXEL_DIR / f"rho_{spec['job_id']}.npy"
    cmd = [PYTHON, str(VOXELIZER),
           "--input", str(latest),
           "--output", str(vox),
           "--job-id", spec["job_id"],
           "--alpha", str(int(spec["alpha"]) if float(spec["alpha"]).is_integer() else spec["alpha"]),
           "--seed", str(spec["seed"]),
           "--height-nm", "300",
           "--pitch-nm", str(int(spec["pitch"])),
           "--rotation-mode", "helical",
           "--box-width-nm", "100", "--box-depth-nm", "100",
           "--voxel-size-xy-nm", "2", "--voxel-size-z-nm", "2",
           "--grid-nx", "50", "--grid-ny", "50",
           "--xy-periodic-wrap", "--force"]
    out = {"command": cmd, "voxel": rel(vox), "metadata": rel(vox.with_suffix(".json"))}
    if not do_run:
        out["status"] = "DRY_RUN"
        return out
    VOXEL_DIR.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    out["returncode"] = rc
    out["status"] = "OK" if rc == 0 else "FAILED"
    return out


def run_descriptors(spec: dict[str, Any], do_run: bool) -> dict[str, Any]:
    vox = VOXEL_DIR / f"rho_{spec['job_id']}.npy"
    meta = vox.with_suffix(".json")
    cmd = [PYTHON, str(DESCRIPTOR),
           "--voxel", str(vox), "--metadata", str(meta),
           "--out-csv", str(DESC_CSV), "--out-json", str(DESC_JSONL)]
    out = {"command": cmd}
    if not do_run:
        out["status"] = "DRY_RUN"
        return out
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    out["returncode"] = rc
    out["status"] = "OK" if rc == 0 else "FAILED"
    return out


def voxel_status(job_id: str) -> dict[str, Any]:
    meta = VOXEL_DIR / f"rho_{job_id}.json"
    if not meta.exists():
        return {"exists": False}
    try:
        d = json.loads(meta.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return {"exists": True, "error": str(e)}
    return {
        "exists": True,
        "adapter_status": d.get("adapter_status"),
        "voxel_grid_crop_status": d.get("voxel_grid_crop_status"),
        "crop_fraction_estimate": d.get("crop_fraction_estimate"),
        "post_wrap_out_xy_fraction": d.get("post_wrap_out_xy_fraction"),
        "warnings": d.get("warnings"),
        "grid_shape": d.get("grid_shape"),
        "source_file": d.get("source_file"),
        "height_nm": d.get("height_nm"),
    }


def descriptor_row_count(job_id: str) -> int:
    if not DESC_CSV.exists():
        return 0
    n = 0
    with DESC_CSV.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("job_id") == job_id:
                n += 1
    return n


# --------------------------------------------------------------------------- #
# run_metadata.json
# --------------------------------------------------------------------------- #
def read_run_metadata(jd: Path) -> dict[str, Any]:
    p = jd / "run_metadata.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_run_metadata(spec: dict[str, Any], jd: Path, status: str,
                       extra: dict[str, Any]) -> None:
    p = jd / "run_metadata.json"
    base = read_run_metadata(jd)
    base.update({
        "job_id": spec["job_id"],
        "stage": LEVELA_STAGE,
        "alpha": spec["alpha"],
        "seed": spec["seed"],
        "pitch": spec["pitch"],
        "height": spec["height"],
        "rotation_mode": spec["mode"],
        "active_window_height": spec["active_window_height"],
        "resume_active_window_height": spec["resume_active_window_height"],
        "status": status,
        "updated_at": now_iso(),
        "runner": "run_levelA_staged_safe.py",
    })
    base.update(extra)
    p.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Queue audit (Task 1) + global status (Task 8)
# --------------------------------------------------------------------------- #
def job_overview(spec: dict[str, Any], stages: list[float]) -> dict[str, Any]:
    jid = spec["job_id"]
    jd = job_dir(jid)
    fa = fmt_alpha(spec["alpha"])
    accepted = (fa, str(spec["seed"])) in ACCEPTED_JOBS
    best, _ = find_best_checkpoint(jd / "checkpoints")
    h = float(best["current_height"]) if best else 0.0
    meta = read_run_metadata(jd)
    vs = voxel_status(jid)
    drows = descriptor_row_count(jid)

    if accepted:
        status = "ACCEPTED_COMPLETED"
    elif meta.get("status") == "DESCRIPTORS_EXTRACTED":
        status = "DESCRIPTORS_EXTRACTED"
    elif best is None:
        status = "FRESH"
    elif h >= FINAL_TARGET - HEIGHT_TOL:
        status = "SIMULATION_COMPLETE_PENDING_POSTPROCESS"
    else:
        status = f"PARTIAL@{h:.1f}nm"

    todo = plan_stages(stages, h)
    next_target = (todo[0] if todo else (FINAL_TARGET if status.startswith("SIMULATION_COMPLETE") else None))

    if accepted:
        action = "SKIP (accepted)"
    elif status == "DESCRIPTORS_EXTRACTED":
        action = "SKIP (done)"
    elif status.startswith("SIMULATION_COMPLETE"):
        action = "POSTPROCESS ONLY (voxel+descriptors)"
    elif best is None:
        action = f"FRESH start, first stage {int(stages[0])}"
    else:
        action = f"RESUME from {h:.1f}nm, next stage {int(next_target)}"

    return {
        "job_id": jid,
        "alpha": spec["alpha"],
        "seed": spec["seed"],
        "pitch": spec["pitch"],
        "current_best_checkpoint_height": round(h, 3) if best else "",
        "current_best_checkpoint_file": best["name"] if best else "",
        "current_status": status,
        "next_stage_target": int(next_target) if next_target else "",
        "final_voxel_status": vs.get("voxel_grid_crop_status", "NONE") if vs.get("exists") else "NONE",
        "descriptor_status": f"{drows} rows" if drows else "0 rows",
        "action": action,
    }


def write_queue_preview(cfg: dict[str, Any], stages: list[float]) -> list[dict[str, Any]]:
    STAGED_DIR.mkdir(parents=True, exist_ok=True)
    rows = [job_overview(spec, stages) for spec in levelA_job_specs(cfg)]
    cols = ["job_id", "alpha", "seed", "pitch", "current_best_checkpoint_height",
            "current_status", "next_stage_target", "final_voxel_status",
            "descriptor_status", "action"]
    csv_path = STAGED_DIR / "staged_levelA_queue_preview.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    md = [f"# Staged Level-A Queue Preview", "", f"Generated: {now_iso()}",
          f"Stages: {', '.join(str(int(s)) for s in stages)} nm", "",
          "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        md.append("| " + " | ".join(str(r.get(k, "")) for k in cols) + " |")
    (STAGED_DIR / "staged_levelA_queue_preview.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return rows


def write_global_status(cfg: dict[str, Any], stages: list[float]) -> None:
    STAGED_DIR.mkdir(parents=True, exist_ok=True)
    rows = [job_overview(spec, stages) for spec in levelA_job_specs(cfg)]
    (STAGED_DIR / "staged_levelA_status.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    cols = ["job_id", "alpha", "seed", "current_best_checkpoint_height", "current_status",
            "next_stage_target", "final_voxel_status", "descriptor_status", "action"]
    with (STAGED_DIR / "staged_levelA_status.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    md = [f"# Staged Level-A Status", "", f"Generated: {now_iso()}", "",
          "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        md.append("| " + " | ".join(str(r.get(k, "")) for k in cols) + " |")
    (STAGED_DIR / "staged_levelA_status.md").write_text("\n".join(md) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Drive one job
# --------------------------------------------------------------------------- #
def run_job(spec: dict[str, Any], stages: list[float], args: argparse.Namespace) -> int:
    jid = spec["job_id"]
    jd = job_dir(jid)
    jd.mkdir(parents=True, exist_ok=True)
    ckpt_dir = jd / "checkpoints"
    do_run = bool(args.run) and not args.dry_run
    fa = fmt_alpha(spec["alpha"])

    print("=" * 72)
    print(f"JOB: {jid}")
    print(f"alpha={spec['alpha']} seed={spec['seed']} pitch={spec['pitch']} "
          f"window={spec['active_window_height']}/{spec['resume_active_window_height']} nm")
    print(f"stages: {', '.join(str(int(s)) for s in stages)} nm")
    print(f"mode: {'RUN' if do_run else 'DRY-RUN'}")
    print("=" * 72)

    if (fa, str(spec["seed"])) in ACCEPTED_JOBS:
        print("REFUSING: this job is ACCEPTED/COMPLETED (alpha60 seed0). Not touched.")
        return 0

    # --- checkpoint discovery / validation ---
    if args.force_fresh:
        print("[force-fresh] ignoring existing checkpoints; starting from substrate.")
        best = None
        all_infos = []
    else:
        best, all_infos = find_best_checkpoint(ckpt_dir)

    print(f"\nCheckpoint scan in {rel(ckpt_dir)}: {len(all_infos)} file(s)")
    for info in all_infos:
        flag = "VALID" if info.get("valid") else "INVALID"
        print(f"  [{flag}] {info.get('name')}  h={info.get('current_height')}  ({info.get('reason')})")
    fresh = best is None
    if best:
        print("\nBest valid checkpoint:")
        print_checkpoint_info(best)
        current_h = float(best["current_height"])
    else:
        if all_infos and not args.force_fresh:
            # files exist but none valid -> refuse (do not silently start fresh)
            print("\nFATAL: checkpoints exist but NONE are valid. Refusing to start fresh.")
            print("       Inspect/rename the bad files (with backup) and retry.")
            return 2
        print("\nNo valid checkpoint found -> fresh substrate start.")
        current_h = 0.0

    todo = plan_stages(stages, current_h)
    if args.max_stages:
        todo = todo[: args.max_stages]

    # only-final-postprocess: skip simulation, go straight to voxel/descriptors
    if args.only_final_postprocess:
        if fresh or current_h < FINAL_TARGET - HEIGHT_TOL:
            print(f"\nFATAL: --only-final-postprocess but best height {current_h:.1f} < {FINAL_TARGET}.")
            return 2
        todo = []

    print(f"\nPlanned stages to run: {[int(s) for s in todo] if todo else '(none — already at/above final)'}")

    # --- pre-stage: ensure latest points to best (so resume uses validated best) ---
    if not fresh and do_run:
        copy_to_latest(best, ckpt_dir, do=True)
        print(f"[latest] checkpoint_v3_latest.h5 <- {best['name']}")
    elif not fresh:
        print(f"[latest] (dry-run) would copy {best['name']} -> checkpoint_v3_latest.h5")

    stage_results = []
    for i, target in enumerate(todo):
        use_load = (not fresh) or (i > 0)
        if not do_run:
            res = run_one_stage(spec, jd, target, use_load, do_run=False)
            print(f"\n[DRY-RUN stage {int(target)}] load_v3={'yes' if use_load else 'no (fresh)'}")
            print("  cmd: " + " ".join(f'"{c}"' if " " in c else c for c in res["command"]))
            stage_results.append(res)
            continue

        res = run_one_stage(spec, jd, target, use_load, do_run=True)
        stage_results.append(res)
        write_run_metadata(spec, jd, f"STAGE_{int(target)}_{res['status']}",
                           {"last_stage": res})
        if res["status"] != "STAGE_OK":
            print(f"\n[STOP] Stage {int(target)} FAILED "
                  f"(rc={res.get('returncode')}, fatal={res.get('fatal_marker_seen')}, "
                  f"best_h={res.get('post_stage_best_height')}). Halting per stop-on-failure.")
            write_run_metadata(spec, jd, f"FAILED_STAGE_{int(target)}", {"last_stage": res})
            write_global_status(load_config(args.config), stages)
            return 1
        print(f"[STAGE {int(target)}] OK — best height now {res['post_stage_best_height']:.2f} nm")

    # --- final post-processing (only if we are at/above 300) ---
    best, _ = find_best_checkpoint(ckpt_dir)
    final_h = float(best["current_height"]) if best else current_h
    at_final = final_h >= FINAL_TARGET - HEIGHT_TOL

    postproc = {}
    if args.skip_voxel:
        print("\n[skip-voxel] skipping voxelization + descriptors as requested.")
        status = "SIMULATION_COMPLETE" if at_final else f"PARTIAL@{final_h:.1f}nm"
        if do_run:
            write_run_metadata(spec, jd, status, {"final_height_nm": final_h})
    elif at_final:
        will_execute = do_run or args.only_final_postprocess
        if not will_execute:
            print(f"\n[DRY-RUN] already at/above 300 nm (best {final_h:.2f} nm). "
                  "Would run voxelization + descriptors (use --run or --only-final-postprocess).")
        else:
            print("\n[POSTPROCESS] voxelization ...")
            vx = run_voxelize(spec, jd, do_run=True)
            postproc["voxelize"] = vx
            vs = voxel_status(jid)
            crop_ok = (vs.get("voxel_grid_crop_status") == "OK"
                       and (vs.get("crop_fraction_estimate") or 1.0) <= 0.01
                       and not vs.get("warnings"))
            if not crop_ok:
                print(f"[STOP] Voxel crop check failed: {vs}")
                write_run_metadata(spec, jd, "SIMULATION_COMPLETED_BUT_VOXEL_FAILED",
                                   {"final_height_nm": final_h, "voxel": vs, "postproc": postproc})
                write_global_status(load_config(args.config), stages)
                return 1
            print("[POSTPROCESS] descriptors ...")
            dx = run_descriptors(spec, do_run=True)
            postproc["descriptors"] = dx
            drows = descriptor_row_count(jid)
            if drows != 15:
                print(f"[STOP] descriptor row count = {drows} (expected 15).")
                write_run_metadata(spec, jd, "SIMULATION_COMPLETED_BUT_DESCRIPTORS_FAILED",
                                   {"final_height_nm": final_h, "voxel": vs,
                                    "descriptor_rows": drows, "postproc": postproc})
                write_global_status(load_config(args.config), stages)
                return 1
            write_run_metadata(spec, jd, "DESCRIPTORS_EXTRACTED", {
                "final_height_nm": final_h,
                "final_atoms": (best or {}).get("positions_shape", [None])[0],
                "checkpoint_latest": rel(ckpt_dir / "checkpoint_v3_latest.h5"),
                "voxel_path": vs_path(jid, ".npy"),
                "voxel_metadata_path": vs_path(jid, ".json"),
                "descriptor_rows": drows,
                "crop_fraction_estimate": vs.get("crop_fraction_estimate"),
                "voxel_grid_crop_status": vs.get("voxel_grid_crop_status"),
                "accepted_for_nif_training": True,
                "postproc": postproc,
            })
            print(f"\n[ACCEPTED] {jid}: 300 nm + voxel OK + 15 descriptors -> DESCRIPTORS_EXTRACTED")
    else:
        print(f"\nNot at final 300 nm (best {final_h:.1f} nm); no post-processing this invocation.")
        if do_run:
            write_run_metadata(spec, jd, f"PARTIAL@{final_h:.1f}nm", {"final_height_nm": final_h})

    if do_run:
        write_global_status(load_config(args.config), stages)
    return 0


def vs_path(job_id: str, suffix: str) -> str:
    return rel(VOXEL_DIR / f"rho_{job_id}{suffix}")


# --------------------------------------------------------------------------- #
def load_config(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="pipeline JSON config")
    p.add_argument("--alpha", type=float, default=None, help="single alpha to run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stages", default="150,200,250,275,300", help="comma-separated stage targets (nm)")
    p.add_argument("--run", action="store_true", help="actually execute (default is dry-run)")
    p.add_argument("--dry-run", action="store_true", help="force dry-run even with --run")
    p.add_argument("--max-stages", type=int, default=None, help="cap number of stages this invocation")
    p.add_argument("--stop-on-failure", action="store_true", default=True,
                   help="halt on first failing stage (always on)")
    p.add_argument("--skip-voxel", action="store_true", help="skip voxel+descriptor post-processing")
    p.add_argument("--only-final-postprocess", action="store_true",
                   help="skip simulation; only voxel+descriptors on existing 300 nm checkpoint")
    p.add_argument("--force-fresh", action="store_true", help="ignore existing checkpoints; fresh start")
    p.add_argument("--resume", action="store_true", help="resume from best valid checkpoint (default behaviour)")
    p.add_argument("--audit-queue", action="store_true",
                   help="scan ALL remaining Level-A jobs, write queue preview, run nothing")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not SIM_CORE.exists():
        print(f"FATAL: simulator not found at {SIM_CORE}")
        return 2
    cfg = load_config(args.config)
    stages = [float(s) for s in str(args.stages).split(",") if s.strip()]

    STAGED_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.audit_queue:
        rows = write_queue_preview(cfg, stages)
        print(f"Queue preview written to {rel(STAGED_DIR)}/staged_levelA_queue_preview.(csv|md)")
        print(f"{'job_id':52} {'status':36} {'next':>6} {'action'}")
        for r in rows:
            print(f"{r['job_id']:52} {r['current_status']:36} "
                  f"{str(r['next_stage_target']):>6} {r['action']}")
        return 0

    if args.alpha is None:
        print("FATAL: --alpha is required (or use --audit-queue). One alpha/seed per invocation.")
        return 2

    specs = levelA_job_specs(cfg)
    match = [s for s in specs if abs(s["alpha"] - args.alpha) < 1e-9 and s["seed"] == args.seed]
    if not match:
        print(f"FATAL: alpha={args.alpha} seed={args.seed} is not a Level-A 300 nm job in {args.config}.")
        valid = sorted({(s["alpha"], s["seed"]) for s in specs})
        print(f"       valid (alpha,seed): {valid}")
        return 2

    # always refresh queue preview so it reflects reality
    write_queue_preview(cfg, stages)
    return run_job(match[0], stages, args)


if __name__ == "__main__":
    raise SystemExit(main())
