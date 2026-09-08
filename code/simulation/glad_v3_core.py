"""
glad_v3_core.py
===============
GLAD V3.0 -- Full 3D Bead-Spring Simulator
==========================================
Merges:
  - GPU/CuPy batching performance from glad_v2.py
  - 3D ray-sphere quadratic intersection physics (author / vpython_code.py norm)
  - cKDTree spatial partitioning for O(N log N) complexity (glad_high_fidelity.py)
  - Arrhenius surface diffusion (glad_v2.py)
  - LAMMPS .lammpstrj + .xyz output (compatible with glad_analytics.py)
  - HDF5 checkpoint save/load (V3 format from migrate_legacy.py)

GPU backend (GTX 1650 / CuPy):
  - The quadratic intersection kernel is vectorised over the whole incoming
    batch (up to BATCH_SIZE particles at once) using CuPy arrays.
  - CPU NumPy fallback activates automatically if CuPy is not installed.

Spatial partitioning:
  - scipy.spatial.cKDTree is rebuilt every KDTREE_REBUILD_INTERVAL atoms.
  - For each incoming particle, candidate neighbours are queried within
    a cylinder defined by the shadow range: r_search = max_height / tan(alpha).
    This gives an O(N log N) overall simulation.

Usage
-----
    python glad_v3_core.py [--config glad_config.yaml]
                           [--load_v3 checkpoints/checkpoint_V3_READY.h5]
                           [--n_particles 1000]
                           [--out trajectory.xyz]
"""
import os
import sys
import queue
import threading

# Pause file lives next to the .ps1 scripts, one directory above this script.
# CWD at launch time is unreliable (depends on how run_glad.py sets cwd).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PAUSE_FILE  = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'pause_v3.txt'))

# 1. ENFORCE CUDA DLL INJECTION (GTX 1650 FIX)
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    # Point directly to the .conda Library folder where CUDA binaries reside
    conda_base = os.path.dirname(sys.executable)
    cuda_path = os.path.join(conda_base, "Library")
    os.environ["CUDA_PATH"] = cuda_path
    # Prepend Library/bin to PATH so CuPy can find nvrtc64_*.dll, etc.
    cuda_bin = os.path.join(cuda_path, "bin")
    if os.path.exists(cuda_bin):
        os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
    
    # Ensure stdout handles UTF-8 correctly to avoid some Windows terminal crashes
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8')

# Avoid Windows/CuPy disk-cache lockups during NVRTC RawModule compilation.
# The compiled kernels are small and startup reliability matters more than
# persisting them between runs.
os.environ.setdefault("CUPY_CACHE_IN_MEMORY", "1")

import time
import json
import hashlib
import argparse
import signal
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import h5py
import yaml
from scipy.spatial import cKDTree
import subprocess

def get_gpu_temp():
    """Legacy one-shot query — superseded by GPUThermalMonitor below."""
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL)
        return int(res.decode().strip())
    except:
        return 0


class GPUThermalMonitor:
    """
    Throttled GPU temperature sampler.

    Queries nvidia-smi at most once every `interval_s` seconds to avoid the
    ~50 ms OS overhead of spawning a subprocess on every batch.  The cached
    value is returned between polls, making per-batch thermal reads effectively
    free.
    """
    def __init__(self, interval_s: float = 5.0):
        self._interval = interval_s
        self._last_poll = 0.0
        self._cached_temp = 0

    def temperature(self) -> int:
        now = time.time()
        if now - self._last_poll >= self._interval:
            self._cached_temp = get_gpu_temp()
            self._last_poll = now
        return self._cached_temp
# ---------------------------------------------------------------------------
# CUDA DLL hard-link loader (ctypes) -- MUST BE CALLED BEFORE CUPY
# ---------------------------------------------------------------------------
_CUDA_DLL_DIR_HANDLES = []

def _hardlink_cuda_dlls():
    """
    Explicitly load CUDA DLLs via ctypes.CDLL.
    """
    import ctypes
    import os
    import sys
    import site
    import glob

    packages_dirs = site.getsitepackages() + [os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages")]
    search_roots = [os.path.join(d, "nvidia") for d in packages_dirs]
    search_roots += [os.path.join(os.path.dirname(sys.executable), "Library", "bin")]
    search_roots += [os.path.join(os.path.dirname(sys.executable), "bin")]
    search_roots += [os.path.dirname(os.path.dirname(sys.executable))] # conda root
    
    loaded = []
    # Identify unique directories that contain NVIDIA/CUDA DLLs
    dll_dirs = set()
    for root in search_roots:
        if not os.path.exists(root): continue
        for r, d, files in os.walk(root):
            if any(f.lower().endswith(".dll") and (f.lower().startswith("cu") or "nv" in f.lower()) for f in files):
                dll_dirs.add(os.path.abspath(r))

    for d in dll_dirs:
        try:
            if hasattr(os, "add_dll_directory"):
                _CUDA_DLL_DIR_HANDLES.append(os.add_dll_directory(d))
        except Exception: pass
        if d not in os.environ["PATH"]:
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")

    # Pre-emptively load the core CUDA DLLs
    CORE_DLLS = ["nvrtc64_120_0.dll", "nvrtc64_112_0.dll", "nvrtc-builtins64_12.dll", "nvrtc-builtins64_112.dll", "nvrtc-builtins64_118.dll", 
                 "cudart64_12.dll", "cudart64_110.dll", "cublas64_12.dll", "cublas64_11.dll", "cusparse64_12.dll", "cusparse64_11.dll", 
                 "curand64_10.dll", "cusolver64_12.dll", "cusolver64_11.dll", "cufft64_10.dll"]
    for root in search_roots:
        if not os.path.exists(root): continue
        for r, d, files in os.walk(root):
            for f in files:
                if f.lower() in CORE_DLLS:
                    try:
                        ctypes.CDLL(os.path.join(r, f))
                        loaded.append(f)
                    except Exception as e:
                        pass
    if loaded:
        print(f"[GPU-FORCE] Linked {len(set(loaded))} unique CUDA binaries.", flush=True)

if sys.platform == "win32":
    _hardlink_cuda_dlls()

# ---------------------------------------------------------------------------
# GPU backend selection
# ---------------------------------------------------------------------------
GPU_AVAILABLE = False
xp = np   # default to NumPy

# PERF-1 (2026-09-06, validated). The CUDA managed-memory allocator below lets allocations
# page to host RAM when 4GB VRAM is exhausted (needed for box >= ~175nm at r=0.128). But
# managed memory is ~13x SLOWER than a plain device pool for this sim's rebuild-heavy alloc
# pattern when VRAM is NOT exhausted -- measured on a real box=100 run: plain 19,740 rays/s
# vs managed 1,488 rays/s, and the two runs produced BYTE-IDENTICAL atom positions
# (max |delta| = 0). See 01_GLAD_SIMULATION/PERF1_ALLOCATOR_VALIDATION_20260906/. Selection:
#   1. env var GLAD_GPU_ALLOCATOR = "plain" | "managed"  -> pins the choice
#   2. otherwise GLADV3Simulator.__init__ auto-picks by box_width (plain if <=160nm)
#   3. module default here stays MANAGED so any import-only caller keeps today's behavior.
_ALLOC_ENV = os.environ.get("GLAD_GPU_ALLOCATOR", "").strip().lower()
_ALLOC_ENV_FORCED = _ALLOC_ENV in ("plain", "managed")

try:
    import cupy as cp
    if _ALLOC_ENV == "plain":
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        print("[GPU-FORCE] allocator: PLAIN device MemoryPool (GLAD_GPU_ALLOCATOR=plain)", flush=True)
    else:
        cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
    xp = cp
    GPU_AVAILABLE = True
    _dev = cp.cuda.Device(0)
    _mem = _dev.mem_info
    print("=" * 55, flush=True)
    print(f"[GPU-FORCE] [OK] CuPy ACTIVE -- GPU acceleration ENABLED", flush=True)
    print(f"            Device   : CUDA Device {_dev.id}", flush=True)
    print(f"            VRAM     : {_mem[1]//1024**2} MB total, {_mem[0]//1024**2} MB free", flush=True)
    print(f"            CuPy ver : {cp.__version__}", flush=True)
    print("=" * 55, flush=True)
except Exception as _e:
    print(f"[GPU-FORCE] [!] CuPy not fully available ({_e})", flush=True)
    GPU_AVAILABLE = False
    xp = np


# CPU core count for parallel KDTree
import multiprocessing as _mp
_CPU_WORKERS = _mp.cpu_count()

# ---------------------------------------------------------------------------
# Process priority -- HIGH to prevent Windows throttling background jobs
# ---------------------------------------------------------------------------
try:
    import psutil as _psutil
    _proc = _psutil.Process(os.getpid())
    _proc.nice(_psutil.HIGH_PRIORITY_CLASS)   # Windows HIGH priority
    print(f"[V3] [OK] Process priority set to HIGH (PID {os.getpid()})", flush=True)
except Exception as _pe:
    print(f"[V3] Note: Could not set HIGH priority ({_pe}) -- continuing at normal priority", flush=True)



# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _np(arr):
    """Ensure array is on CPU NumPy (no-op if already numpy)."""
    if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return arr.get()
    return arr


def _gpu(arr):
    """Send array to GPU if available."""
    if GPU_AVAILABLE:
        return cp.asarray(arr)
    return arr


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AUTHOR_RADIUS       = np.float32(0.128)    # nm — Cordero 2008 Cu covalent radius
COLLISION_DIAMETER  = np.float32(0.256)    # nm — physical contact distance = 2 * AUTHOR_RADIUS
# PHYSICS NOTE: Ray-sphere intersection must use COLLISION_DIAMETER (2R), NOT AUTHOR_RADIUS (1R).
# A new atom (radius R) contacts an existing atom (radius R) when center-to-center = 2R.
# Using 1R places new atoms halfway inside existing ones, halving the effective column cross-section
# and severely reducing shadowing, which causes near-zero column tilt instead of the expected
# Tangent-rule angle beta = atan(tan(alpha)/2) ~ 80 deg at alpha=85 deg.
# Tait-style estimate is lower; validation reports both conventions.
KDTREE_REBUILD_INT  = 500                  # rebuild cKDTree every N atoms
MATERIAL_MELTING_POINTS_K = {
    'Cu': 1357.77,   # 1084.62 C; used only for Ts/Tm, not as substrate temperature
    'CuO': 1599.0,   # approximate oxide reference
}

# Per-material atom/bead radius default (nm). Cu=0.128nm is the pre-existing,
# already-correct value (Cordero et al. 2008 covalent radius, kept as-is -- see
# AUTHOR_RADIUS below). Other entries use the SAME source/convention, matching this
# project's own canonical table at 02_PINN_NIF/P3_ML/atomic_radius_lookup.py
# (COVALENT_RADIUS_PM, Cordero 2008), converted pm -> nm.
# Found 2026-08-23: `material` was previously metadata-only for atom size -- every
# non-Cu-labeled run used Cu's 0.128nm radius regardless of the material field.
# Explicit author_radius_nm in a config always overrides this (see load_config()).
MATERIAL_ATOM_RADIUS_NM = {
    'Cu': 0.128,   # kept exactly as the pre-existing AUTHOR_RADIUS value, not overwritten
    'Ag': 0.145,   # Cordero 2008 covalent radius, 145 pm (atomic_radius_lookup.py COVALENT_RADIUS_PM['Ag'])
}

# ── Native Helical-GLAD dynamics ─────────────────────────────────────────────
# These govern the dynamic flux rotation and the 4D timestamp schema.
# Both are exposed in glad_config.yaml under the [physics] section.
PITCH               = 250.0   # nm  — one full substrate rotation per PITCH nm of growth
GROWTH_RATE         = 0.2     # nm/s — physical vertical rate; T = Z / GROWTH_RATE


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(yaml_path: str) -> dict:
    """Load glad_config.yaml and return a flat config dict."""
    angular_defaults = {
        'enabled': False,
        'type': 'deterministic_cone_ring',
        'angular_sigma_deg': 0.0,
        'angular_num_samples': 1,
        'angular_seed': 0,
        'scheme': 'ring',
        'weights': 'equal',
        'rotate_with_helical_azimuth': True,
        'preserve_nominal_mean_direction': True,
    }
    defaults = {
        'n_columns': 2500,
        'target_height': 700.0,
        'alpha': 85.0,
        'material': 'Cu',
        'box_width': 100.0,
        'box_depth': 100.0,
        'r_dep': 0.2,
        'growth_efficiency': 1.0,
        'temperature': 300.0,
        'melting_point_K': None,
        'source_distance_cm': None,
        # Physical source aperture (boat/crucible/target radius). Additive: when both
        # this and source_distance_cm are provided, and angular_sigma_deg is not
        # explicitly set, angular_sigma_deg defaults to degrees(atan(source_radius_cm /
        # source_distance_cm)) instead of the arbitrary 0.0/free-parameter default.
        'source_radius_cm': None,
        'rotation_rpm': None,
        'checkpoint_height_interval': 50.0,
        'checkpoint_particle_interval': 20_000_000,
        'checkpoint_dir': './checkpoints',
        'checkpoint_filename': 'checkpoint_v3.h5',
        'allow_legacy_resume': False,
        # substrate_spacing: initial seed-monolayer grid spacing (sets seed_atoms count
        # via initialise_substrate() below), NOT a bulk/steady-state contact distance
        # (that role belongs to COLLISION_DIAMETER=2R, unchanged). Was 1.0nm with no
        # traceable origin (R03_GEOMETRY_PARAMETER_LITERATURE_AUDIT_20260902.md flagged
        # HIGH risk, no Cu lattice/literature basis). Corrected 2026-09-02 to 0.256nm =
        # 2*AUTHOR_RADIUS, crystallographically grounded two independent ways: (1) Cordero
        # et al., Dalton Trans. 2008, 2832-2838, Cu covalent radius 0.128nm, doubled; (2)
        # Cu FCC lattice constant a=3.615A (Straumanis & Yu, Acta Cryst. A25, 676 (1969);
        # Kittel, Introduction to Solid State Physics) nearest-neighbor distance a/sqrt(2)
        # = 0.2556nm -- agrees with (1) to within 0.16%. Forward-looking only: does NOT
        # retroactively change any already-run campaign/checkpoint (those keep whatever
        # substrate_spacing their own config specified/defaulted to at the time).
        'substrate_spacing': 0.256,
        'cell_size': 1.0,
        'active_window_height': 120.0,
        # Optional smaller active window applied ONLY when resuming from a
        # checkpoint. Lets large films resume on small GPUs without changing the
        # default 120 nm growth-front window. None => use active_window_height.
        'resume_active_window_height': None,
        'thermal_throttle_temp': 84,
        'thermal_critical_temp': 87,
        'enable_surface_diffusion': False,
        'diffusion_radius': 0.5,   # nm
        'diffusion_hops': 5,
        # Cap on the per-batch physical-diffusion-model hop count (GPU-cost control, see
        # `_step_batch`). Was previously READ via self.cfg.get(...) but never populated from
        # YAML by load_config() -- a real bug: any user attempt to override it in physics: was
        # silently ignored. Fixed 2026-08-24 (see P1_BETA_AND_PD_DISCREPANCY_AUDIT_20260824.md
        # H7). Default unchanged (200), so this fix alone does not change any already-run
        # result -- it only makes the parameter actually overridable, which it always should
        # have been.
        'diffusion_dwell_hops_cap': 200,
        'activation_energy': 0.041,  # eV
        'use_arrhenius': True,
        # Physically-calibrated diffusion model (additive; default OFF reproduces the
        # toy fixed-hop-count kernel above byte-for-byte). See
        # 01_GLAD_SIMULATION/P1_PHYSICAL_DIFFUSION_MODEL_SCOPE_20260822.md.
        'use_physical_diffusion_model': False,
        'nu0_hz': 8.2e11,   # attempt frequency [Hz]; Jamnig 2019 Cu-on-C cluster-diffusion default
        # D1 (2026-09-07): resolve diffusion-kernel same-batch blindness. The parallel
        # DIFFUSION_KERNEL runs one thread per diffusing atom against the FROZEN pre-batch
        # snapshot; threads never see each other's in-flight hops, so two same-batch atoms
        # can hop to < collision_diameter apart unnoticed (confirmed: 97.8-100 % of genuine
        # overlaps are intra-batch -- D1_DIFFUSION_KERNEL_CONTACT_RECHECK_DESIGN_20260907.md).
        # When True, a deterministic deposition-order post-kernel pass REVERTS any post-hop
        # atom that landed within cd of an already-accepted atom (established structure or an
        # earlier same-batch atom) back to its pre-hop position -- i.e. that atom did not hop
        # this batch. Revert, NOT lift: lifting to a z-floor applies an uphill displacement
        # the kernel's own dz>=0||Boltzmann acceptance never tested and re-introduces the
        # height ratchet. Only ever active when enable_surface_diffusion is also True.
        #
        # DISABLED 2026-09-07 (default -> False): this pass as written breaks the
        # simulator's coordinate/wrapping invariant. A/B at box=40/alpha=85/diff-ON:
        # fix OFF -> 0 atoms outside [0,box]; fix ON -> ~30 % outside, x reaching 149 nm
        # on a 40 nm box. Non-hopping atoms in DIFFUSION_KERNEL keep their raw (possibly
        # unwrapped ballistic-landing) position; hopping atoms get wrapped by the kernel's
        # fmodf. Reverting a hop restores the pre-wrap position, so the revert pass leaves
        # the diffusion batch with mixed wrapped/unwrapped atoms -> at box=100 the grid
        # loses spatial coherence and runs degenerate (~99 % ray-miss, early termination:
        # P1_DIFFUSION_DECOMP_C2_20260907 alpha 82/75). The same-batch-blindness problem
        # is real and still open; the fix needs to preserve wrapping exactly. See
        # D1_DIFFUSION_KERNEL_CONTACT_RECHECK_DESIGN_20260907.md.
        'diffusion_resolve_same_batch': False,
        # Additive geometric post-contact relaxation (default OFF, unchanged rigid-2R-contact
        # behaviour). See P1_CONTACT_RELAXATION_MODEL_DESIGN_20260823.md.
        'enable_contact_relaxation': False,
        'contact_relaxation_radius_nm': None,      # None -> derived as 0.5*collision_diameter
        'contact_relaxation_num_candidates': 6,
        'contact_relaxation_bond_energy_eV': 0.35, # Breeman1995 isotropic avg NN bond (100)/(111)
        # Coordination-shell cutoff factor k (cutoff_radius = k x collision_diameter). PROVEN
        # (not just asserted) to lie in the geometrically valid first-shell-only window
        # k in [1, sqrt(2)) for an FCC lattice (d_NN=a/sqrt(2), d_NNN=a, a=3.615A Cu lattice
        # constant) -- k=1.15 sits at 36.2% of that window, margins +0.038nm/-0.068nm from the
        # two boundaries. Full derivation, equations, and an honest statement of what is/isn't
        # proven: P1_COORDINATION_SHELL_CUTOFF_DERIVATION_20260824.md. No historical record of
        # why 1.15 specifically (vs. another point in the window) was first chosen -- searched
        # exhaustively (git/design docs/DB), found none; the interval is proven, the specific
        # interior point is a documented convention, not a literature-derived unique value.
        'contact_relaxation_coord_shell_factor': 1.15,
        # Additive diffuse (cosine-law/Lambertian) re-emission for sticking-rejected atoms
        # (default OFF, unchanged discard-only behaviour). See
        # P1_DIFFUSE_REEMISSION_MODEL_DESIGN_20260823.md.
        'enable_diffuse_reemission': False,
        'max_reemission_attempts': 2,
        # Same k=1.15 coordination-shell-cutoff convention as contact_relaxation_coord_shell_factor
        # above -- see that field's comment and P1_COORDINATION_SHELL_CUTOFF_DERIVATION_20260824.md
        # for the full geometric derivation of the valid [1, sqrt(2)) window this sits inside.
        'reemission_normal_shell_factor': 1.15,
        'metrics_interval': 100.0,
        'log_file': 'simulation_v3.log',
        'verbose': True,
        'write_interval': 100,
        'auto_postprocess': False,
        # Native helical dynamics
        'pitch': PITCH,
        'growth_rate': GROWTH_RATE,
        'angular_distribution': angular_defaults.copy(),
        'angular_distribution_enabled': False,
        'angular_distribution_type': 'none',
        'angular_sigma_deg': 0.0,
        'angular_num_samples': 1,
        'angular_seed': 0,
        'random_seed': None,   # None = unseeded (OS entropy); any int (incl. 0) = seeded
        'enable_sticking': False,        # master gate; default OFF == current behaviour
        'sticking_probability': 1.0,     # s in [0,1]; only consulted when enable_sticking=True
        # Additive local-coordination-dependent hop-barrier correction (default OFF,
        # unchanged flat-Ea behaviour). REAL fitted parameters from Mehl, Biham, Furman &
        # Karimi, Phys. Rev. B 60, 2106 (1999) "Models for adatom diffusion on fcc (001)
        # metal surfaces" -- Model II (Table II), a linear best-fit to the FULL 128-config
        # EAM hopping-barrier landscape (Table I) for Cu(001): EB = E0 + dNN*n_NN +
        # dNNN*n_NNN. E0=0.487eV independently cross-validates Boisvert1997's own EAM
        # isolated-adatom value (0.49eV, A4B01) almost exactly -- two independent methods
        # agreeing to within 0.003eV. Distance-shell (NN/NNN) neighbour counting is a
        # coarse-grained proxy in this project's amorphous/ballistic packing (not a rigid
        # fcc lattice like Mehl's own system) -- documented simplification, not a literal
        # reproduction of Mehl's exact 7-site configuration classification. Full PDF
        # absorbed 2026-08-24 (04_LITERATURE_AND_PARAMETER_WORKFLOW/PRIMARY_SOURCE_PDFS/
        # mehl1999.pdf). See P1_CONTACT_RELAXATION_MODEL_DESIGN_20260823.md.
        #
        # THIRD independent cross-check, 2026-08-28 (P4 Phase 1, this project's own
        # calculation, not a literature lookup): a real NEB hop-barrier calculation on
        # an isolated Cu(100) adatom, using the Mishin et al. Cu EAM potential (Phys.
        # Rev. B 63, 224106, 2001), gives 0.5106 eV -- within 0.021-0.024 eV of both
        # Boisvert (0.49) and Mehl (0.487), i.e. three independent methods (two prior
        # literature EAM fits, plus this project's own from-scratch NEB run against a
        # third EAM potential) now agree to ~0.02-0.03 eV. This is a self-consistency
        # confirmation of E0 below, not a recalibration: it does NOT resolve the
        # separate, still-open column-tilt-vs-diffusion discrepancy (see
        # ssec:limits in P1_ballistic_GLAD_scope_limits.tex), and eam_e0_eV is left
        # unchanged rather than silently replaced. Reference-only constant below is
        # not read by any hop calculation. Full derivation, robustness check (NEB
        # method sensitivity), and pre-registered prediction this confirms:
        # 01_GLAD_SIMULATION/P4_EAM_MD_FUTURE_WORK/P4_EAM_MD_MULTISCALE_CALIBRATION_
        # PLAN_20260827.md and results/phase1_hop_barrier_result.json.
        'eam_e0_eV_mishin2001_crosscheck_reference': 0.5106,  # NOT wired into any
        # calculation; provenance/citation record only, per Phase 4's "additive,
        # literature-cited... never replacing the existing default silently" rule.
        'use_eam_neighbor_barrier': False,
        'eam_e0_eV': 0.487,       # Mehl1999 Table II, Cu Model II
        'eam_dNN_eV': 0.274,      # Mehl1999 Table II, Cu Model II (per NN bond)
        'eam_dNNN_eV': 0.027,     # Mehl1999 Table II, Cu Model II (per NNN bond)
        'eam_nn_cutoff_nm': None,   # None -> derived as collision_diameter (real Cu NN dist., 0.256nm)
        'eam_nnn_cutoff_nm': None,  # None -> derived as collision_diameter*1.414 (real Cu NNN dist., ~0.362nm)
        # TWO-TIER GRID (dev, 2026-09-03): registered here and explicitly pulled from
        # raw['performance'] below -- NOT just referenced via a bare cfg.get(...) default
        # at the call site, per this file's own documented `diffusion_dwell_hops_cap`
        # incident (a cfg.get default that silently ignored any YAML override because
        # load_config never actually copied the key out of raw). See
        # GRID_REBUILD_PERFORMANCE_OPTIMIZATION_SCOPED_20260903.md.
        'grid_rebuild_stride': 10,
    }
    if not os.path.exists(yaml_path):
        print(f"[V3] Config file not found ({yaml_path}) -- using defaults", flush=True)
        return defaults

    with open(yaml_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    cfg = defaults.copy()
    if 'simulation' in raw:
        cfg.update({k: raw['simulation'][k] for k in raw['simulation'] if k in cfg})
        cfg['n_columns']     = raw['simulation'].get('n_columns', cfg['n_columns'])
        cfg['target_height'] = raw['simulation'].get('target_height', cfg['target_height'])
        cfg['alpha']         = raw['simulation'].get('alpha', cfg['alpha'])
        cfg['material']      = raw['simulation'].get('material', cfg['material'])
        cfg['substrate_spacing']    = raw['simulation'].get('substrate_spacing', cfg['substrate_spacing'])
        cfg['enable_sticking']      = raw['simulation'].get('enable_sticking', cfg['enable_sticking'])
        cfg['sticking_probability'] = float(raw['simulation'].get('sticking_probability', cfg['sticking_probability']))
        cfg['enable_diffuse_reemission'] = bool(raw['simulation'].get('enable_diffuse_reemission', cfg['enable_diffuse_reemission']))
        cfg['max_reemission_attempts'] = int(raw['simulation'].get('max_reemission_attempts', cfg['max_reemission_attempts']))

    if 'performance' in raw:
        cfg['use_gpu'] = raw['performance'].get('use_gpu', False)
        cfg['batch_size'] = raw['performance'].get('batch_size', 512)
        cfg['cell_size'] = raw['performance'].get('cell_size', cfg['cell_size'])
        cfg['kdtree_rebuild_interval'] = raw['performance'].get('kdtree_rebuild_interval', 1000000)
        cfg['grid_rebuild_stride'] = raw['performance'].get('grid_rebuild_stride', cfg['grid_rebuild_stride'])
    else:
        cfg['use_gpu'] = False
        cfg['batch_size'] = 512

    if 'box' in raw:
        cfg['box_width'] = raw['box'].get('width', cfg['box_width'])
        cfg['box_depth'] = raw['box'].get('depth', cfg['box_depth'])
    if 'deposition' in raw:
        cfg['r_dep']        = raw['deposition'].get('rate', cfg['r_dep'])
        cfg['growth_efficiency'] = raw['deposition'].get('growth_efficiency', cfg['growth_efficiency'])
        cfg['temperature']  = raw['deposition'].get('temperature', cfg['temperature'])
        cfg['melting_point_K'] = raw['deposition'].get('melting_point_K', cfg['melting_point_K'])
        cfg['source_distance_cm'] = raw['deposition'].get('source_distance_cm', cfg['source_distance_cm'])
        cfg['source_radius_cm'] = raw['deposition'].get('source_radius_cm', cfg['source_radius_cm'])
    if 'rotation' in raw:
        cfg['rotation_rpm'] = raw['rotation'].get('rpm', cfg['rotation_rpm'])
    if 'checkpoint' in raw:
        cfg['checkpoint_height_interval'] = raw['checkpoint'].get('height_interval', cfg['checkpoint_height_interval'])
        cfg['checkpoint_dir']             = raw['checkpoint'].get('output_dir', cfg['checkpoint_dir'])
        cfg['allow_legacy_resume']        = raw['checkpoint'].get('allow_legacy_resume', cfg['allow_legacy_resume'])
        cfg['checkpoint_particle_interval'] = raw['checkpoint'].get('particle_interval', cfg['checkpoint_particle_interval'])
        # Fix: store time_interval at top level so run_simulation can read it directly
        cfg['checkpoint_time_interval']   = raw['checkpoint'].get('time_interval', 600.0)
    if 'physics' in raw:
        cfg['enable_surface_diffusion'] = raw['physics'].get('enable_surface_diffusion', cfg['enable_surface_diffusion'])
        cfg['diffusion_radius']         = raw['physics'].get('diffusion_radius', 5.0) / 10.0  # Å → nm
        cfg['diffusion_hops']           = raw['physics'].get('diffusion_hops', cfg['diffusion_hops'])
        cfg['diffusion_dwell_hops_cap'] = int(raw['physics'].get('diffusion_dwell_hops_cap', cfg['diffusion_dwell_hops_cap']))
        cfg['activation_energy']        = raw['physics'].get('activation_energy', cfg['activation_energy'])
        cfg['use_arrhenius']            = raw['physics'].get('use_arrhenius', cfg['use_arrhenius'])
        cfg['use_physical_diffusion_model'] = bool(raw['physics'].get('use_physical_diffusion_model', cfg['use_physical_diffusion_model']))
        cfg['nu0_hz']                    = float(raw['physics'].get('nu0_hz', cfg['nu0_hz']))
        cfg['diffusion_resolve_same_batch'] = bool(raw['physics'].get('diffusion_resolve_same_batch', cfg['diffusion_resolve_same_batch']))
        cfg['enable_contact_relaxation'] = bool(raw['physics'].get('enable_contact_relaxation', cfg['enable_contact_relaxation']))
        cfg['contact_relaxation_radius_nm'] = raw['physics'].get('contact_relaxation_radius_nm', cfg['contact_relaxation_radius_nm'])
        cfg['contact_relaxation_num_candidates'] = int(raw['physics'].get('contact_relaxation_num_candidates', cfg['contact_relaxation_num_candidates']))
        cfg['contact_relaxation_bond_energy_eV'] = float(raw['physics'].get('contact_relaxation_bond_energy_eV', cfg['contact_relaxation_bond_energy_eV']))
        cfg['contact_relaxation_coord_shell_factor'] = float(raw['physics'].get('contact_relaxation_coord_shell_factor', cfg['contact_relaxation_coord_shell_factor']))
        cfg['reemission_normal_shell_factor'] = float(raw['physics'].get('reemission_normal_shell_factor', cfg['reemission_normal_shell_factor']))
        cfg['use_eam_neighbor_barrier']  = bool(raw['physics'].get('use_eam_neighbor_barrier', cfg['use_eam_neighbor_barrier']))
        cfg['eam_e0_eV']                = float(raw['physics'].get('eam_e0_eV', cfg['eam_e0_eV']))
        cfg['eam_dNN_eV']               = float(raw['physics'].get('eam_dNN_eV', cfg['eam_dNN_eV']))
        cfg['eam_dNNN_eV']              = float(raw['physics'].get('eam_dNNN_eV', cfg['eam_dNNN_eV']))
        cfg['eam_nn_cutoff_nm']         = raw['physics'].get('eam_nn_cutoff_nm', cfg['eam_nn_cutoff_nm'])
        cfg['eam_nnn_cutoff_nm']        = raw['physics'].get('eam_nnn_cutoff_nm', cfg['eam_nnn_cutoff_nm'])
        cfg['pitch']                    = float(raw['physics'].get('pitch', PITCH))
        cfg['growth_rate']              = float(raw['physics'].get('growth_rate', GROWTH_RATE))
        cfg['active_window_height']     = raw['physics'].get('active_window_height', cfg['active_window_height'])
        cfg['resume_active_window_height'] = raw['physics'].get('resume_active_window_height', cfg['resume_active_window_height'])
        cfg['thermal_throttle_temp']    = raw['physics'].get('thermal_throttle_temp', cfg['thermal_throttle_temp'])
        cfg['thermal_critical_temp']    = raw['physics'].get('thermal_critical_temp', cfg['thermal_critical_temp'])
        if 'author_radius_nm' in raw['physics']:
            cfg['author_radius_nm'] = float(raw['physics']['author_radius_nm'])
        angular_raw = dict(raw['physics'].get('angular_distribution', {}) or {})
        # Physically-derived angular_sigma_deg default (source aperture / distance),
        # used only when the config doesn't already explicitly set angular_sigma_deg.
        if 'angular_sigma_deg' not in angular_raw and cfg['source_radius_cm'] is not None \
                and cfg['source_distance_cm'] is not None:
            derived_sigma = _derive_angular_sigma_deg(
                float(cfg['source_radius_cm']), float(cfg['source_distance_cm']))
            if derived_sigma > 10.0:
                print(f"[WARN] Physically-derived angular_sigma_deg={derived_sigma:.3f} deg "
                      f"(source_radius_cm={cfg['source_radius_cm']}, "
                      f"source_distance_cm={cfg['source_distance_cm']}) exceeds the code's "
                      f"validated cap of 10.0 deg (A1 dry-run patch) -- clamping to 10.0.",
                      flush=True)
                derived_sigma = 10.0
            angular_raw['angular_sigma_deg'] = derived_sigma
            angular_raw.setdefault('_angular_sigma_source', 'derived_from_source_geometry')
        cfg['angular_distribution'] = _validate_angular_distribution_config(angular_raw, angular_defaults)
        cfg['angular_distribution_enabled'] = bool(cfg['angular_distribution']['enabled'])
        cfg['angular_distribution_type'] = (
            cfg['angular_distribution']['type'] if cfg['angular_distribution_enabled'] else 'none'
        )
        cfg['angular_sigma_deg'] = float(cfg['angular_distribution']['angular_sigma_deg'])
        cfg['angular_num_samples'] = int(cfg['angular_distribution']['angular_num_samples'])
        cfg['angular_seed'] = int(cfg['angular_distribution']['angular_seed'])
    if 'monitoring' in raw:
        cfg['metrics_interval'] = raw['monitoring'].get('metrics_interval', cfg['metrics_interval'])
        cfg['log_file']         = raw['monitoring'].get('log_file', 'simulation_v3.log')
        cfg['verbose']          = raw['monitoring'].get('verbose', cfg['verbose'])
        cfg['write_interval']   = raw['monitoring'].get('write_interval', cfg['write_interval'])
        cfg['auto_postprocess'] = raw['monitoring'].get('auto_postprocess', cfg['auto_postprocess'])
    if 'postprocess' in raw:
        cfg['auto_postprocess'] = raw['postprocess'].get('enabled', cfg['auto_postprocess'])

    # Resolve runtime outputs relative to the simulation root, not the current
    # working directory. This keeps checkpoint/log files stable when launched
    # from simulation MD/src through the orchestrator.
    sim_root = os.path.dirname(os.path.dirname(os.path.abspath(yaml_path)))
    if not os.path.isabs(cfg['checkpoint_dir']):
        cfg['checkpoint_dir'] = os.path.normpath(os.path.join(sim_root, cfg['checkpoint_dir']))
    if not os.path.isabs(cfg['log_file']):
        cfg['log_file'] = os.path.normpath(os.path.join(sim_root, cfg['log_file']))

    material = str(cfg.get('material', 'Cu'))
    if cfg.get('melting_point_K') is None:
        cfg['melting_point_K'] = MATERIAL_MELTING_POINTS_K.get(material, MATERIAL_MELTING_POINTS_K['Cu'])
    cfg['homologous_temperature'] = float(cfg['temperature']) / float(cfg['melting_point_K'])

    nominal_rate = float(cfg['r_dep'])
    growth_efficiency = float(cfg.get('growth_efficiency', 1.0))
    cfg['growth_efficiency'] = growth_efficiency
    cfg['vertical_growth_rate'] = nominal_rate * growth_efficiency

    # Experimental-control convention:
    # If rotation.rpm is provided, the helix pitch is derived from the lab knobs:
    #     pitch_nm_per_turn = vertical_growth_rate_nm_s * 60 / rotation_rpm
    # where vertical_growth_rate = nominal_rate * growth_efficiency.  If the
    # configured rate is already the measured vertical growth rate, keep
    # growth_efficiency=1.0.
    # This keeps the simulation height capped independently from the number of
    # helix turns and makes rpm/deposition-rate sweeps PINN-ready.
    rpm = cfg.get('rotation_rpm')
    if rpm is not None:
        rpm = float(rpm)
        cfg['rotation_rpm'] = rpm
        cfg['growth_rate'] = float(cfg['vertical_growth_rate'])
        if rpm > 0:
            cfg['pitch'] = float(cfg['vertical_growth_rate']) * 60.0 / rpm
        else:
            cfg['pitch'] = 1.0e12  # effectively no rotation / straight GLAD

    cfg.setdefault('author_radius_nm', float(MATERIAL_ATOM_RADIUS_NM.get(material, AUTHOR_RADIUS)))
    cfg['_source_yaml'] = os.path.abspath(yaml_path)
    return cfg


def _derive_angular_sigma_deg(source_radius_cm: float, source_distance_cm: float) -> float:
    """
    Physically-derived flux angular half-spread for a thermal-evaporation point/area
    source of radius `source_radius_cm` at distance `source_distance_cm`:

        sigma_deg = degrees(atan(source_radius_cm / source_distance_cm))

    Larger source, or closer distance, -> larger (less collimated) spread. This
    replaces an arbitrary angular_sigma_deg guess with a real geometric quantity when
    the source aperture size is known (or assumed and stated as such).
    """
    if source_distance_cm <= 0:
        return 0.0
    return float(np.degrees(np.arctan(source_radius_cm / source_distance_cm)))


def _validate_angular_distribution_config(raw_value, defaults: dict) -> dict:
    """Validate optional angular-distribution settings without changing legacy defaults."""
    if raw_value in (None, False):
        raw = {}
    elif isinstance(raw_value, dict):
        raw = raw_value
    else:
        raise ValueError("physics.angular_distribution must be a mapping when provided")

    cfg = defaults.copy()
    cfg.update(raw)
    cfg['enabled'] = bool(cfg.get('enabled', False))
    cfg['type'] = str(cfg.get('type', 'deterministic_cone_ring'))
    cfg['scheme'] = str(cfg.get('scheme', 'ring'))
    cfg['weights'] = str(cfg.get('weights', 'equal'))
    cfg['rotate_with_helical_azimuth'] = bool(cfg.get('rotate_with_helical_azimuth', True))
    cfg['preserve_nominal_mean_direction'] = bool(cfg.get('preserve_nominal_mean_direction', True))
    cfg['angular_sigma_deg'] = float(cfg.get('angular_sigma_deg', 0.0))
    cfg['angular_num_samples'] = int(cfg.get('angular_num_samples', 1))
    cfg['angular_seed'] = int(cfg.get('angular_seed', 0))

    if not cfg['enabled']:
        cfg['type'] = 'none'
        cfg['angular_sigma_deg'] = 0.0
        cfg['angular_num_samples'] = 1
        return cfg

    if cfg['type'] != 'deterministic_cone_ring':
        raise ValueError("physics.angular_distribution.type must be deterministic_cone_ring")
    if cfg['scheme'] != 'ring':
        raise ValueError("physics.angular_distribution.scheme must be ring")
    if cfg['weights'] != 'equal':
        raise ValueError("physics.angular_distribution.weights must be equal")
    if cfg['angular_sigma_deg'] < 0.0:
        raise ValueError("physics.angular_distribution.angular_sigma_deg must be non-negative")
    if cfg['angular_sigma_deg'] > 10.0:
        raise ValueError("physics.angular_distribution.angular_sigma_deg must be <= 10.0 for A1 dry-run patch")
    if cfg['angular_num_samples'] < 1:
        raise ValueError("physics.angular_distribution.angular_num_samples must be >= 1")
    return cfg


# ---------------------------------------------------------------------------
# Custom CUDA Kernel: Ray-Grid Traversal (DDA) + Sphere Intersection
# ---------------------------------------------------------------------------

RAY_SPHERE_KERNEL_SOURCE = r'''
extern "C" __global__
void RAY_SPHERE_KERNEL(
    const float*  origins,      // [n_rays, 3]
    const float   dir_x,
    const float   dir_y,
    const float   dir_z,
    const float*  positions,    // [n_atoms, 3]
    const int*    cell_starts,   // [nx * ny * nz + 1]
    const int*    sorted_indices,// [n_atoms]
    const int nx, const int ny, const int nz,
    const float gm_x, const float gm_y, const float gm_z,
    const float   cell_size,
    const int     n_rays,
    const int     n_atoms,
    const float   radius,
    float*        t_results,     // [n_rays]
    const int     n_stable,      // TWO-TIER GRID (2026-09-03, dev-only, not yet live):
    const int     n_active_pending // positions[n_stable:n_active_pending] are pending atoms
                                     // not yet in the grid -- checked by brute force below.
                                     // When n_stable == n_active_pending (no pending atoms),
                                     // this loop is a no-op and behaviour is IDENTICAL to the
                                     // original single-tier kernel.
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rays) return;

    int3 grid_dims = {nx, ny, nz};
    float3 grid_min = {gm_x, gm_y, gm_z};
    float3 O = {
        origins[3 * idx + 0],
        origins[3 * idx + 1],
        origins[3 * idx + 2]
    };
    float3 D = {dir_x, dir_y, dir_z};
    float r2 = radius * radius;

    float t_min = 1e20f;
    bool hit = false;

    int3 current_voxel;
    current_voxel.x = floorf((O.x - grid_min.x) / cell_size);
    current_voxel.y = floorf((O.y - grid_min.y) / cell_size);
    current_voxel.z = floorf((O.z - grid_min.z) / cell_size);

    int step_x = (D.x > 0) ? 1 : (D.x < 0 ? -1 : 0);
    int step_y = (D.y > 0) ? 1 : (D.y < 0 ? -1 : 0);
    int step_z = (D.z > 0) ? 1 : (D.z < 0 ? -1 : 0);

    float tDeltaX = (D.x != 0) ? fabsf(cell_size / D.x) : 1e20f;
    float tDeltaY = (D.y != 0) ? fabsf(cell_size / D.y) : 1e20f;
    float tDeltaZ = (D.z != 0) ? fabsf(cell_size / D.z) : 1e20f;

    float tMaxX = (D.x > 0) ? (floorf((O.x - grid_min.x) / cell_size) + 1.0f) * cell_size + grid_min.x - O.x :
                             (floorf((O.x - grid_min.x) / cell_size))        * cell_size + grid_min.x - O.x;
    float tMaxY = (D.y > 0) ? (floorf((O.y - grid_min.y) / cell_size) + 1.0f) * cell_size + grid_min.y - O.y :
                             (floorf((O.y - grid_min.y) / cell_size))        * cell_size + grid_min.y - O.y;
    float tMaxZ = (D.z > 0) ? (floorf((O.z - grid_min.z) / cell_size) + 1.0f) * cell_size + grid_min.z - O.z :
                             (floorf((O.z - grid_min.z) / cell_size))        * cell_size + grid_min.z - O.z;
    tMaxX = (D.x != 0) ? tMaxX / D.x : 1e20f;
    tMaxY = (D.y != 0) ? tMaxY / D.y : 1e20f;
    tMaxZ = (D.z != 0) ? tMaxZ / D.z : 1e20f;

    // Box dimensions for PBC minimum-image correction
    float box_w = grid_dims.x * cell_size;
    float box_d = grid_dims.y * cell_size;

    for (int step = 0; step < 10000; step++) {
        if (current_voxel.z < -1) break;
        int gx = (current_voxel.x % grid_dims.x + grid_dims.x) % grid_dims.x;
        int gy = (current_voxel.y % grid_dims.y + grid_dims.y) % grid_dims.y;
        int gz = current_voxel.z;

        // Virtual (unwrapped) ray position at the current voxel centre
        float vx = grid_min.x + (current_voxel.x + 0.5f) * cell_size;
        float vy = grid_min.y + (current_voxel.y + 0.5f) * cell_size;

        if (gz >= 0 && gz < grid_dims.z) {
            int cell_idx = (gx * grid_dims.y + gy) * grid_dims.z + gz;
            int start = cell_starts[cell_idx];
            int end   = cell_starts[cell_idx + 1];
            for (int k = start; k < end; k++) {
                int atom_idx = sorted_indices[k];
                float3 C = {
                    positions[3 * atom_idx + 0],
                    positions[3 * atom_idx + 1],
                    positions[3 * atom_idx + 2]
                };

                // Minimum-image correction: shift C to the periodic image
                // nearest the ray's current virtual (unwrapped) position
                float dx = vx - C.x;
                float dy = vy - C.y;
                C.x += roundf(dx / box_w) * box_w;
                C.y += roundf(dy / box_d) * box_d;

                float3 L = {O.x - C.x, O.y - C.y, O.z - C.z};
                float b = 2.0f * (D.x * L.x + D.y * L.y + D.z * L.z);
                float c = (L.x * L.x + L.y * L.y + L.z * L.z) - r2;
                float disc = b * b - 4.0f * c;
                if (disc >= 0) {
                    float sqrt_d = sqrtf(disc);
                    float t = (-b - sqrt_d) / 2.0f;
                    if (t > 1e-4f && t < t_min) {
                        t_min = t;
                        hit = true;
                    }
                }
            }
        }
        if (hit && t_min <= fminf(fminf(tMaxX, tMaxY), tMaxZ)) break;
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) { current_voxel.x += step_x; tMaxX += tDeltaX; }
            else               { current_voxel.z += step_z; tMaxZ += tDeltaZ; }
        } else {
            if (tMaxY < tMaxZ) { current_voxel.y += step_y; tMaxY += tDeltaY; }
            else               { current_voxel.z += step_z; tMaxZ += tDeltaZ; }
        }
    }

    // TWO-TIER GRID (2026-09-03, dev-only, not yet live): brute-force check against
    // pending atoms not yet covered by the grid (positions[n_stable:n_active_pending]).
    //
    // CORRECTNESS NOTE (found and fixed during implementation, not shipped with the
    // bug): the grid-cell loop above uses `vx`/`vy` (the DDA traversal's CURRENT
    // voxel-centre position) as the minimum-image reference point, not the ray
    // origin `O` -- correct there because it is testing many different cells along
    // a potentially long ray path, and wants the periodic image nearest to the
    // SPECIFIC cell currently being tested. This brute-force loop has no equivalent
    // "current position along the ray" (it is not iterating cell-by-cell), so a
    // single fixed reference point (e.g. the ray origin) is NOT a valid substitute --
    // an early draft of this code used `O` here and was wrong for exactly this
    // reason on grazing rays that travel far laterally before a possible hit.
    // Fixed to test all nine periodic images of C explicitly (dx,dy in {-1,0,1}) and
    // keep the best (smallest valid t) intersection across all nine -- this is
    // unambiguous and does not depend on any single "current position" choice. Same
    // nearest-image-only assumption (no image beyond +-1 box width) as every other
    // periodic lookup in this file (the grid's own (gx%nx+nx)%nx wraparound is
    // likewise a 1-cell-radius convention), so this introduces no new assumption.
    for (int p = n_stable; p < n_active_pending; p++) {
        float3 C0 = {
            positions[3 * p + 0],
            positions[3 * p + 1],
            positions[3 * p + 2]
        };
        for (int ix = -1; ix <= 1; ix++) {
            for (int iy = -1; iy <= 1; iy++) {
                float3 C = { C0.x + ix * box_w, C0.y + iy * box_d, C0.z };
                float3 L = {O.x - C.x, O.y - C.y, O.z - C.z};
                float b = 2.0f * (D.x * L.x + D.y * L.y + D.z * L.z);
                float c = (L.x * L.x + L.y * L.y + L.z * L.z) - r2;
                float disc = b * b - 4.0f * c;
                if (disc >= 0) {
                    float sqrt_d = sqrtf(disc);
                    float t = (-b - sqrt_d) / 2.0f;
                    if (t > 1e-4f && t < t_min) {
                        t_min = t;
                        hit = true;
                    }
                }
            }
        }
    }

    t_results[idx] = hit ? t_min : 1e20f;
}

extern "C" __global__
void DIFFUSION_KERNEL(
    float3*       new_atoms,
    const float3* all_positions,
    const int*    cell_starts,
    const int*    sorted_indices,
    const int nx, const int ny, const int nz,
    const float gm_x, const float gm_y, const float gm_z,
    const float   cell_size,
    const float   diff_radius,
    const int     hops,
    const float   hop_prob,
    const float   kB_T,
    const float   r_nm,
    const float   collision_diameter,
    const int     n_rays,
    const float   bw,
    const float   bd,
    unsigned int  seed,
    int*          debug_out,   // TEMP DIAGNOSTIC: [n_rays], packs a coarse fault code per thread
    float*        debug_vals,  // TEMP DIAGNOSTIC: [n_rays*6] = pos.x,pos.y,r,angle,tx,ty for the first faulting hop
    const int     use_neighbor_barrier,  // Mehl1999 Model-II EAM local barrier (opt-in, default 0)
    const float   eam_e0,                // eV, isolated-adatom barrier (Mehl1999 Table II)
    const float   eam_dNN,               // eV per NN bond
    const float   eam_dNNN,              // eV per NNN bond
    const float   nn_cutoff,             // nm, NN-shell radius
    const float   nnn_cutoff,            // nm, NNN-shell outer radius
    const int     n_stable,              // TWO-TIER GRID (dev, 2026-09-03): atoms below this index are covered by cell_starts/sorted_indices
    const int     n_active_pending       // TWO-TIER GRID: atoms [n_stable, n_active_pending) are NOT in the grid yet, checked by brute force below
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rays) return;
    debug_out[idx] = 0;

    int3 grid_dims = {nx, ny, nz};
    float3 grid_min = {gm_x, gm_y, gm_z};
    float2 box_size = {bw, bd};

    float3 pos = new_atoms[idx];
    if (pos.z > 1e19f) return;   // safety: numerical-overflow guard (kernel receives only S_active via mask_active)
                                  // Domain restriction enforced at Python level: new_pos[mask_active==1] → S_active.
                                  // See CONFIRMED_CORRECTIONS_GLAD_V3_CORE_20260626.md §C2 Option C.

    unsigned int state = seed + idx;
    auto next_rand = [&]() {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        return (float)state / (float)0xffffffff;
    };

    // --- Surface-adatom mobility gate (2026-09-07 diffusion-model fix) ---
    // Only genuinely under-coordinated surface atoms diffuse. Without this every
    // active-window atom hops, and because a hop settles onto the local z-floor
    // (contact height on the tallest laterally-reachable neighbour) each accepted
    // hop is a *climb*: interior atoms ratchet toward the column apex, the film
    // gains height with a fraction of the mass, connectivity collapses
    // (P1_DIFFUSION_HOPBUDGET_SWEEP_20260907: cap=1 -> 15 % of the ballistic atom
    // count, ISOLATED_COLUMNS vs ballistic CONNECTED_FILM). Coordination is counted
    // in the same 3x3 grid columns used for the z-floor below, around the atom's
    // CURRENT position, within a 1.15*collision_diameter first shell. An atom with
    // coord >= MOBILE_COORD_MAX is embedded -> immobile (write back, done).
    {
        const int   MOBILE_COORD_MAX = 10;
        const float mob_shell2 = (1.15f * collision_diameter) * (1.15f * collision_diameter);
        // positions are stored UNWRAPPED (periodic); wrap into [0,box) for the grid
        // index exactly as the z-floor loop does with tx/ty, else a large unwrapped
        // pos.x makes the (g + d%N + N)%N formula produce a negative index ->
        // cudaErrorIllegalAddress. Neighbour DISTANCES below still use minimum-image
        // against the raw (unwrapped) coordinates, which is correct.
        float mwpx = fmodf(pos.x, box_size.x); if (mwpx < 0.0f) mwpx += box_size.x;
        float mwpy = fmodf(pos.y, box_size.y); if (mwpy < 0.0f) mwpy += box_size.y;
        int mgx0 = (int)floorf((mwpx - grid_min.x) / cell_size);
        int mgy0 = (int)floorf((mwpy - grid_min.y) / cell_size);
        mgx0 = (mgx0 % grid_dims.x + grid_dims.x) % grid_dims.x;
        mgy0 = (mgy0 % grid_dims.y + grid_dims.y) % grid_dims.y;
        int coord0 = 0;
        for (int mdx = -1; mdx <= 1 && coord0 < MOBILE_COORD_MAX; mdx++) {
            for (int mdy = -1; mdy <= 1 && coord0 < MOBILE_COORD_MAX; mdy++) {
                int mgx = (mgx0 + mdx % grid_dims.x + grid_dims.x) % grid_dims.x;
                int mgy = (mgy0 + mdy % grid_dims.y + grid_dims.y) % grid_dims.y;
                for (int mgz = grid_dims.z - 1; mgz >= 0; mgz--) {
                    int m_idx = (mgx * grid_dims.y + mgy) * grid_dims.z + mgz;
                    int ms = cell_starts[m_idx];
                    int me = cell_starts[m_idx + 1];
                    for (int mk = ms; mk < me; mk++) {
                        float3 q = all_positions[sorted_indices[mk]];
                        float qdx = pos.x - q.x; qdx += roundf(-qdx / box_size.x) * box_size.x;
                        float qdy = pos.y - q.y; qdy += roundf(-qdy / box_size.y) * box_size.y;
                        float qdz = pos.z - q.z;
                        float md2 = qdx * qdx + qdy * qdy + qdz * qdz;
                        if (md2 > 1e-6f && md2 < mob_shell2) coord0++;
                    }
                }
            }
        }
        for (int pidx = n_stable; pidx < n_active_pending && coord0 < MOBILE_COORD_MAX; pidx++) {
            float3 q = all_positions[pidx];
            float qdx = pos.x - q.x; qdx += roundf(-qdx / box_size.x) * box_size.x;
            float qdy = pos.y - q.y; qdy += roundf(-qdy / box_size.y) * box_size.y;
            float qdz = pos.z - q.z;
            float md2 = qdx * qdx + qdy * qdy + qdz * qdz;
            if (md2 > 1e-6f && md2 < mob_shell2) coord0++;
        }
        if (coord0 >= MOBILE_COORD_MAX) { new_atoms[idx] = pos; return; }
    }

    for (int h = 0; h < hops; h++) {
        if (next_rand() > hop_prob) continue;
        debug_out[idx] = 9;  // TEMP DIAGNOSTIC: marks "hop body entered at least once", overwritten below if a fault is found

        // Mehl1999 Model-II EAM local-neighbour barrier (opt-in, use_neighbor_barrier!=0):
        // count real 3D neighbours within two distance shells (NN, NNN) around the atom's
        // CURRENT site (pos, before choosing a candidate target), form the local barrier
        // EB_local = eam_e0 + eam_dNN*n_NN + eam_dNNN*n_NNN (Mehl, Biham, Furman & Karimi,
        // Phys. Rev. B 60, 2106 (1999), Table II, Cu Model II), and apply an EXTRA
        // Boltzmann suppression factor relative to eam_e0 (already baked into the
        // passed-in hop_prob/kB_T, which was calibrated against an isolated-adatom Ea).
        // Only ever suppresses further (local barrier >= eam_e0 for any real neighbour
        // count), never boosts a hop above the base attempt rate — consistent with
        // Mehl1999's own finding that added neighbours always raise the barrier
        // (dNN, dNNN > 0 for Cu). Distance-shell counting is a coarse-grained proxy for
        // Mehl's exact 7-site fcc(001) classification, appropriate here since this
        // project's packing is amorphous/ballistic, not a rigid lattice.
        if (use_neighbor_barrier != 0) {
            int gx0 = (int)floorf((pos.x - grid_min.x) / cell_size);
            int gy0 = (int)floorf((pos.y - grid_min.y) / cell_size);
            int n_nn = 0;
            int n_nnn = 0;
            for (int ndx = -1; ndx <= 1; ndx++) {
                for (int ndy = -1; ndy <= 1; ndy++) {
                    int ngx = (gx0 + ndx % grid_dims.x + grid_dims.x) % grid_dims.x;
                    int ngy = (gy0 + ndy % grid_dims.y + grid_dims.y) % grid_dims.y;
                    for (int ngz = grid_dims.z - 1; ngz >= 0; ngz--) {
                        int n_idx = (ngx * grid_dims.y + ngy) * grid_dims.z + ngz;
                        int nstart = cell_starts[n_idx];
                        int nend = cell_starts[n_idx + 1];
                        for (int nk = nstart; nk < nend; nk++) {
                            float3 q = all_positions[sorted_indices[nk]];
                            float qdx = pos.x - q.x;
                            float qdy = pos.y - q.y;
                            qdx += roundf(-qdx / box_size.x) * box_size.x;
                            qdy += roundf(-qdy / box_size.y) * box_size.y;
                            float qdz = pos.z - q.z;
                            float d2 = qdx * qdx + qdy * qdy + qdz * qdz;
                            if (d2 > 1e-6f && d2 < nn_cutoff * nn_cutoff) n_nn++;
                            else if (d2 < nnn_cutoff * nnn_cutoff) n_nnn++;
                        }
                    }
                }
            }
            // TWO-TIER GRID (dev, 2026-09-03): pending atoms [n_stable, n_active_pending)
            // are not yet in cell_starts/sorted_indices -- brute-force them too, using the
            // same fixed query point (pos) the grid-cell loop above already uses for its own
            // minimum-image correction (safe here, unlike RAY_SPHERE_KERNEL's traversal loop,
            // because `pos` does not change during this scan -- there is no "current position
            // along a ray" ambiguity for a single fixed query point).
            for (int pidx = n_stable; pidx < n_active_pending; pidx++) {
                float3 q = all_positions[pidx];
                float qdx = pos.x - q.x;
                float qdy = pos.y - q.y;
                qdx += roundf(-qdx / box_size.x) * box_size.x;
                qdy += roundf(-qdy / box_size.y) * box_size.y;
                float qdz = pos.z - q.z;
                float d2 = qdx * qdx + qdy * qdy + qdz * qdz;
                if (d2 > 1e-6f && d2 < nn_cutoff * nn_cutoff) n_nn++;
                else if (d2 < nnn_cutoff * nnn_cutoff) n_nnn++;
            }
            float ea_local = eam_e0 + eam_dNN * (float)n_nn + eam_dNNN * (float)n_nnn;
            float extra_prob = expf(-fmaxf(0.0f, ea_local - eam_e0) / kB_T);
            if (next_rand() > extra_prob) continue;
        }

        float angle = next_rand() * 6.2831853f;
        float r = next_rand() * diff_radius;
        float tx = fmodf(pos.x + r * cosf(angle), box_size.x);
        if (tx < 0) tx += box_size.x;
        float ty = fmodf(pos.y + r * sinf(angle), box_size.y);
        if (ty < 0) ty += box_size.y;

        debug_vals[idx*6+0] = pos.x; debug_vals[idx*6+1] = pos.y;
        debug_vals[idx*6+2] = r;     debug_vals[idx*6+3] = angle;
        debug_vals[idx*6+4] = tx;    debug_vals[idx*6+5] = ty;

        float tz = r_nm;   // floor = r (was hardcoded 0.1f for legacy r=0.10 nm)
        int gx = floorf((tx - grid_min.x) / cell_size);
        int gy = floorf((ty - grid_min.y) / cell_size);

        // TEMP DIAGNOSTIC: flag if gx/gy land outside the expected single-step-wrappable
        // range [0, grid_dims) BEFORE the (gx+dx%N+N)%N formula runs -- that formula only
        // correctly wraps a +-1 step, not a gx/gy that is already far out of range.
        if (gx < 0 || gx >= grid_dims.x) debug_out[idx] = 1;
        if (gy < 0 || gy >= grid_dims.y) debug_out[idx] = 2;
        if (isnan(tx) || isnan(ty) || isinf(tx) || isinf(ty)) debug_out[idx] = 3;
        if (isnan(pos.x) || isnan(pos.y) || isnan(pos.z)) debug_out[idx] = 4;
        if (isinf(pos.x) || isinf(pos.y) || isinf(pos.z)) debug_out[idx] = 5;
        if (fabsf(pos.x) > 1e15f || fabsf(pos.y) > 1e15f) debug_out[idx] = 6;  // huge-but-finite check

        // UNVERIFIED CANDIDATE DIAGNOSTIC (2026-08-23, not yet GPU-tested): this loop used
        // to declare its wrapped-cell-index locals as `nx`/`ny`/`nz`, shadowing the outer
        // kernel parameters of the SAME name (grid dimensions, declared in the
        // DIFFUSION_KERNEL signature above). CONTACT_RELAXATION_KERNEL's equivalent loop
        // already avoids this by using distinct names (`gxx`/`gyy`/`gz`). Renamed here to
        // `cgx`/`cgy`/`cgz` (purely a rename, no logic change) as the first thing to test
        // against the dwell-time-fix crash (NaN + z~-3.7e19 at batch 0, box=100nm/alpha=85,
        // hops_batch>1) -- suspected but NOT CONFIRMED as the actual cause; hold as a
        // hypothesis until re-tested on GPU. See
        // P1_CONTACT_RELAXATION_MODEL_DESIGN_20260823.md `S8_DIFFUSION_DWELL_TIME_FIX`.
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int cgx = (gx + dx % grid_dims.x + grid_dims.x) % grid_dims.x;
                int cgy = (gy + dy % grid_dims.y + grid_dims.y) % grid_dims.y;
                for (int cgz = grid_dims.z - 1; cgz >= 0; cgz--) {
                    int c_idx = (cgx * grid_dims.y + cgy) * grid_dims.z + cgz;
                    int start = cell_starts[c_idx];
                    int end = cell_starts[c_idx + 1];
                    if (start < end) {
                        for (int k = start; k < end; k++) {
                            float3 p_atom = all_positions[sorted_indices[k]];
                            // Minimum-image correction: shift the neighbour to the periodic
                            // image nearest the candidate hop site (tx,ty) -- same convention
                            // as RAY_SPHERE_KERNEL's minimum-image handling above.
                            float ndx = tx - p_atom.x;
                            float ndy = ty - p_atom.y;
                            p_atom.x += roundf(ndx / box_size.x) * box_size.x;
                            p_atom.y += roundf(ndy / box_size.y) * box_size.y;
                            ndx = tx - p_atom.x;
                            ndy = ty - p_atom.y;
                            float lateral2 = ndx * ndx + ndy * ndy;
                            // True lateral-distance-filtered floor: a neighbour can only raise
                            // the floor if it's within lateral reach of the candidate site, and
                            // only by the sphere-sphere touching height at that lateral offset
                            // (sqrt(collision_diameter^2 - lateral^2)) -- NOT unconditionally by
                            // the full collision_diameter regardless of how far away it sits
                            // laterally (old bug: every neighbour anywhere in the 3x3-cell scan
                            // raised the floor by the full diameter, ratcheting atoms upward on
                            // ~all hops once real hop-body execution frequency rose after the
                            // dwell-time fix).
                            if (lateral2 < collision_diameter * collision_diameter) {
                                float z_floor = p_atom.z + sqrtf(collision_diameter * collision_diameter - lateral2);
                                if (z_floor > tz) tz = z_floor;
                            }
                        }
                    }
                }
            }
        }

        // TWO-TIER GRID (dev, 2026-09-03): pending atoms [n_stable, n_active_pending) are
        // not yet in cell_starts/sorted_indices -- brute-force the same lateral-distance-
        // filtered floor check against them, using (tx,ty) as the fixed query point (same
        // reasoning as the neighbour-barrier loop above: tx,ty does not change during this
        // scan, so a single minimum-image correction against it is exact).
        for (int pidx = n_stable; pidx < n_active_pending; pidx++) {
            float3 p_atom = all_positions[pidx];
            float ndx = tx - p_atom.x;
            float ndy = ty - p_atom.y;
            p_atom.x += roundf(ndx / box_size.x) * box_size.x;
            p_atom.y += roundf(ndy / box_size.y) * box_size.y;
            ndx = tx - p_atom.x;
            ndy = ty - p_atom.y;
            float lateral2 = ndx * ndx + ndy * ndy;
            if (lateral2 < collision_diameter * collision_diameter) {
                float z_floor = p_atom.z + sqrtf(collision_diameter * collision_diameter - lateral2);
                if (z_floor > tz) tz = z_floor;
            }
        }
        float dz = tz - pos.z;
        // Diffusion-model fix (2026-09-07): a hop is a surface move, not a climb.
        //  - one hop may not lift the atom more than a single atomic step
        //    (collision_diameter). The unbounded "settle onto the tallest
        //    laterally-reachable neighbour" rule + unconditional uphill accept was the
        //    ratchet that emptied the film (see mobility-gate note above).
        //  - reject a hop whose target found NO support (tz fell back to the r_nm
        //    floor) while the atom sits well above the substrate -- a hop into vacuum.
        if (dz > collision_diameter) continue;
        if (tz <= r_nm + 1e-4f && pos.z > r_nm + collision_diameter) continue;
        // settle (dz <= 0) is free; a small climb (0 < dz <= collision_diameter) is
        // Boltzmann-gated (was previously accepted unconditionally).
        if (dz <= 0.0f || next_rand() < expf(-(dz * 0.1f) / kB_T)) {
            pos.x = tx; pos.y = ty; pos.z = tz;
        }
    }
    new_atoms[idx] = pos;
}

extern "C" __global__
void CONTACT_RELAXATION_KERNEL(
    float3*       new_atoms,      // [n_active] raw rigid-2R contact positions, IN/OUT
    const float3* all_positions,
    const int*    cell_starts,
    const int*    sorted_indices,
    const int nx, const int ny, const int nz,
    const float gm_x, const float gm_y, const float gm_z,
    const float   cell_size,
    const float   relax_radius,
    const int     num_ring_candidates,
    const float   bond_energy_eV,
    const float   coord_shell_factor,   // neighbour counted as "touching" if dist < collision_diameter*coord_shell_factor
    const float   collision_diameter,
    const float   flux_dx, const float flux_dy, const float flux_dz,  // this batch's incoming flux direction
    const float   kB_T,            // kB * deposition temperature [eV] -- makes bond_energy_eV load-bearing (BUG A fix)
    const int     n_active,
    const float   bw, const float bd,
    const int     n_stable,          // TWO-TIER GRID (dev, 2026-09-03): atoms below this index are covered by cell_starts/sorted_indices
    const int     n_active_pending   // TWO-TIER GRID: atoms [n_stable, n_active_pending) are NOT in the grid yet, checked by brute force below
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_active) return;

    int3 grid_dims = {nx, ny, nz};
    float3 grid_min = {gm_x, gm_y, gm_z};
    float2 box_size = {bw, bd};

    float3 P0 = new_atoms[idx];
    if (P0.z > 1e19f) return;   // safety: same overflow guard as DIFFUSION_KERNEL

    float shell2 = (collision_diameter * coord_shell_factor) * (collision_diameter * coord_shell_factor);

    // Local helper (inline, no function pointers on device): count neighbours within
    // the "touching" shell around (px,py,pz), given the (gx,gy) cell already known.
    // Used for BOTH P0 (BUG B fix: P0's own coordination, at its true unmodified
    // position, with no floor-recompute) and every ring candidate.
    auto count_coord = [&](float px, float py, float pz, int gx, int gy) -> int {
        int coord = 0;
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int gxx = (gx + dx % grid_dims.x + grid_dims.x) % grid_dims.x;
                int gyy = (gy + dy % grid_dims.y + grid_dims.y) % grid_dims.y;
                for (int gz = grid_dims.z - 1; gz >= 0; gz--) {
                    int c_idx = (gxx * grid_dims.y + gyy) * grid_dims.z + gz;
                    int start = cell_starts[c_idx];
                    int end = cell_starts[c_idx + 1];
                    for (int k = start; k < end; k++) {
                        float3 nb = all_positions[sorted_indices[k]];
                        float nddx = nb.x - px, nddy = nb.y - py, nddz = nb.z - pz;
                        float d2 = nddx*nddx + nddy*nddy + nddz*nddz;
                        if (d2 > 1e-8f && d2 < shell2) coord++;
                    }
                }
            }
        }
        // TWO-TIER GRID (dev, 2026-09-03): pending atoms [n_stable, n_active_pending) are
        // not yet in cell_starts/sorted_indices -- brute-force them too, same distance
        // test as the grid-cell loop above (no minimum-image correction here, matching
        // that loop exactly: all_positions is populated from already-wrapped deposition
        // coordinates, unlike the transient P0 handled separately above via p0wx/p0wy).
        for (int pidx = n_stable; pidx < n_active_pending; pidx++) {
            float3 nb = all_positions[pidx];
            float nddx = nb.x - px, nddy = nb.y - py, nddz = nb.z - pz;
            float d2 = nddx*nddx + nddy*nddy + nddz*nddz;
            if (d2 > 1e-8f && d2 < shell2) coord++;
        }
        return coord;
    };

    // BUG B fix: candidate 0 (the safe no-op default) is P0 verbatim -- no floor
    // recompute. The floor-raise loop (used only for ring candidates below) adds the
    // FULL collision_diameter regardless of lateral offset, which overestimates height
    // for a side-touch contact and can push a recomputed "P0" above the true ray-sphere
    // contact point -- exactly the failure the audit caught at near-grazing incidence.
    //
    // OOB FIX (post-re-audit): positions_gpu stores UNWRAPPED trajectory coordinates
    // (no fmodf anywhere in the Python insertion path), and at near-grazing incidence
    // P0.x/P0.y can land many multiples of box_size outside [0, box_width) -- the DDA
    // ray-sphere search travels far laterally before a hit. Grid-cell indices must be
    // computed from a WRAPPED copy (mirrors every other lookup in this file: ring
    // candidates below, DIFFUSION_KERNEL, RAY_SPHERE_KERNEL); using raw unwrapped P0.x/
    // P0.y here (as the first fix did) can make gx0/gy0 more negative than -grid_dims,
    // breaking the "(gx0 + dx%grid_dims.x + grid_dims.x) % grid_dims.x" wraparound trick
    // and producing a negative c_idx -- an out-of-bounds GPU memory read into
    // cell_starts/sorted_indices. Only the INDEX computation wraps; count_coord still
    // receives the TRUE unwrapped P0 position for the distance/score calculation, so the
    // returned position/score are unchanged from the Bug B fix.
    float p0wx = fmodf(P0.x, box_size.x); if (p0wx < 0) p0wx += box_size.x;
    float p0wy = fmodf(P0.y, box_size.y); if (p0wy < 0) p0wy += box_size.y;
    int gx0 = floorf((p0wx - grid_min.x) / cell_size);
    int gy0 = floorf((p0wy - grid_min.y) / cell_size);
    float p0_score = (float)count_coord(P0.x, P0.y, P0.z, gx0, gy0) * bond_energy_eV;

    float3 best_pos = P0;
    // BUG A fix: bond_energy_eV now compared against kB_T, not merely used as a
    // multiplicative constant on an integer count (which never affects an argmax).
    // A ring candidate only replaces P0 if its coordination-energy gain exceeds one
    // thermal unit -- a real, deterministic (non-stochastic, consistent with this
    // kernel's reproducible-settling design) energetic acceptance test in which the
    // bond_energy_eV value actually changes the outcome: a higher value clears the
    // kB_T bar more easily, a lower value keeps the safe P0 default more often.
    float best_score = p0_score;

    for (int c = 1; c <= num_ring_candidates; c++) {
        float angle = 6.2831853f * (float)(c - 1) / (float)num_ring_candidates;
        float cx = P0.x + relax_radius * cosf(angle);
        float cy = P0.y + relax_radius * sinf(angle);
        float wx = fmodf(cx, box_size.x); if (wx < 0) wx += box_size.x;
        float wy = fmodf(cy, box_size.y); if (wy < 0) wy += box_size.y;

        // z-floor: candidate cannot overlap any existing neighbour (hard constraint,
        // same logic/tolerance as the pre-existing DIFFUSION_KERNEL). Only applied to
        // ring candidates -- P0 already has its true, already-valid contact z (BUG B).
        float cz = collision_diameter;
        int gx = floorf((wx - grid_min.x) / cell_size);
        int gy = floorf((wy - grid_min.y) / cell_size);
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int gxx = (gx + dx % grid_dims.x + grid_dims.x) % grid_dims.x;
                int gyy = (gy + dy % grid_dims.y + grid_dims.y) % grid_dims.y;
                for (int gz = grid_dims.z - 1; gz >= 0; gz--) {
                    int c_idx = (gxx * grid_dims.y + gyy) * grid_dims.z + gz;
                    int start = cell_starts[c_idx];
                    int end = cell_starts[c_idx + 1];
                    for (int k = start; k < end; k++) {
                        float z_atom = all_positions[sorted_indices[k]].z;
                        if (z_atom + collision_diameter > cz) cz = z_atom + collision_diameter;
                    }
                }
            }
        }
        // TWO-TIER GRID (dev, 2026-09-03): pending atoms, same unconditional-floor logic
        // as the grid-cell loop above (matches it exactly, no lateral filtering here --
        // that is this loop's own pre-existing behaviour, not something this change adds).
        for (int pidx = n_stable; pidx < n_active_pending; pidx++) {
            float z_atom = all_positions[pidx].z;
            if (z_atom + collision_diameter > cz) cz = z_atom + collision_diameter;
        }

        // Causality constraint: candidate displacement from P0, projected onto the
        // incoming flux direction, must not be negative -- the atom may not move
        // backward along the path it just ballistically traveled.
        float ddx = wx - P0.x, ddy = wy - P0.y, ddz = cz - P0.z;
        float proj = ddx * flux_dx + ddy * flux_dy + ddz * flux_dz;
        if (proj < -1e-4f) continue;

        float score = (float)count_coord(wx, wy, cz, gx, gy) * bond_energy_eV;
        if (score > best_score + kB_T) {
            best_score = score;
            best_pos.x = wx; best_pos.y = wy; best_pos.z = cz;
        }
    }

    new_atoms[idx] = best_pos;
}

extern "C" __global__
void REEMISSION_KERNEL(
    float3*       reject_atoms,   // [n_reject] raw rejected-contact positions, IN/OUT
    int*          stuck_flags,    // [n_reject] OUT: 1 = re-stuck, 0 = exhausted attempts (lost)
    const float3* all_positions,
    const int*    cell_starts,
    const int*    sorted_indices,
    const int nx, const int ny, const int nz,
    const float gm_x, const float gm_y, const float gm_z,
    const float   cell_size,
    const float   normal_shell_factor,   // neighbours within collision_diameter*this define the local normal/coord shell
    const float   collision_diameter,
    const float   sticking_probability,
    const int     max_attempts,
    const float   flux_dx, const float flux_dy, const float flux_dz,  // fallback normal if no neighbours found
    const int     n_reject,
    const float   bw, const float bd,
    unsigned int  seed,
    const int     n_stable,          // TWO-TIER GRID (dev, 2026-09-03): atoms below this index are covered by cell_starts/sorted_indices
    const int     n_active_pending   // TWO-TIER GRID: atoms [n_stable, n_active_pending) are NOT in the grid yet, checked by brute force below
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_reject) return;

    int3 grid_dims = {nx, ny, nz};
    float3 grid_min = {gm_x, gm_y, gm_z};
    float box_w = bw, box_d = bd;
    float shell = collision_diameter * normal_shell_factor;
    float r2 = collision_diameter * collision_diameter;

    unsigned int state = seed + idx * 2654435761u;
    auto next_rand = [&]() {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        return (float)state / (float)0xffffffff;
    };

    float3 P = reject_atoms[idx];
    if (P.z > 1e19f) { stuck_flags[idx] = 0; return; }  // safety guard, matches other kernels

    int stuck = 0;

    for (int attempt = 0; attempt < max_attempts; attempt++) {
        // ── 1. Local surface normal: sum of unit vectors from each in-shell
        // neighbour TO P (points away from the local mass -- "outward"). ──
        float pxw = fmodf(P.x, box_w); if (pxw < 0) pxw += box_w;
        // SECONDARY BUG FOUND (2026-08-24, same re-audit): this used to wrap with `box_w`
        // (box WIDTH) instead of `box_d` (box DEPTH). Harmless for every square box tested
        // tonight (box_w==box_d in all configs), but wrong for a non-square box. Fixed.
        float pyw = fmodf(P.y, box_d); if (pyw < 0) pyw += box_d;
        int gx = floorf((pxw - grid_min.x) / cell_size);
        int gy = floorf((pyw - grid_min.y) / cell_size);

        float nx_sum = 0.0f, ny_sum = 0.0f, nz_sum = 0.0f;
        int n_neighbours = 0;
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int gxx = (gx + dx % grid_dims.x + grid_dims.x) % grid_dims.x;
                int gyy = (gy + dy % grid_dims.y + grid_dims.y) % grid_dims.y;
                for (int gz = grid_dims.z - 1; gz >= 0; gz--) {
                    int c_idx = (gxx * grid_dims.y + gyy) * grid_dims.z + gz;
                    int start = cell_starts[c_idx];
                    int end = cell_starts[c_idx + 1];
                    for (int k = start; k < end; k++) {
                        float3 nb = all_positions[sorted_indices[k]];
                        // Minimum-image correction (matches RAY_SPHERE_KERNEL and the DDA
                        // loop below): all_positions stores UNWRAPPED trajectory coordinates,
                        // so a periodically-close neighbour can have a raw ddx/ddy of many
                        // box-widths. Without this, most real neighbours are silently
                        // undercounted -- this was the root cause of a 0/598-restuck failure
                        // caught in live testing (n_neighbours was ~always 0, forcing the
                        // -flux_dir fallback for every atom, so every re-emitted ray flew in
                        // the same fixed direction regardless of true local geometry).
                        float rawx = P.x - nb.x, rawy = P.y - nb.y;
                        rawx -= roundf(rawx / box_w) * box_w;
                        rawy -= roundf(rawy / box_d) * box_d;
                        float ddx = rawx, ddy = rawy, ddz = P.z - nb.z;
                        float d2 = ddx*ddx + ddy*ddy + ddz*ddz;
                        if (d2 > 1e-8f && d2 < shell*shell) {
                            float d = sqrtf(d2);
                            nx_sum += ddx / d; ny_sum += ddy / d; nz_sum += ddz / d;
                            n_neighbours++;
                        }
                    }
                }
            }
        }
        // TWO-TIER GRID (dev, 2026-09-03): pending atoms [n_stable, n_active_pending) are
        // not yet in cell_starts/sorted_indices -- brute-force them too, using P (this
        // attempt's fixed query point) as the minimum-image reference, same convention
        // as the grid-cell loop directly above (safe here: P does not change within one
        // attempt iteration, unlike the DDA traversal loop's vx/vy further below).
        for (int pidx = n_stable; pidx < n_active_pending; pidx++) {
            float3 nb = all_positions[pidx];
            float rawx = P.x - nb.x, rawy = P.y - nb.y;
            rawx -= roundf(rawx / box_w) * box_w;
            rawy -= roundf(rawy / box_d) * box_d;
            float ddx = rawx, ddy = rawy, ddz = P.z - nb.z;
            float d2 = ddx*ddx + ddy*ddy + ddz*ddz;
            if (d2 > 1e-8f && d2 < shell*shell) {
                float d = sqrtf(d2);
                nx_sum += ddx / d; ny_sum += ddy / d; nz_sum += ddz / d;
                n_neighbours++;
            }
        }
        if (attempt == 0) { stuck_flags[idx] = n_neighbours; }  // TEMP DIAGNOSTIC: raw neighbour count, attempt 0 only
        float3 normal;
        if (n_neighbours > 0) {
            float nlen = sqrtf(nx_sum*nx_sum + ny_sum*ny_sum + nz_sum*nz_sum);
            if (nlen > 1e-6f) { normal.x = nx_sum/nlen; normal.y = ny_sum/nlen; normal.z = nz_sum/nlen; }
            else { normal.x = -flux_dx; normal.y = -flux_dy; normal.z = -flux_dz; }
        } else {
            normal.x = -flux_dx; normal.y = -flux_dy; normal.z = -flux_dz;  // fallback: away from incoming beam
        }

        // ── 2. Orthonormal basis (t1, t2, normal) ──
        float3 up;
        if (fabsf(normal.z) < 0.99f) { up.x = 0.0f; up.y = 0.0f; up.z = 1.0f; }
        else                          { up.x = 1.0f; up.y = 0.0f; up.z = 0.0f; }
        float3 t1;
        t1.x = up.y*normal.z - up.z*normal.y;
        t1.y = up.z*normal.x - up.x*normal.z;
        t1.z = up.x*normal.y - up.y*normal.x;
        float t1len = sqrtf(t1.x*t1.x + t1.y*t1.y + t1.z*t1.z);
        t1.x /= t1len; t1.y /= t1len; t1.z /= t1len;
        float3 t2;
        t2.x = normal.y*t1.z - normal.z*t1.y;
        t2.y = normal.z*t1.x - normal.x*t1.z;
        t2.z = normal.x*t1.y - normal.y*t1.x;

        // ── 3. Cosine-law (Lambertian) direction sample in local frame ──
        float u1 = next_rand(), u2 = next_rand();
        float cos_th = sqrtf(u1);
        float sin_th = sqrtf(1.0f - u1);
        float phi = 6.2831853f * u2;
        float lx = sin_th * cosf(phi), ly = sin_th * sinf(phi), lz = cos_th;
        float3 D;
        D.x = lx*t1.x + ly*t2.x + lz*normal.x;
        D.y = lx*t1.y + ly*t2.y + lz*normal.y;
        D.z = lx*t1.z + ly*t2.z + lz*normal.z;

        // ── 4. DDA ray-sphere search from P along D (per-thread direction;
        // same traversal algorithm as RAY_SPHERE_KERNEL, adapted per-thread). ──
        int3 current_voxel;
        current_voxel.x = floorf((pxw - grid_min.x) / cell_size);
        current_voxel.y = floorf((pyw - grid_min.y) / cell_size);
        current_voxel.z = floorf((P.z - grid_min.z) / cell_size);
        int step_x = (D.x > 0) ? 1 : (D.x < 0 ? -1 : 0);
        int step_y = (D.y > 0) ? 1 : (D.y < 0 ? -1 : 0);
        int step_z = (D.z > 0) ? 1 : (D.z < 0 ? -1 : 0);
        float tDeltaX = (D.x != 0) ? fabsf(cell_size / D.x) : 1e20f;
        float tDeltaY = (D.y != 0) ? fabsf(cell_size / D.y) : 1e20f;
        float tDeltaZ = (D.z != 0) ? fabsf(cell_size / D.z) : 1e20f;
        float tMaxX = (D.x > 0) ? (floorf((pxw - grid_min.x)/cell_size)+1.0f)*cell_size + grid_min.x - pxw :
                                   (floorf((pxw - grid_min.x)/cell_size))*cell_size + grid_min.x - pxw;
        float tMaxY = (D.y > 0) ? (floorf((pyw - grid_min.y)/cell_size)+1.0f)*cell_size + grid_min.y - pyw :
                                   (floorf((pyw - grid_min.y)/cell_size))*cell_size + grid_min.y - pyw;
        float tMaxZ = (D.z > 0) ? (floorf((P.z - grid_min.z)/cell_size)+1.0f)*cell_size + grid_min.z - P.z :
                                   (floorf((P.z - grid_min.z)/cell_size))*cell_size + grid_min.z - P.z;
        tMaxX = (D.x != 0) ? tMaxX / D.x : 1e20f;
        tMaxY = (D.y != 0) ? tMaxY / D.y : 1e20f;
        tMaxZ = (D.z != 0) ? tMaxZ / D.z : 1e20f;

        float t_min = 1e20f;
        bool hit = false;
        for (int step = 0; step < 10000; step++) {
            if (current_voxel.z < -1) break;  // matches RAY_SPHERE_KERNEL exactly (no upper bound --
                                                // out-of-range z cells are simply skipped below, not
                                                // a traversal-stopping condition; an added upper bound
                                                // here was cutting off upward-pointing re-emission rays
                                                // before they could reach any real structure)
            int gxx2 = (current_voxel.x % grid_dims.x + grid_dims.x) % grid_dims.x;
            int gyy2 = (current_voxel.y % grid_dims.y + grid_dims.y) % grid_dims.y;
            int gz2 = current_voxel.z;
            float vx = grid_min.x + (current_voxel.x + 0.5f) * cell_size;
            float vy = grid_min.y + (current_voxel.y + 0.5f) * cell_size;
            if (gz2 >= 0 && gz2 < grid_dims.z) {
                int c_idx = (gxx2 * grid_dims.y + gyy2) * grid_dims.z + gz2;
                int start = cell_starts[c_idx];
                int end = cell_starts[c_idx + 1];
                for (int k = start; k < end; k++) {
                    float3 C = all_positions[sorted_indices[k]];
                    float cdx = vx - C.x, cdy = vy - C.y;
                    C.x += roundf(cdx / box_w) * box_w;
                    C.y += roundf(cdy / box_d) * box_d;
                    float3 L = {pxw - C.x, pyw - C.y, P.z - C.z};
                    float b = 2.0f * (D.x*L.x + D.y*L.y + D.z*L.z);
                    float c = (L.x*L.x + L.y*L.y + L.z*L.z) - r2;
                    float disc = b*b - 4.0f*c;
                    if (disc >= 0) {
                        float sd = sqrtf(disc);
                        float t = (-b - sd) / 2.0f;
                        if (t > 1e-4f && t < t_min) { t_min = t; hit = true; }
                    }
                }
            }
            if (hit && t_min <= fminf(fminf(tMaxX, tMaxY), tMaxZ)) break;
            if (tMaxX < tMaxY) {
                if (tMaxX < tMaxZ) { current_voxel.x += step_x; tMaxX += tDeltaX; }
                else               { current_voxel.z += step_z; tMaxZ += tDeltaZ; }
            } else {
                if (tMaxY < tMaxZ) { current_voxel.y += step_y; tMaxY += tDeltaY; }
                else               { current_voxel.z += step_z; tMaxZ += tDeltaZ; }
            }
        }

        // TWO-TIER GRID (dev, 2026-09-03): pending atoms are not covered by
        // cell_starts/sorted_indices, so the DDA cell traversal above cannot see them.
        // Brute-force them here using the same explicit periodic-image test as
        // RAY_SPHERE_KERNEL's own pending-atom loop -- NOT a single fixed reference
        // point, because this ray (like RAY_SPHERE_KERNEL's) has no one "current
        // position" valid for every candidate atom along a potentially long grazing
        // path; see RAY_SPHERE_KERNEL's own comment for the full reasoning. Placed
        // after the traversal loop (not gated by its early-break) since it is
        // independent of cell-visit order and the early break's correctness there
        // only concerns the grid part.
        for (int pidx = n_stable; pidx < n_active_pending; pidx++) {
            float3 C0 = all_positions[pidx];
            for (int ix = -1; ix <= 1; ix++) {
                for (int iy = -1; iy <= 1; iy++) {
                    float3 C = { C0.x + ix * box_w, C0.y + iy * box_d, C0.z };
                    float3 L = { pxw - C.x, pyw - C.y, P.z - C.z };
                    float b = 2.0f * (D.x*L.x + D.y*L.y + D.z*L.z);
                    float c = (L.x*L.x + L.y*L.y + L.z*L.z) - r2;
                    float disc = b*b - 4.0f*c;
                    if (disc >= 0) {
                        float sd = sqrtf(disc);
                        float t = (-b - sd) / 2.0f;
                        if (t > 1e-4f && t < t_min) { t_min = t; hit = true; }
                    }
                }
            }
        }

        if (!hit) { break; }  // escaped -- re-evaporated, no further attempts (stuck stays whatever it was)
        stuck = -1;  // TEMP DIAGNOSTIC: -1 = "at least one hit found", distinguishes from 0 = "never hit"

        // ── 5. New contact point, fresh sticking test ──
        float new_x = pxw + t_min * D.x;
        float new_y = pyw + t_min * D.y;
        float new_z = P.z + t_min * D.z;
        P.x = new_x; P.y = new_y; P.z = new_z;

        float u_stick = next_rand();
        if (u_stick < sticking_probability) { stuck = 1; break; }
        // else: rejected again -- loop continues, re-emitting from this new contact point
    }

    reject_atoms[idx] = P;
    stuck_flags[idx] = stuck;
}
'''

DIFFUSION_KERNEL_MARKER = '\nextern "C" __global__\nvoid DIFFUSION_KERNEL'
RAY_SPHERE_ONLY_KERNEL_SOURCE = RAY_SPHERE_KERNEL_SOURCE.split(DIFFUSION_KERNEL_MARKER, 1)[0]

class GPUGrid:
    """
    Uniform Grid spatial partition on GPU using CSR-like sorting.
    """
    def __init__(self, cell_size: float, box_dims: tuple):
        self.cell_size = float(cell_size)
        self.bw, self.bd = box_dims
        self.grid_dims = None
        self.grid_min = xp.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        self.cell_starts = None
        self.sorted_indices = None
        
    def rebuild(self, positions_gpu, max_height: float, active_window_height: float = 600.0):
        n = len(positions_gpu)
        if n == 0: return

        # Clamp Z-grid to the active slab — keeps nz constant regardless of film height
        gz_min = max(0.0, max_height - active_window_height)
        z_range = min(max_height + 25.0, active_window_height + 25.0)
        nz = max(1, int(np.ceil(z_range / self.cell_size)))
        nx = max(1, int(np.ceil(self.bw / self.cell_size)))
        ny = max(1, int(np.ceil(self.bd / self.cell_size)))
        self.grid_dims = (nx, ny, nz)

        # Cache grid_min on host (avoids .get() sync in the hot path)
        self._host_grid_min = [0.0, 0.0, gz_min]
        self.grid_min = cp.array([0.0, 0.0, gz_min], dtype=np.float32)
        
        # 1. Compute cell index for each atom (Z offset by slab floor)
        coords = positions_gpu
        ix = cp.floor(coords[:, 0] / self.cell_size).astype(cp.int32) % nx
        iy = cp.floor(coords[:, 1] / self.cell_size).astype(cp.int32) % ny
        iz = cp.clip(
            cp.floor((coords[:, 2] - gz_min) / self.cell_size).astype(cp.int32),
            0, nz - 1
        )
        
        cell_indices = (ix * ny + iy) * nz + iz

        # --- GPU-memory preflight (avoid CuPy std::bad_alloc) -----------------
        # cp.argsort on n int32 keys allocates an int64 index array + thrust temp
        # + the .astype(int32) copy + sorted_cells gather. Model ~28 bytes/atom on
        # top of what is already live. If that does not fit, abort cleanly with a
        # clear message BEFORE the C++ allocator throws std::bad_alloc.
        try:
            _free_b, _total_b = cp.cuda.runtime.memGetInfo()
        except Exception:
            _free_b = _total_b = None
        if _free_b is not None:
            _need = int(n) * 28
            if _need > 0.92 * _free_b:
                raise MemoryError(
                    f"[GRID-PREFLIGHT] Aborting before cp.argsort to avoid std::bad_alloc: "
                    f"sorting n={n:,} atoms needs ~{_need/1024**3:.2f} GB but only "
                    f"{_free_b/1024**3:.2f} GB free / {_total_b/1024**3:.2f} GB total. "
                    f"Reduce active_window_height / resume_active_window_height so the "
                    f"active slab fits this GPU."
                )

        # 2. Sort atoms by cell index
        self.sorted_indices = cp.argsort(cell_indices).astype(cp.int32)
        sorted_cells = cell_indices[self.sorted_indices]
        
        # 3. Build cell_starts (CSR style)
        n_cells = nx * ny * nz
        self.cell_starts = cp.zeros(n_cells + 1, dtype=cp.int32)
        counts = cp.bincount(sorted_cells, minlength=n_cells).astype(cp.int32)
        self.cell_starts[1:] = cp.cumsum(counts).astype(cp.int32)

    def get_params(self):
        nx, ny, nz = self.grid_dims
        return (
            int(nx), int(ny), int(nz),
            self.grid_min,
            float(self.cell_size)
        )

    def get_params_host(self):
        """Return grid parameters as plain Python scalars (no GPU sync needed)."""
        nx, ny, nz = self.grid_dims
        # grid_min is a tiny 3-element array; cache on host at rebuild time
        gm = self._host_grid_min
        return int(nx), int(ny), int(nz), gm[0], gm[1], gm[2], float(self.cell_size)


# ---------------------------------------------------------------------------
# GLAD V3 Simulator
# ---------------------------------------------------------------------------

class GLADV3Simulator:
    """
    Full 3D GLAD simulator (Bead-Spring model).

    The substrate is represented as a set of rigid spheres of radius R=0.1 nm
    (author norm). Incoming atoms travel along the ballistic flux direction and
    either:
      (a) Hit a sphere → deposit at the contact point + surface diffusion.
      (b) Miss entirely → atom is lost (open surface, consistent with GLAD).

    Shadowing is implicit: an atom that would hit a sphere lower in its
    trajectory will be stopped by that sphere — the cKDTree query returns ALL
    spheres along the ray's XY projection first, letting the quadratic kernel
    pick the nearest one.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        # PERF-1 (2026-09-06, validated byte-identical): pick the fast plain device allocator
        # for boxes that fit in VRAM; keep managed memory (host-RAM paging headroom) for large
        # boxes. Skipped if GLAD_GPU_ALLOCATOR pinned a choice at import. set_allocator affects
        # only FUTURE allocations and runs here before any large grid array is built, so
        # switching is safe. Does NOT change physics -- V1 gate: max |delta| = 0 on a real
        # box=100 run (PERF1_ALLOCATOR_VALIDATION_20260906/).
        if GPU_AVAILABLE and not _ALLOC_ENV_FORCED:
            _bw_perf1 = float(cfg.get('box_width', 100.0))
            if _bw_perf1 <= 160.0:
                cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
                print(f"[PERF-1] box_width={_bw_perf1:.0f}nm <= 160 -> PLAIN device MemoryPool (faster; fits VRAM)", flush=True)
            else:
                print(f"[PERF-1] box_width={_bw_perf1:.0f}nm > 160 -> MANAGED memory (paging headroom)", flush=True)
        for _k in ('alpha', 'pitch', 'batch_size', 'author_radius_nm'):
            if _k not in self.cfg:
                raise ValueError(f"[GLAD FATAL] Missing required physics parameter: '{_k}'")
        self._r  = np.float32(self.cfg['author_radius_nm'])
        self._cd = np.float32(2.0 * self.cfg['author_radius_nm'])
        _relax_r_cfg = self.cfg.get('contact_relaxation_radius_nm')
        self._relax_radius_nm = np.float32(_relax_r_cfg if _relax_r_cfg is not None else 0.5 * float(self._cd))
        # Dedicated, always-correctly-temperature-coupled kB*T for CONTACT_RELAXATION_KERNEL.
        # NOT the same as self._kB_T above, which is gated behind the unrelated legacy
        # `use_arrhenius` toy-diffusion flag and silently falls back to a hardcoded 0.025eV
        # (~300K) when that flag is off -- ~3% off at this session's 298.15K test cases, up to
        # ~40% off at e.g. 500K. A separate value avoids touching that old toy-kernel's
        # behaviour (which must stay byte-identical for existing configs) while giving the new
        # relaxation kernel a real kB*cfg['temperature'] regardless of the unrelated flag.
        self.kB = 8.617333262145e-5
        self._kB_T_relax = np.float32(self.kB * cfg['temperature'])
        material = str(self.cfg.get('material', 'Cu'))
        if self.cfg.get('melting_point_K') is None:
            self.cfg['melting_point_K'] = MATERIAL_MELTING_POINTS_K.get(material, MATERIAL_MELTING_POINTS_K['Cu'])
        self.cfg['homologous_temperature'] = (
            float(self.cfg.get('temperature', 300.0)) / float(self.cfg['melting_point_K'])
        )
        self.cfg['growth_efficiency'] = float(self.cfg.get('growth_efficiency', 1.0))
        self.cfg['vertical_growth_rate'] = (
            float(self.cfg.get('r_dep', self.cfg.get('growth_rate', GROWTH_RATE)))
            * self.cfg['growth_efficiency']
        )

        # --- GPU Resident State (ACTIVE-SLAB-ONLY model) ---
        # positions_gpu holds ONLY the active slab: atoms within active_window_height
        # of the growth front. Buried atoms (which are outside the shadow window and
        # are never deposited onto again) are offloaded to host RAM in
        # frozen_positions_cpu. This keeps GPU memory bounded so large films can be
        # grown / resumed on small GPUs (e.g. the 4 GB GTX 1650). The full structure
        # is always reconstructable as concat(frozen_positions_cpu, slab) via
        # _all_positions_host().
        # Default slab capacity is small: the GPU holds only the ACTIVE slab,
        # which grows on demand (1.5x) as atoms deposit. On resume,
        # load_v3_checkpoint resizes this to fit the restored active slab.
        self.capacity = int(cfg.get('capacity', 4 * 1024 * 1024))   # GPU SLAB capacity
        self.positions_gpu = xp.zeros((self.capacity, 3), dtype=np.float32)
        # Buried atoms live on the host only.
        self.frozen_positions_cpu = np.empty((0, 3), dtype=np.float32)
        # active_indices is no longer used as a gather: the active set is simply the
        # contiguous slice positions_gpu[:n_active]. Kept as None for clarity.
        self.active_indices = None
        self.n_active = 0
        self.last_compaction_height = 0.0

        self.n_atoms = 0
        self.n_seed_atoms = 0
        self.max_height = 0.0
        self.atoms_since_deposit = 0
        self.last_checkpoint_height = 0.0
        self.write_interval = cfg.get('write_interval', 100)
        self.active_window_height = cfg.get('active_window_height', 600.0)
        # Optional resume-only window override (memory bound for small GPUs).
        self.resume_active_window_height = cfg.get('resume_active_window_height', None)

        # ── Grid-rebuild throttle ──────────────────────────────────────────
        # OVERLAP-VIOLATION FIX (2026-08-23, post-independent-audit): the old
        # `batch_size * 4` throttle let up to ~2048 atoms deposit against a stale
        # (pre-window) grid snapshot before the next rebuild, so contact/floor
        # checks for most of those atoms never saw each other -- confirmed via
        # live-GPU measurement to cause 55-90% of newly-deposited atoms to end up
        # closer than `collision_diameter` to a neighbour (99.3% of violations were
        # same-window, i.e. this throttle, not intra-batch parallelism, which is a
        # separate, smaller, NOT-fixed-here residual -- see
        # P1_CONTACT_RELAXATION_MODEL_DESIGN_20260823.md `S7_GRID_REBUILD_STALENESS_FIX`
        # section for the full before/after data).
        #
        # delta=1, not delta=batch_size: `batch_size` is the number of ray TRIALS
        # per deposit_batch() call, not the number that actually land (many miss,
        # especially at high alpha -- e.g. ~40-50% miss rate at alpha=85). Using
        # `_grid_rebuild_delta = batch_size` therefore does NOT guarantee a rebuild
        # every batch -- it only guarantees one once *cumulative yield* reaches
        # batch_size, which spans 1-3 batches depending on miss rate (confirmed:
        # only 5 rebuilds fired over 10 batches at alpha=85/box=25 with delta=512).
        # delta=1 forces a rebuild before every batch that added >=1 new active
        # atom (a batch that adds 0 needs no rebuild -- nothing changed), which is
        # the literal "rebuild every batch" behaviour intended by this fix.
        # TWO-TIER GRID (dev, 2026-09-03, DEV_GRID_PERF_VALIDATION_20260903/): delta no
        # longer needs to be 1 to preserve the S7 correctness guarantee. Every atom
        # deposited since the last stable rebuild (positions_gpu[n_stable:n_active]) is
        # now brute-force checked directly by all four kernels (see
        # GRID_REBUILD_PERFORMANCE_OPTIMIZATION_SCOPED_20260903.md) -- delta only
        # controls how large that brute-force "pending" range is allowed to grow before
        # the next full cp.argsort rebuild folds it into the stable grid, a PERFORMANCE
        # knob now, not a correctness one. Still defaults conservatively (K=10, per the
        # scoping doc's explicit recommendation, not a guess) pending the
        # baseline-vs-candidate violation-rate comparison in
        # DEV_GRID_PERF_VALIDATION_20260903/ -- raise only after that passes.
        self._cached_global_sorted_indices = None
        self._cached_grid_params = None
        self._n_active_at_last_rebuild = 0
        self._grid_rebuild_delta = int(cfg.get('grid_rebuild_stride', 10))
        self._grid_needs_rebuild = True   # always rebuild on first batch

        # ── Miss-rate tracking ─────────────────────────────────────────────
        self._consecutive_high_miss = 0   # batches in a row with > 99% miss
        self._last_batch_miss_count = 0

        # ── Probabilistic sticking (Option A; default OFF) ─────────────────
        self._last_batch_sticking_reject = 0
        sticking_seed = int(cfg.get('sticking_seed', 0x571CCC))
        self._sticking_rng = cp.random.default_rng(sticking_seed)

        # ── Async checkpoint — single persistent worker, non-blocking queue ─
        # Queue capacity = 1: if the worker is still writing, the new snapshot
        # is silently dropped (best-effort). The simulation never blocks on I/O.
        self._ckpt_queue: queue.Queue = queue.Queue(maxsize=1)
        self._ckpt_worker = threading.Thread(
            target=self._checkpoint_worker_loop,
            daemon=True,
            name="glad-checkpoint-worker",
        )
        self._ckpt_worker.start()

        # Legacy CPU sync (xyz/logs only)
        self.positions = None 
        self._tree = None
        self._tree_n = 0

        self.xyz_file = None
        self.lmp_file = None
        self._frame_idx = 0
        self.log_fh = open(cfg['log_file'], 'a', buffering=1, encoding='utf-8')
        self.start_time = time.time()

        alpha_rad = np.radians(cfg['alpha'])
        # Precompute scalar trig components; used by _compute_helical_flux_dir every batch
        self._sin_alpha  = float(np.sin(alpha_rad))
        self._cos_alpha  = float(np.cos(alpha_rad))
        # Helical dynamics parameters — read from config, fallback to module constants
        self.pitch       = float(cfg.get('pitch', PITCH))
        self.growth_rate = float(cfg.get('growth_rate', GROWTH_RATE))
        # Initial flux direction at θ=0 (Z=0, substrate level, before any rotation)
        self.flux_dir = np.array([self._sin_alpha, 0.0, -self._cos_alpha], dtype=np.float32)
        self.flux_dir /= np.linalg.norm(self.flux_dir)

        if cfg['use_arrhenius']:
            beta_kT = 1.0 / (self.kB * cfg['temperature'])
            self._hop_prob = float(np.exp(-cfg['activation_energy'] * beta_kT))
            self._kB_T = self.kB * cfg['temperature']
        else:
            self._hop_prob = 1.0
            self._kB_T = 0.025 # ~300K

        # Physically-calibrated Arrhenius diffusion model (additive; see
        # P1_PHYSICAL_DIFFUSION_MODEL_SCOPE_20260822.md §1). When enabled, hop_prob is
        # no longer a flat per-hop-attempt constant applied over a fixed hop count --
        # it is recomputed per batch in _step_batch() as
        # hop_probability = 1 - exp(-hop_rate * dt_real), with dt_real derived from the
        # real deposition time this batch of atoms represents. hop_rate itself (Hz) is
        # constant for the run (only T, Ea, nu0 depend on it, all fixed per-run), so it
        # is computed once here.
        self.use_physical_diffusion_model = bool(cfg.get('use_physical_diffusion_model', False))
        self.nu0_hz = float(cfg.get('nu0_hz', 8.2e11))
        # D1 (2026-09-07): see 'diffusion_resolve_same_batch' cfg-defaults comment.
        self._diffusion_resolve_same_batch = bool(cfg.get('diffusion_resolve_same_batch', False))  # DISABLED 2026-09-07, see cfg-defaults note
        self._d1_reverts_last_batch = 0
        self._d1_reverts_total = 0
        if self.use_physical_diffusion_model:
            beta_kT_phys = 1.0 / (self.kB * cfg['temperature'])
            self._hop_rate_hz = self.nu0_hz * float(np.exp(-cfg['activation_energy'] * beta_kT_phys))
        else:
            self._hop_rate_hz = 0.0

        # Mehl1999 Model-II EAM local-neighbour hop-barrier correction (opt-in; see
        # config-defaults comment block and P1_CONTACT_RELAXATION_MODEL_DESIGN_20260823.md).
        self.use_eam_neighbor_barrier = bool(cfg.get('use_eam_neighbor_barrier', False))
        self.eam_e0_eV   = float(cfg.get('eam_e0_eV', 0.487))
        self.eam_dNN_eV  = float(cfg.get('eam_dNN_eV', 0.274))
        self.eam_dNNN_eV = float(cfg.get('eam_dNNN_eV', 0.027))
        # Default cutoffs derived from collision_diameter (real Cu FCC nearest-neighbour
        # distance, 0.256nm) and its sqrt(2) multiple (real Cu FCC next-nearest-neighbour
        # in-plane distance, ~0.362nm) — self._cd is already set above this block.
        _nn_cfg = cfg.get('eam_nn_cutoff_nm', None)
        _nnn_cfg = cfg.get('eam_nnn_cutoff_nm', None)
        self.eam_nn_cutoff_nm  = float(_nn_cfg) if _nn_cfg is not None else float(self._cd)
        self.eam_nnn_cutoff_nm = float(_nnn_cfg) if _nnn_cfg is not None else float(self._cd) * 1.41421356

        self._shadow_range = cfg['box_width']
        cell_size_cfg = float(cfg.get('cell_size', 0.4))
        self.gpu_grid = GPUGrid(cell_size=cell_size_cfg, box_dims=(cfg['box_width'], cfg['box_depth']))
        self._kernel = None
        self._diff_kernel = None
        self._relax_kernel = None
        self._reemission_kernel = None

        # ── Grid OOM pre-check ────────────────────────────────────────────
        _nx_e = max(1, int(np.ceil(cfg['box_width']  / cell_size_cfg)))
        _ny_e = max(1, int(np.ceil(cfg['box_depth']  / cell_size_cfg)))
        _nz_e = max(1, int(np.ceil(
            min(float(cfg.get('target_height', 700)) + 25.0,
                float(cfg.get('active_window_height', 600.0)) + 25.0) / cell_size_cfg
        )))
        _n_cells_est = _nx_e * _ny_e * _nz_e
        _grid_vram_mb = _n_cells_est * 4 * 2 / (1024**2)   # cell_starts + counts int32
        if _n_cells_est > 100_000_000:
            print(
                f"[WARN] cell_size={cell_size_cfg}nm → estimated {_n_cells_est/1e6:.0f}M grid cells "
                f"({_grid_vram_mb:.0f} MB VRAM for grid alone). "
                "Consider cell_size: 1.0 or 2.0 in the config to avoid OOM.", flush=True
            )

        if GPU_AVAILABLE:
            try:
                needs_full = (self.cfg.get('enable_surface_diffusion', False)
                              or self.cfg.get('enable_contact_relaxation', False)
                              or self.cfg.get('enable_diffuse_reemission', False))
                kernel_source = RAY_SPHERE_KERNEL_SOURCE if needs_full else RAY_SPHERE_ONLY_KERNEL_SOURCE
                mod = cp.RawModule(code=kernel_source)
                self._kernel = mod.get_function('RAY_SPHERE_KERNEL')
                compiled = ['Ballistic']
                if self.cfg.get('enable_surface_diffusion', False):
                    self._diff_kernel = mod.get_function('DIFFUSION_KERNEL')
                    compiled.append('Diffusion')
                if self.cfg.get('enable_contact_relaxation', False):
                    self._relax_kernel = mod.get_function('CONTACT_RELAXATION_KERNEL')
                    compiled.append('ContactRelaxation')
                if self.cfg.get('enable_diffuse_reemission', False):
                    self._reemission_kernel = mod.get_function('REEMISSION_KERNEL')
                    compiled.append('Reemission')
                self._log(f"[V3] [OK] CUDA Kernels ({' + '.join(compiled)}) compiled")
            except Exception as e:
                self._log(f"[V3] [!] CUDA Compilation failed: {e}")

        # Throttled GPU thermal monitor (polls nvidia-smi every 5 s, not per-batch)
        self._thermal_monitor = GPUThermalMonitor(interval_s=5.0)

        if cfg.get('use_gpu', False) and not GPU_AVAILABLE:
            print("[FATAL] use_gpu: true but CuPy/CUDA initialization failed!", flush=True)
            sys.exit(1)

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self._log(f"[V3] Initialised (Full-GPU VRAM persistence active)")
        self._log(f"     Backend: {'GPU (VRAM-ONLY)' if self._kernel else 'CPU fallback'}")
        self._log(f"     Batch Size: {self.cfg.get('batch_size', 512)}")
        self._log(f"     Target: {self.cfg.get('target_height', 0):.1f} nm | "
                  f"vertical_rate={self.growth_rate:.4g} nm/s | "
                  f"rpm={self.cfg.get('rotation_rpm', 'pitch-mode')} | "
                  f"pitch={self.pitch:.3f} nm/turn | "
                  f"turns@target={self.cfg.get('target_height', 0) / self.pitch:.3f}")
        self._log(f"     Nominal rate: {self.cfg.get('r_dep', self.growth_rate):.4g} nm/s | "
                  f"growth_efficiency={self.cfg.get('growth_efficiency', 1.0):.4g}")
        self._log(f"     Temperature: Ts={self.cfg.get('temperature', 300.0):.2f} K | "
                  f"Tm={self.cfg.get('melting_point_K', 1357.77):.2f} K | "
                  f"Ts/Tm={self.cfg.get('homologous_temperature', 0.0):.3f}")
        if self.cfg.get('enable_surface_diffusion', False):
            if self.use_physical_diffusion_model:
                self._log(f"     Surface diffusion: ON | PHYSICAL Arrhenius model | "
                          f"Ea={self.cfg.get('activation_energy', 0.0):.3f} eV | "
                          f"nu0={self.nu0_hz:.3e} Hz | hop_rate={self._hop_rate_hz:.3e} Hz "
                          f"(dt_real, hop_prob computed per batch)")
            else:
                self._log(f"     Surface diffusion: ON | toy fixed-hop-count kernel | "
                          f"Ea={self.cfg.get('activation_energy', 0.0):.3f} eV | "
                          f"hop_prob={self._hop_prob:.3e}")
                if self._hop_prob < 1e-5:
                    self._log("     [NOTE] Hop probability is extremely small; this run is effectively ballistic.")
            if self.use_eam_neighbor_barrier:
                self._log(f"     Mehl1999 EAM barrier: ON | E0={self.eam_e0_eV:.3f}eV "
                          f"dNN={self.eam_dNN_eV:.3f}eV dNNN={self.eam_dNNN_eV:.3f}eV | "
                          f"NN_cutoff={self.eam_nn_cutoff_nm:.4f}nm NNN_cutoff={self.eam_nnn_cutoff_nm:.4f}nm "
                          "(Model II linear fit, Table II; distance-shell proxy, not literal fcc-lattice sites)")
        else:
            self._log("     Surface diffusion: OFF (pure ballistic bead-spring mode)")
        if self.cfg.get('source_distance_cm') is not None:
            self._log(f"     Source distance: {self.cfg['source_distance_cm']:.2f} cm "
                      "(metadata only; thermal/flux-distance model not yet coupled)")
        if self.cfg.get('angular_distribution_enabled', False):
            self._log("     Angular distribution: ON | "
                      f"type={self.cfg.get('angular_distribution_type')} | "
                      f"sigma={self.cfg.get('angular_sigma_deg', 0.0):.3f} deg | "
                      f"samples={self.cfg.get('angular_num_samples', 1)} | "
                      f"seed={self.cfg.get('angular_seed', 0)}")
        else:
            self._log("     Angular distribution: OFF (legacy mono-directional beam)")

        self._write_execution_manifest()

    # ------------------------------------------------------------------
    # Execution Manifest
    # ------------------------------------------------------------------

    def _write_execution_manifest(self) -> None:
        """Write execution_manifest.json AFTER all cfg overrides — this is the ground truth."""
        def _s(v):
            try: return float(v)
            except: return str(v)
        cfg_stable = json.dumps({k: _s(v) for k, v in sorted(self.cfg.items())}, sort_keys=True)
        cfg_hash = hashlib.sha256(cfg_stable.encode()).hexdigest()[:16]
        manifest = {
            "timestamp": datetime.now().isoformat(timespec='seconds'),
            "alpha_deg": float(self.cfg['alpha']),
            "pitch_nm": float(self.pitch),
            "batch_size": int(self.cfg.get('batch_size', 512)),
            "author_radius_nm": float(self._r),
            "enable_surface_diffusion": bool(self.cfg.get('enable_surface_diffusion', False)),
            "enable_sticking": bool(self.cfg.get('enable_sticking', False)),
            "sticking_probability": float(self.cfg.get('sticking_probability', 1.0)),
            "target_height_nm": float(self.cfg.get('target_height', 0)),
            "box_width_nm": float(self.cfg.get('box_width', 100)),
            "box_depth_nm": float(self.cfg.get('box_depth', 100)),
            "material": str(self.cfg.get('material', 'Cu')),
            "random_seed": int(self.cfg.get('random_seed', 0)),
            "temperature_K": float(self.cfg.get('temperature', 300.0)),
            "growth_efficiency": float(self.cfg.get('growth_efficiency', 1.0)),
            "cfg_hash": cfg_hash,
            "source_config": str(self.cfg.get('_source_yaml', 'unknown')),
        }
        out = Path(self.cfg['checkpoint_dir']) / 'execution_manifest.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        self._log(f"[V3] execution_manifest.json → {out}")
        self._log(f"     cfg_hash={cfg_hash} | alpha={manifest['alpha_deg']}° | "
                  f"pitch={manifest['pitch_nm']}nm | r={manifest['author_radius_nm']}nm | "
                  f"diffusion={manifest['enable_surface_diffusion']} | sticking={manifest['enable_sticking']}")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        if self.cfg['verbose']:
            print(line, flush=True)
        if getattr(self, 'log_fh', None) is not None and not self.log_fh.closed:
            self.log_fh.write(line + '\n')
            self.log_fh.flush()

    def _signal_handler(self, signum, frame):
        sig_name = signal.Signals(signum).name
        self._log(f"[V3] {sig_name} — draining checkpoint queue, then saving emergency ...")
        self.save_checkpoint(emergency=True)
        self._log("[V3] Emergency checkpoint saved. Exiting.")
        if getattr(self, 'log_fh', None) and not self.log_fh.closed:
            self.log_fh.close()
        sys.exit(0)

    # ------------------------------------------------------------------
    # Dynamic Helical Flux Direction
    # ------------------------------------------------------------------

    def _compute_helical_flux_dir(self, z_height: float) -> np.ndarray:
        """
        Compute the helically-rotated ballistic flux direction for a given height Z.

        Physical model: the substrate rotates by one full revolution per PITCH nm of
        deposited material.  In the fixed lab frame this is equivalent to the flux
        vector sweeping around the Z-axis:

            θ(Z) = 2π × Z / pitch

            v_flux(Z) = [ sin(α)·cos(θ),  sin(α)·sin(θ),  -cos(α) ]

        This is recomputed at the start of every batch so each atom experiences the
        correct azimuthal flux angle for its deposition height, producing authentic
        helical column growth without any retroactive coordinate remapping.
        """
        theta   = 2.0 * np.pi * z_height / self.pitch
        cos_th  = np.cos(theta)
        sin_th  = np.sin(theta)
        direction = np.array(
            [self._sin_alpha * cos_th,
             self._sin_alpha * sin_th,
             -self._cos_alpha],
            dtype=np.float32
        )
        norm = float(np.linalg.norm(direction))
        return direction / norm if norm > 1e-9 else direction

    def _compute_angular_flux_samples(self, nominal_dir: np.ndarray,
                                      z_height: float | None = None) -> np.ndarray:
        """
        Return deterministic A1 cone/ring directions around the nominal helical beam.

        Disabled, zero-spread, and one-sample cases deliberately reduce to the
        exact nominal vector so legacy behavior is preserved unless explicitly
        enabled with a nonzero spread and multiple samples.
        """
        nominal = np.asarray(nominal_dir, dtype=np.float64)
        n_norm = float(np.linalg.norm(nominal))
        if n_norm <= 1e-12:
            return np.asarray([nominal_dir], dtype=np.float32)
        nominal = nominal / n_norm

        ad = self.cfg.get('angular_distribution', {})
        enabled = bool(self.cfg.get('angular_distribution_enabled', False))
        sigma_deg = float(ad.get('angular_sigma_deg', 0.0))
        num_samples = int(ad.get('angular_num_samples', 1))
        if (not enabled) or sigma_deg <= 0.0 or num_samples <= 1:
            return np.asarray([nominal], dtype=np.float32)

        seed = int(ad.get('angular_seed', 0))
        height_key = 0 if z_height is None else int(round(float(z_height) * 1000.0))
        phase_rng = np.random.default_rng(np.uint64((seed + 0x9E3779B97F4A7C15 + height_key) & 0xFFFFFFFFFFFFFFFF))
        phase = float(phase_rng.uniform(0.0, 2.0 * np.pi))

        if ad.get('rotate_with_helical_azimuth', True):
            ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        u = np.cross(ref, nominal)
        if float(np.linalg.norm(u)) <= 1e-12:
            u = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float64), nominal)
        u /= float(np.linalg.norm(u))
        v = np.cross(nominal, u)
        v /= float(np.linalg.norm(v))

        cone = np.radians(sigma_deg)
        dirs = []
        for i in range(num_samples):
            phi = phase + 2.0 * np.pi * i / num_samples
            d = np.cos(cone) * nominal + np.sin(cone) * (np.cos(phi) * u + np.sin(phi) * v)
            d /= float(np.linalg.norm(d))
            dirs.append(d)
        return np.asarray(dirs, dtype=np.float32)

    def angular_distribution_metadata(self) -> dict:
        """Additive metadata used by dry-run previews and future checkpoints."""
        ad = self.cfg.get('angular_distribution', {})
        return {
            'angular_distribution_enabled': bool(self.cfg.get('angular_distribution_enabled', False)),
            'angular_distribution_type': self.cfg.get('angular_distribution_type', 'none'),
            'angular_sigma_deg': float(self.cfg.get('angular_sigma_deg', ad.get('angular_sigma_deg', 0.0))),
            'angular_num_samples': int(self.cfg.get('angular_num_samples', ad.get('angular_num_samples', 1))),
            'angular_seed': int(self.cfg.get('angular_seed', ad.get('angular_seed', 0))),
        }

# ------------------------------------------------------------------
    # Checkpoint I/O (CORRIGÉ)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Active-slab-only GPU helpers
    # ------------------------------------------------------------------

    def _gpu_free_gb(self):
        """Return (free_gb, total_gb) of CUDA device memory, or (None, None)."""
        if not GPU_AVAILABLE:
            return None, None
        try:
            free_b, total_b = cp.cuda.runtime.memGetInfo()
            return free_b / 1024**3, total_b / 1024**3
        except Exception:
            return None, None

    def _all_positions_host(self) -> np.ndarray:
        """
        Reconstruct the FULL structure as a single host (N, 3) float32 array:
        concat(frozen_positions_cpu [buried, host], active slab [GPU -> host]).

        In the active-slab-only model positions_gpu holds ONLY the active slab
        ([0:n_active]); buried atoms live in frozen_positions_cpu. This is the
        single source of truth for checkpoint save / validation / CPU tree.
        """
        slab = self.positions_gpu[:self.n_active]
        slab_host = slab.get() if (GPU_AVAILABLE and hasattr(slab, 'get')) else np.asarray(slab)
        slab_host = np.ascontiguousarray(slab_host, dtype=np.float32)
        if self.frozen_positions_cpu is None or self.frozen_positions_cpu.shape[0] == 0:
            return slab_host
        return np.ascontiguousarray(
            np.concatenate([self.frozen_positions_cpu, slab_host], axis=0),
            dtype=np.float32,
        )

    def _evict_buried_to_cpu(self, z_min_filter: float):
        """
        Sliding-window compaction (active-slab-only model): move atoms in the GPU
        slab whose z < z_min_filter to host RAM (frozen_positions_cpu) and compact
        the kept atoms to the front of positions_gpu so [0:n_active] stays
        contiguous and active. Bounds GPU memory; no physics/formula change.
        """
        if self.n_active == 0:
            return
        slab = self.positions_gpu[:self.n_active]
        keep_mask = slab[:, 2] >= z_min_filter
        n_keep = int(keep_mask.sum())
        if n_keep == self.n_active:
            return  # nothing buried below the slab floor
        buried = slab[~keep_mask]
        buried_host = buried.get() if (GPU_AVAILABLE and hasattr(buried, 'get')) else np.asarray(buried)
        if buried_host.shape[0]:
            self.frozen_positions_cpu = np.ascontiguousarray(
                np.concatenate([self.frozen_positions_cpu,
                                buried_host.astype(np.float32)], axis=0),
                dtype=np.float32,
            )
        kept = slab[keep_mask]                  # fancy-index copy (safe to write back)
        self.positions_gpu[:n_keep] = kept
        self.n_active = n_keep
        self.last_compaction_height = self.max_height
        self._grid_needs_rebuild = True
        if GPU_AVAILABLE:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

    def load_v3_checkpoint(self, filepath: str) -> bool:
        """
        Load a V3 HDF5 checkpoint into VRAM.

        By default, only post-fix checkpoints are accepted. Older checkpoints
        were generated with the 1R contact-geometry bug, so resuming them with
        the corrected 2R physics would mix incompatible growth laws.

        Supports both legacy (N, 3) checkpoints and the current (N, 4) helical schema
        only when checkpoint.allow_legacy_resume is explicitly enabled.
        When a 4D checkpoint is loaded, the T column is discarded — it is always
        recomputed deterministically as Z / growth_rate on the next save.  The internal
        positions_gpu array remains (N, 3) throughout the simulation; 4D serialisation
        happens only at checkpoint write time.
        """
        if not os.path.exists(filepath):
            return False
        try:
            with h5py.File(filepath, 'r') as f:
                schema = f.attrs.get('schema', '3D_legacy')
                schema_str = str(schema)
                contact_geometry = str(f.attrs.get('contact_geometry', ''))
                collision_diameter = float(f.attrs.get('collision_diameter', 0.0))
                is_fixed_checkpoint = (
                    contact_geometry == '2R_physical_contact'
                    and schema_str.endswith('v3.2')
                    and abs(collision_diameter - float(self._cd)) < 1e-6
                )
                if not is_fixed_checkpoint and not self.cfg.get('allow_legacy_resume', False):
                    raise ValueError(
                        "Refusing incompatible checkpoint: missing post-fix 2R metadata "
                        f"(schema={schema_str}, contact_geometry={contact_geometry or 'missing'}, "
                        f"collision_diameter={collision_diameter:g}). Start fresh, or set "
                        "checkpoint.allow_legacy_resume=true only for forensic comparison."
                    )

                raw = np.array(f['positions'], dtype=np.float32)

                if raw.shape[1] == 4:
                    # Native helical 4D checkpoint: extract XYZ, discard T column
                    pos = raw[:, :3]
                    # Also restore helical parameters if they were saved
                    if 'pitch' in f.attrs:
                        self.pitch       = float(f.attrs['pitch'])
                    if 'growth_rate' in f.attrs:
                        self.growth_rate = float(f.attrs['growth_rate'])
                    self._log(f"[V3] 4D checkpoint detected (schema={schema}), "
                              f"pitch={self.pitch} nm, growth_rate={self.growth_rate} nm/s")
                elif raw.shape[1] == 3:
                    pos = raw
                    self._log(f"[V3] Legacy 3D checkpoint detected — T column will be generated on next save")
                else:
                    raise ValueError(f"Unexpected positions shape {raw.shape}: expected (N,3) or (N,4)")

                self.n_atoms = len(pos)
                self.n_seed_atoms = int(f.attrs.get('seed_atoms', 0))
                # NOTE: full positions are intentionally NOT uploaded to GPU here.
                # The active-slab-only split below keeps all N atoms on the HOST and
                # uploads ONLY the active slab. (Uploading all N was the cause of the
                # std::bad_alloc OOM on the 4 GB GTX 1650.)

                real_h = float(pos[:, 2].max())
                if getattr(self, 'force_dynamic', False):
                    self._log(f"[V3] [FORCE] Recalculating height: {real_h:.2f} nm "
                              f"(ignored saved {f.attrs.get('current_height', 0):.2f})")
                    self.max_height = real_h
                else:
                    self.max_height = float(f.attrs.get('current_height', real_h))

                self.last_checkpoint_height = self.max_height

            # --- ACTIVE-SLAB-ONLY resume restoration -------------------------
            # Keep the FULL structure on the HOST (frozen_positions_cpu + the
            # active slab) and upload ONLY the active slab to the GPU. Uploading
            # all N positions was the cause of the std::bad_alloc OOM on the 4 GB
            # GTX 1650 (positions_gpu ~ N*12 B alone exhausted VRAM before any
            # grid rebuild over the few-million active atoms could run).
            #
            # Optional resume-time window override: a smaller window keeps both
            # the slab and the grid rebuild (cp.argsort over the slab) inside
            # small-GPU VRAM. When set it becomes the operative window for the
            # whole resumed run (so the sliding-window eviction stays consistent).
            # No physics/formula change.
            _rw = getattr(self, 'resume_active_window_height', None)
            if _rw is not None and float(_rw) > 0:
                if abs(float(_rw) - self.active_window_height) > 1e-9:
                    self._log(f"[V3] Resume window override: active_window_height "
                              f"{self.active_window_height:.1f} -> {float(_rw):.1f} nm "
                              f"(resume_active_window_height)")
                self.active_window_height = float(_rw)
            win = self.active_window_height

            free0, total0 = self._gpu_free_gb()

            # Split on the HOST: active slab (z >= floor) vs frozen buried bulk.
            z_min_filter = max(0.0, self.max_height - win)
            active_mask  = pos[:, 2] >= z_min_filter
            active_host  = np.ascontiguousarray(pos[active_mask], dtype=np.float32)
            self.frozen_positions_cpu = np.ascontiguousarray(
                pos[~active_mask], dtype=np.float32)
            self.active_indices = None     # unused in active-slab-only model
            self.n_active = int(active_host.shape[0])

            self._log(f"[V3] Resume (ACTIVE-SLAB-ONLY): total atoms loaded={self.n_atoms:,} | "
                      f"active_window_height={win:.1f} nm | n_active={self.n_active:,} "
                      f"(z >= {z_min_filter:.2f} nm) | frozen(host)={self.frozen_positions_cpu.shape[0]:,}")
            if free0 is not None:
                self._log(f"[V3] GPU free BEFORE active-slab upload: "
                          f"{free0:.2f} / {total0:.2f} GB")

            # Size the GPU slab to the active set (+50% headroom for new growth).
            self.capacity = max(2 * 1024 * 1024, int(self.n_active * 1.5))

            # Resume memory preflight: slab array (12 B/atom) + one rebuild
            # transient (~52 B/active atom) must fit free VRAM. Abort cleanly
            # BEFORE CuPy throws std::bad_alloc.
            if free0 is not None:
                need_gb = (self.capacity * 12 + self.n_active * 52) / 1024**3
                if need_gb > 0.90 * free0:
                    raise MemoryError(
                        f"[RESUME-PREFLIGHT] Active slab too large for this GPU: "
                        f"slab capacity {self.capacity:,} (+ rebuild transient) needs "
                        f"~{need_gb:.2f} GB but only {free0:.2f} GB free / "
                        f"{total0:.2f} GB total. Lower resume_active_window_height.")

            if GPU_AVAILABLE:
                self.positions_gpu = cp.zeros((self.capacity, 3), dtype=cp.float32)
                if self.n_active > 0:
                    self.positions_gpu[:self.n_active] = cp.asarray(active_host)
            else:
                self.positions_gpu = np.zeros((self.capacity, 3), dtype=np.float32)
                if self.n_active > 0:
                    self.positions_gpu[:self.n_active] = active_host

            self.last_compaction_height = self.max_height
            self._grid_needs_rebuild = True
            self._n_active_at_last_rebuild = 0
            del active_host, active_mask, pos, raw
            if GPU_AVAILABLE:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass

            free1, total1 = self._gpu_free_gb()
            if free1 is not None:
                self._log(f"[V3] GPU free AFTER active-slab upload: "
                          f"{free1:.2f} / {total1:.2f} GB")
            self._log(f"[V3] positions_gpu = ACTIVE-ONLY slab "
                      f"({self.n_active:,} atoms / capacity {self.capacity:,}); "
                      f"full {self.n_atoms:,}-atom structure lives on HOST "
                      f"(frozen + slab). NOT the full structure on GPU.")
            self._log(f"[V3] Loaded: total={self.n_atoms:,} atoms, "
                      f"max_height={self.max_height:.2f} nm")
            return True
        except Exception as e:
            self._log(f"[V3] [!] Load failed: {e}")
            return False

    def _checkpoint_worker_loop(self):
        """Persistent daemon: drains _ckpt_queue and writes checkpoints to disk."""
        while True:
            item = self._ckpt_queue.get()
            if item is None:            # shutdown sentinel
                self._ckpt_queue.task_done()
                return
            pos3d, snap_n, snap_h, cfg_snap = item
            self._do_save_checkpoint(pos3d, snap_n, snap_h, cfg_snap, emergency=False)
            self._ckpt_queue.task_done()

    def save_checkpoint(self, emergency: bool = False):
        """Save VRAM → HDF5 as an (N, 4) dataset [X, Y, Z, T].

        Normal saves push a snapshot into the worker queue and return immediately
        (best-effort: silently dropped when the worker is still writing the
        previous checkpoint — the simulation loop is never blocked by disk I/O).
        Emergency saves (Ctrl+C, pause, thermal abort) drain the queue first,
        then write synchronously so data is guaranteed on disk before returning.
        """
        # Snapshot atomically: full structure = frozen(host) + active slab(GPU→CPU).
        pos3d    = self._all_positions_host()   # (N, 3) float32
        snap_n   = self.n_atoms
        snap_h   = self.max_height
        cfg_snap = dict(self.cfg)

        if emergency:
            # Drain in-flight async write, then write synchronously in this thread.
            self._ckpt_queue.join()
            self._do_save_checkpoint(pos3d, snap_n, snap_h, cfg_snap, emergency=True)
            return

        try:
            self._ckpt_queue.put_nowait((pos3d, snap_n, snap_h, cfg_snap))
            self._log(f"[V3] Checkpoint queued (async): h={snap_h:.1f} nm, {snap_n:,} atoms")
        except queue.Full:
            self._log("[V3] Checkpoint skipped — worker busy (best-effort, non-blocking)")

    def _do_save_checkpoint(self, pos3d: np.ndarray, n_atoms: int, max_height: float,
                            cfg: dict, emergency: bool = False):
        """Internal: write HDF5 to disk (runs in background or foreground)."""

        # Append virtual timestamp column — T is uniquely determined by Z
        t_col = (pos3d[:, 2] / self.growth_rate).reshape(-1, 1).astype(np.float32)
        pos4d = np.hstack([pos3d, t_col])                       # (N, 4) float32

        chk_dir  = cfg['checkpoint_dir']
        suffix   = "_A" if (n_atoms // 5000) % 2 == 0 else "_B"
        chk_name = f"checkpoint_v3{suffix}.h5" if not emergency else "checkpoint_v3_emergency.h5"
        filepath = os.path.join(chk_dir, chk_name)
        Path(chk_dir).mkdir(parents=True, exist_ok=True)
        try:
            with h5py.File(filepath, 'w', libver='latest') as f:
                f.create_dataset('positions', data=pos4d, compression='gzip', compression_opts=1, shuffle=True)
                f.attrs['current_height']      = np.float32(max_height)
                # NAMING NOTE (found 2026-08-30, not fixed to avoid breaking checkpoint-
                # format compatibility for any existing reader keyed on this exact name):
                # despite the name, this is the TOTAL atom count in `positions` (identical
                # to len(pos4d)), INCLUDING the seed_atoms below, not "atoms deposited since
                # seeding". seed_atoms + atoms_deposited will double-count the seed layer if
                # naively summed to estimate a total.
                f.attrs['atoms_deposited']     = n_atoms
                f.attrs['seed_atoms']          = int(getattr(self, 'n_seed_atoms', 0))
                f.attrs['radius']              = self._r
                f.attrs['collision_diameter']  = self._cd
                f.attrs['pitch']               = np.float32(self.pitch)
                f.attrs['growth_rate']         = np.float32(self.growth_rate)
                f.attrs['deposition_rate_nm_s'] = np.float32(cfg.get('r_dep', self.growth_rate))
                f.attrs['deposition_rate_role'] = 'nominal_or_measured_dense_equivalent_rate'
                f.attrs['growth_efficiency']   = np.float32(cfg.get('growth_efficiency', 1.0))
                f.attrs['vertical_growth_rate_nm_s'] = np.float32(self.growth_rate)
                if cfg.get('rotation_rpm') is not None:
                    f.attrs['rotation_rpm']     = np.float32(cfg['rotation_rpm'])
                    f.attrs['pitch_formula']    = 'pitch_nm_per_turn = vertical_growth_rate_nm_s * 60 / rotation_rpm'
                if cfg.get('source_distance_cm') is not None:
                    f.attrs['source_distance_cm'] = np.float32(cfg['source_distance_cm'])
                f.attrs['alpha_deg']           = np.float32(cfg.get('alpha', 85.0))
                f.attrs['alpha_convention']    = 'angle_from_substrate_normal_deg'
                f.attrs['material']            = cfg.get('material', 'Cu')
                f.attrs['temperature_K']       = np.float32(cfg.get('temperature', 300.0))
                f.attrs['melting_point_K']     = np.float32(cfg.get('melting_point_K', 1357.77))
                f.attrs['homologous_temperature_Ts_over_Tm'] = np.float32(cfg.get('homologous_temperature', 0.0))
                f.attrs['surface_diffusion_enabled'] = bool(cfg.get('enable_surface_diffusion', False))
                f.attrs['activation_energy_eV'] = np.float32(cfg.get('activation_energy', 0.0))
                f.attrs['use_physical_diffusion_model'] = bool(cfg.get('use_physical_diffusion_model', False))
                f.attrs['nu0_hz'] = np.float32(cfg.get('nu0_hz', 0.0))
                f.attrs['hop_rate_hz'] = np.float32(getattr(self, '_hop_rate_hz', 0.0))
                f.attrs['angular_distribution_enabled'] = bool(cfg.get('angular_distribution_enabled', False))
                f.attrs['angular_distribution_type'] = cfg.get('angular_distribution_type', 'none')
                f.attrs['angular_sigma_deg'] = np.float32(cfg.get('angular_sigma_deg', 0.0))
                f.attrs['angular_num_samples'] = int(cfg.get('angular_num_samples', 1))
                f.attrs['angular_seed'] = int(cfg.get('angular_seed', 0))
                f.attrs['enable_sticking']      = bool(cfg.get('enable_sticking', False))
                f.attrs['sticking_probability'] = np.float32(cfg.get('sticking_probability', 1.0))
                f.attrs['contact_geometry']    = '2R_physical_contact'
                f.attrs['schema']              = '4D_helical_v3.2'
            with open(filepath, 'a') as f_sync:
                os.fsync(f_sync.fileno())
            self._log(f"[V3] Checkpoint {'(emergency) ' if emergency else ''}saved: "
                      f"{n_atoms:,} atoms, h={max_height:.1f} nm")
        except Exception as e:
            self._log(f"[ERROR] Checkpoint save failed: {e}")

    # ------------------------------------------------------------------
    # Trajectory output
    # ------------------------------------------------------------------

    def _open_trajectory_files(self, xyz_path: str, lmp_path: str):
        self.xyz_file = open(xyz_path, 'w')
        self.lmp_file = open(lmp_path, 'w')
        self._log(f"[V3] Output: {xyz_path}  +  {lmp_path}")

    def _write_xyz_frame(self):
        """Trajectory output disabled in 100% H5 mode."""
        pass

    def _write_lammps_frame(self):
        """Trajectory output disabled in 100% H5 mode."""
        pass

    # ------------------------------------------------------------------
    # KD-Tree management
    # ------------------------------------------------------------------

    def _rebuild_tree(self):
        """Rebuild cKDTree from current positions (Only if needed for CPU)."""
        if self.n_atoms == 0:
            self._tree = None
            return
        
        # Bypass for GPU production runs to avoid GPU->CPU sync overhead
        if GPU_AVAILABLE and self._kernel is not None:
            self._tree_n = self.n_atoms
            return

        # Fetch full structure (frozen host + active slab) ONLY if we really
        # need the tree on CPU (CPU-fallback path).
        self.positions = self._all_positions_host()
        self._tree = cKDTree(self.positions[:, :2])   # 2-D (x, y)
        self._tree_n = self.n_atoms

    def _get_tree(self) -> Optional[cKDTree]:
        """Disabled in production to save RAM (OOM prophylaxis)."""
        return None

    # ------------------------------------------------------------------
    # Ray-sphere 3D intersection with cKDTree pre-filter
    # ------------------------------------------------------------------

    def _find_landing_position(self, x0: float, y0: float) -> Optional[np.ndarray]:
        """
        Trace a single ballistic ray from above the canopy (CPU fallback).
        """
        tree = self._get_tree()
        if tree is None:
            return np.array([x0, y0, self._r], dtype=np.float32)

        z_start = self.max_height + 20.0
        origin  = np.array([x0, y0, z_start], dtype=np.float32)

        bw = self.cfg['box_width']
        bd = self.cfg['box_depth']
        r_search = float(bw)

        # Query candidates (with PBC)
        all_idx = []
        for dx in [-bw, 0.0, bw]:
            for dy in [-bd, 0.0, bd]:
                try:
                    all_idx.extend(tree.query_ball_point([x0 + dx, y0 + dy], r=r_search))
                except Exception: pass
        
        all_idx = list(set(all_idx))
        if not all_idx:
            return np.array([x0, y0, self._r], dtype=np.float32)

        candidates = self.positions[all_idx]
        D = self.flux_dir.astype(np.float32)
        # PHYSICS FIX: use COLLISION_DIAMETER (2R) for center-to-center contact distance
        r2 = self._cd * self._cd
        
        t_min = 1e20
        hit = False

        for C in candidates:
            L = origin - C
            b = 2.0 * np.dot(D, L)
            c = np.dot(L, L) - r2
            disc = b*b - 4.0*c
            if disc >= 0:
                t = (-b - np.sqrt(disc)) / 2.0
                if t > 1e-4 and t < t_min:
                    t_min = t
                    hit = True
        
        if not hit:
            return None

        landing = origin + t_min * D
        landing[0] %= bw
        landing[1] %= bd
        landing[2] = max(landing[2], self._r)
        return landing.astype(np.float32)

    # ------------------------------------------------------------------
    # Surface diffusion (Arrhenius, 3D-aware)
    # ------------------------------------------------------------------

    def _surface_diffuse(self, pos: np.ndarray) -> np.ndarray:
        """
        Perform Arrhenius surface diffusion in 3D.

        The atom hops up to `diffusion_hops` times. Each hop moves in a
        random XY direction by `diffusion_radius` nm, then we check the
        local height (by querying the KDTree) and accept uphill hops
        (reinforces column growth) or penalise downhill hops.
        """
        if not self.cfg['enable_surface_diffusion']:
            return pos

        cfg = self.cfg
        tree = self._get_tree()
        if tree is None:
            return pos

        best = pos.copy()
        bw, bd = cfg['box_width'], cfg['box_depth']
        dr = cfg['diffusion_radius']

        # Find local height at current position (max Z of neighbours in dr)
        nbs = tree.query_ball_point([best[0], best[1]], r=dr, workers=1)
        best_z = best[2] if not nbs else max(best[2],
                   float(self.positions[nbs, 2].max()))

        hop_prob = self._hop_prob

        for _ in range(cfg['diffusion_hops']):
            if np.random.random() > hop_prob:
                continue  # Arrhenius rejection

            # Random hop in XY
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, dr)
            tx = (best[0] + radius * np.cos(angle)) % bw
            ty = (best[1] + radius * np.sin(angle)) % bd

            # Local height at trial position
            nbs_t = tree.query_ball_point([tx, ty], r=dr, workers=1)
            if nbs_t:
                # PHYSICS FIX: land at 2R above existing atom center (full contact distance)
                tz = float(self.positions[nbs_t, 2].max()) + float(self._cd)
            else:
                tz = float(self._r)   # first atom on bare substrate

            # Accept? uphill always, downhill with Boltzmann factor
            dz = tz - best_z
            if dz >= 0:
                accept = True
            else:
                dE = abs(dz) * 0.1    # 0.1 eV per nm (approximate)
                beta_inv = self.kB * cfg['temperature']
                accept = np.random.random() < np.exp(-dE / beta_inv)

            if accept:
                best    = np.array([tx, ty, tz], dtype=np.float32)
                best_z  = tz

        return best

    # ------------------------------------------------------------------
    # Atom deposition
    # ------------------------------------------------------------------

    def _preflight_rebuild_memory(self, n_active: int):
        """
        Print n_active + active window, then estimate whether the upcoming grid
        rebuild (gather + cp.argsort over the slab) fits in free VRAM. Abort
        cleanly (MemoryError) BEFORE CuPy throws std::bad_alloc if it does not.
        Per-atom transient model (~52 B) matches tools/analyze_resume_slab_memory.py.
        """
        self._log(f"[GRID] Rebuild preflight: n_active={int(n_active):,}, "
                  f"active_window_nm={self.active_window_height:.1f}")
        if not GPU_AVAILABLE:
            return
        try:
            free_b, total_b = cp.cuda.runtime.memGetInfo()
        except Exception:
            return
        need = int(n_active) * 52
        if need > 0.90 * free_b:
            msg = (f"[GRID-PREFLIGHT] ABORT before grid rebuild to avoid std::bad_alloc: "
                   f"rebuild over n_active={int(n_active):,} needs ~{need/1024**3:.2f} GB "
                   f"transient; only {free_b/1024**3:.2f} GB free / "
                   f"{total_b/1024**3:.2f} GB total. Lower active_window_height / "
                   f"resume_active_window_height so the active slab fits this GPU.")
            self._log(msg)
            raise MemoryError(msg)

    def deposit_batch_gpu(self, n: int = 512) -> int:
        """
        100% Vectorized GPU Deposition.
        """
        if not GPU_AVAILABLE or self._kernel is None: return self.deposit_batch_cpu(n)

        bw, bd = self.cfg['box_width'], self.cfg['box_depth']

        # 1. Sliding Window — every 10 nm, evict atoms below the slab floor from
        #    the GPU to host RAM (frozen_positions_cpu). Active slab stays bounded.
        if self.max_height - self.last_compaction_height > 10.0:
            z_min_filter = max(0.0, self.max_height - self.active_window_height)
            self._evict_buried_to_cpu(z_min_filter)

        n_active = self.n_active
        if n_active == 0:
            # No active atoms to deposit onto (should not happen post-substrate/resume).
            self.atoms_since_deposit += n
            return n

        # ── Throttled grid rebuild ────────────────────────────────────────
        # Rebuild when: first call, after eviction, or enough new atoms deposited.
        # Between rebuilds the cached sorted indices and params are reused —
        # newly added atoms won't cast shadows until next rebuild, but they
        # represent < batch_size*4 / n_active ≈ 0.5% of the slab (negligible).
        #
        # ACTIVE-SLAB-ONLY: positions_gpu[:n_active] IS the active set, so the
        # grid sorts LOCAL slab indices directly — no global gather/map needed.
        _do_rebuild = (
            self._grid_needs_rebuild
            or self.gpu_grid.grid_dims is None
            or (n_active - self._n_active_at_last_rebuild) >= self._grid_rebuild_delta
        )
        if _do_rebuild and n_active > 0:
            # Visibility + coarse memory preflight BEFORE the cp.argsort over the slab.
            self._preflight_rebuild_memory(n_active)
            self.gpu_grid.rebuild(self.positions_gpu[:n_active], self.max_height,
                                  self.active_window_height)
            # sorted_indices already index positions_gpu directly (local == global)
            self._cached_global_sorted_indices = self.gpu_grid.sorted_indices
            self._n_active_at_last_rebuild = n_active
            self._cached_grid_params = self.gpu_grid.get_params_host()
            self._grid_needs_rebuild = False
            if GPU_AVAILABLE:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass

        # 2. Ballistic Collision Kernel — use cached grid params (no GPU sync)
        nx, ny, nz, gm_x, gm_y, gm_z, csize = self._cached_grid_params

        # ── DYNAMIC HELICAL FLUX ROTATION ───────────────────────────────────
        # θ(Z) = 2π × max_height / pitch.  All atoms in this batch deposit at the
        # same instantaneous substrate rotation angle, which is physically correct
        # because a single batch covers only Δh ≪ pitch of vertical growth.
        # The result is stored back into self.flux_dir so that the CPU fallback path
        # and telemetry logging always reflect the current physical flux direction.
        self.flux_dir = self._compute_helical_flux_dir(self.max_height)
        angular_dirs = self._compute_angular_flux_samples(self.flux_dir, self.max_height)

        origins = cp.random.uniform(0, bw, (n, 2)).astype(cp.float32)
        # +2.0 nm offset (not +20.0): at α=85°, +20 nm causes t ≈ 229 nm which
        # triggers float32 catastrophic cancellation in the discriminant (b²≈4c≈210k).
        # +2.0 nm gives t ≈ 23 nm, keeping the float32 discriminant well-represented.
        origins = cp.column_stack([origins, cp.full(n, self.max_height + 2.0, dtype=cp.float32)])
        t_results = cp.full(n, 1e20, dtype=cp.float32)
        
        # ACTIVE-SLAB-ONLY: sorted_indices are LOCAL indices into positions_gpu
        # (the slab itself), so they are used directly by the kernel (cached).
        global_sorted_indices = self._cached_global_sorted_indices
        
        # Adaptive CUDA block size: 512 threads for large batches (better GTX 1650 occupancy)
        block = 512 if n >= 8192 else 256
        grid = (n + block - 1) // block
        # PHYSICS FIX: pass COLLISION_DIAMETER (2R = 2*AUTHOR_RADIUS = self._cd) as intersection radius.
        # The ray-sphere test finds where the incoming atom center touches the existing
        # atom surface, i.e. center-to-center = R_existing + R_new = 2*R = COLLISION_DIAMETER.
        if len(angular_dirs) == 1:
            flux_dir_gpu = cp.asarray(self.flux_dir, dtype=cp.float32)
            self._kernel((grid,), (block,), (origins,
                                             np.float32(self.flux_dir[0]),
                                             np.float32(self.flux_dir[1]),
                                             np.float32(self.flux_dir[2]),
                                             self.positions_gpu,
                                             self.gpu_grid.cell_starts, global_sorted_indices,
                                             np.int32(nx), np.int32(ny), np.int32(nz),
                                             np.float32(gm_x), np.float32(gm_y), np.float32(gm_z),
                                             np.float32(csize), np.int32(n), np.int32(n_active),
                                             self._cd, t_results,
                                             np.int32(self._n_active_at_last_rebuild), np.int32(n_active)))
            new_pos = origins + t_results[:, None] * flux_dir_gpu
        else:
            new_pos = cp.empty((n, 3), dtype=cp.float32)
            counts = np.full(len(angular_dirs), n // len(angular_dirs), dtype=np.int32)
            counts[: n % len(angular_dirs)] += 1
            start = 0
            for direction, count in zip(angular_dirs, counts):
                count_i = int(count)
                if count_i <= 0:
                    continue
                stop = start + count_i
                origins_i = origins[start:stop]
                t_i = t_results[start:stop]
                grid_i = (count_i + block - 1) // block
                self._kernel((grid_i,), (block,), (origins_i,
                                                   np.float32(direction[0]),
                                                   np.float32(direction[1]),
                                                   np.float32(direction[2]),
                                                   self.positions_gpu,
                                                   self.gpu_grid.cell_starts, global_sorted_indices,
                                                   np.int32(nx), np.int32(ny), np.int32(nz),
                                                   np.float32(gm_x), np.float32(gm_y), np.float32(gm_z),
                                                   np.float32(csize), np.int32(count_i), np.int32(n_active),
                                                   self._cd, t_i,
                                                   np.int32(self._n_active_at_last_rebuild), np.int32(n_active)))
                flux_i = cp.asarray(direction, dtype=cp.float32)
                new_pos[start:stop] = origins_i + t_i[:, None] * flux_i
                start = stop
	        
        # 3. Handle Missed Atoms — build mask_active (first-class state variable: Option C)
        mask_miss = t_results > 1e19
        self._last_batch_miss_count = int(cp.sum(mask_miss))
        mask_active = cp.ones(n, dtype=cp.int8)  # K_deposit: all atoms start active
        mask_active[mask_miss] = 0               # geometric misses → inactive

        # --- K_stick: state transition on mask_active only (Option C — no z mutation) ---
        if self.cfg.get('enable_sticking', False):
            s = float(self.cfg.get('sticking_probability', 1.0))
            hit_mask = mask_active.astype(cp.bool_)                  # active geometric hits only
            u = self._sticking_rng.random(size=n, dtype=cp.float32) # SEPARATE RNG stream
            reject = hit_mask & (u >= s)                             # u<s attaches; u>=s rejected
            self._last_batch_sticking_reject = int(cp.sum(reject))

            # K_reemit: additive diffuse (cosine-law) re-emission for rejected atoms,
            # instead of immediate discard. See P1_DIFFUSE_REEMISSION_MODEL_DESIGN_20260823.md.
            # `reject` is narrowed in place to only the atoms that STILL end up discarded
            # (exhausted all re-emission attempts) so the single `mask_active[reject] = 0`
            # below is the only place mask_active gets zeroed for this mechanism.
            if self.cfg.get('enable_diffuse_reemission', False) and self._reemission_kernel is not None:
                n_reject = int(cp.sum(reject))
                if n_reject > 0:
                    reject_indices = cp.where(reject)[0]
                    reject_pos = new_pos[reject_indices]
                    stuck_flags = cp.zeros(n_reject, dtype=cp.int32)
                    seed = np.random.randint(0, 1000000)
                    grid_re = (n_reject + block - 1) // block
                    # ROOT-CAUSE FIX (2026-08-24, fresh-eyes re-audit of the 0/598-restuck bug):
                    # nx,ny,nz,gm_x,gm_y,gm_z,csize,n_reject,bw,bd were being passed as bare
                    # Python int/float, NOT np.int32/np.float32, while every OTHER argument in
                    # this same call already used explicit numpy scalar types. Confirmed via an
                    # isolated minimal CuPy RawModule test that CuPy does NOT reliably marshal a
                    # bare Python scalar mixed with explicitly-typed numpy scalars into a raw
                    # kernel launch: a bare Python float argument came back read as 0 inside the
                    # kernel in that isolated test. In REEMISSION_KERNEL this corrupted
                    # `cell_size`/`grid_min` as seen by the kernel, making the neighbour-cell
                    # index computation (`gx = floor((pxw-grid_min.x)/cell_size)`) land on
                    # essentially the wrong cell every time -- confirmed directly: an
                    # instrumented standalone copy of this kernel reported n_neighbours=0 for
                    # every rejected atom with the bare-typed call, and n_neighbours=1-2 (the
                    # physically correct answer -- every rejected atom is by construction
                    # exactly collision_diameter from the neighbour that caused its rejection)
                    # once every scalar below was explicitly wrapped, with no other change.
                    # This is why re-emission never found a hit and 0/598 atoms ever re-stuck.
                    # See P1_DIFFUSE_REEMISSION_MODEL_DESIGN_20260823.md for the full trace.
                    self._reemission_kernel((grid_re,), (block,), (
                        reject_pos, stuck_flags, self.positions_gpu,
                        self.gpu_grid.cell_starts, global_sorted_indices,
                        np.int32(nx), np.int32(ny), np.int32(nz),
                        np.float32(gm_x), np.float32(gm_y), np.float32(gm_z), np.float32(csize),
                        np.float32(self.cfg['reemission_normal_shell_factor']),
                        self._cd, np.float32(s),
                        np.int32(self.cfg['max_reemission_attempts']),
                        np.float32(self.flux_dir[0]), np.float32(self.flux_dir[1]), np.float32(self.flux_dir[2]),
                        np.int32(n_reject), np.float32(bw), np.float32(bd), np.uint32(seed),
                        np.int32(self._n_active_at_last_rebuild), np.int32(n_active)))
                    new_pos[reject_indices] = reject_pos   # write back final (re-emitted or last-tried) positions
                    # DEAD-CODE REMOVAL (dev, 2026-09-03): `self._debug_last_stuck_flags` was
                    # write-only project-wide (grepped both dev and live copies, zero reads) --
                    # a leftover debug host-sync (.get()) forcing an unconditional GPU->host
                    # transfer every batch this mechanism runs, for data nothing ever consumed.
                    # `re_stuck` below already computes what's actually needed, entirely on-GPU,
                    # no sync required for correctness (CuPy default-stream ordering guarantees
                    # `re_stuck` sees the kernel's writes to stuck_flags without an explicit sync).
                    re_stuck = stuck_flags.astype(cp.bool_)
                    still_lost_indices = reject_indices[~re_stuck]
                    reject = cp.zeros(n, dtype=cp.bool_)
                    reject[still_lost_indices] = True      # narrow: only atoms that never re-stuck
                    self._last_batch_sticking_reject = int(reject.sum())
                    self._last_batch_reemission_restuck = int(cp.sum(re_stuck))
                    self._last_batch_reemission_attempted = n_reject
            mask_active[reject] = 0   # K_stick: state transition, NOT z mutation. `reject` above is
                                       # narrowed to exhausted-re-emission atoms when that mechanism ran.
        else:
            self._last_batch_sticking_reject = 0

        # 3b. Contact Relaxation Kernel — K_relax : S_active → S_active (domain-restricted)
        # Runs BEFORE the (separate, pre-existing) diffusion kernel so relaxation settles the
        # rigid ballistic contact point first; any subsequent diffusion hops start from there.
        if self.cfg.get('enable_contact_relaxation', False):
            active_bool_relax = mask_active.astype(cp.bool_)
            n_active_relax = int(active_bool_relax.sum())
            if n_active_relax > 0:
                new_pos_relax = new_pos[active_bool_relax]
                grid_r = (n_active_relax + block - 1) // block
                self._relax_kernel((grid_r,), (block,), (new_pos_relax, self.positions_gpu,
                                                            self.gpu_grid.cell_starts, global_sorted_indices,
                                                            np.int32(nx), np.int32(ny), np.int32(nz),
                                                            np.float32(gm_x), np.float32(gm_y), np.float32(gm_z),
                                                            np.float32(csize),
                                                            np.float32(self._relax_radius_nm),
                                                            np.int32(self.cfg['contact_relaxation_num_candidates']),
                                                            np.float32(self.cfg['contact_relaxation_bond_energy_eV']),
                                                            np.float32(self.cfg['contact_relaxation_coord_shell_factor']),
                                                            np.float32(self._cd),
                                                            np.float32(self.flux_dir[0]), np.float32(self.flux_dir[1]),
                                                            np.float32(self.flux_dir[2]),
                                                            np.float32(self._kB_T_relax),
                                                            np.int32(n_active_relax), np.float32(bw), np.float32(bd),
                                                            np.int32(self._n_active_at_last_rebuild), np.int32(n_active)))
                new_pos[active_bool_relax] = new_pos_relax

        # 4. Parallel Diffusion Kernel — K_diffuse : S_active → S_active (domain-restricted)
        if self.cfg['enable_surface_diffusion']:
            active_bool = mask_active.astype(cp.bool_)
            n_active_batch = int(active_bool.sum())
            if n_active_batch > 0:
                new_pos_active = new_pos[active_bool]   # restrict to S_active domain
                # D1 (2026-09-07): snapshot pre-hop positions so overlapping post-hop atoms
                # can be REVERTED (not lifted) after the kernel. new_pos[active_bool] is a
                # fresh copy (boolean fancy-index), so the kernel's in-place write below does
                # not touch new_pos itself until the explicit write-back.
                _d1_pre_hop = new_pos_active.copy() if self._diffusion_resolve_same_batch else None
                seed = np.random.randint(0, 1000000)
                grid_a = (n_active_batch + block - 1) // block

                if self.use_physical_diffusion_model:
                    # DWELL-TIME FIX (2026-08-23, post grid-rebuild-fix mechanistic audit):
                    # the old `dt_real_s` below (this BATCH's real-time duration,
                    # thickness_added_this_batch / growth_rate) was being used as the atom's
                    # ENTIRE lifetime diffusion budget, applied once (hops_batch=1) at the
                    # exact batch the atom lands, and never revisited. At tonight's
                    # box=100nm/alpha=85 scale that gave dt_real_s ~ 6e-4 s and hop_prob
                    # ~0.7% -- three orders of magnitude too small to be "how long this atom
                    # is diffusively mobile before being buried". active_window_height does
                    # NOT govern this (confirmed by direct trace: it only gates GPU-vs-host
                    # residency, never touches diffusion eligibility) -- there was no dwell
                    # concept in the code at all before this fix.
                    #
                    # tau_dwell_s: first-order physical estimate of real dwell time before an
                    # atom is structurally locked by subsequent growth = (one atomic layer of
                    # burial, `collision_diameter`) / (real vertical growth rate). THIS IS AN
                    # ASSUMPTION, NOT A LITERATURE-SOURCED NUMBER -- no GLAD-specific
                    # island-spacing/nucleation-density measurement was available in this
                    # project's corpus to derive a rigorous dwell time from real data. Anyone
                    # citing results produced with this fix should carry that caveat forward.
                    # See P1_CONTACT_RELAXATION_MODEL_DESIGN_20260823.md `S8_DIFFUSION_DWELL_TIME_FIX`.
                    tau_dwell_s = float(self._cd) / max(self.growth_rate, 1e-12)

                    # Slice tau_dwell_s into `hops_batch` discrete Bernoulli attempts inside
                    # the EXISTING kernel loop (glad_v3_core.py DIFFUSION_KERNEL, `for h in
                    # range(hops)`, unchanged) rather than one aggregate shot. Slice count =
                    # ceil(expected number of real hop events over the dwell time), so the
                    # loop can realise genuine multi-hop movement (not just a single nudge),
                    # capped for GPU-cost predictability. Per-slice dt and hop_prob are chosen
                    # so the CUMULATIVE probability of >=1 hop across all slices reproduces
                    # the same physically-correct aggregate 1-exp(-hop_rate*tau_dwell_s) as
                    # before (Poisson-process identity: slicing a rate process into N
                    # independent sub-intervals of dt=tau/N each with per-slice probability
                    # 1-exp(-rate*dt) preserves the total-window probability exactly,
                    # regardless of N) -- this is not a new physical assumption on top of
                    # tau_dwell_s, only a finer, multi-hop-capable discretisation of it.
                    expected_hops = self._hop_rate_hz * tau_dwell_s
                    hops_cap = int(self.cfg.get('diffusion_dwell_hops_cap', 200))
                    hops_batch = max(1, min(hops_cap, int(np.ceil(expected_hops)) if expected_hops > 0 else 1))
                    dt_slice_s = tau_dwell_s / hops_batch
                    hop_prob_batch = 1.0 - float(np.exp(-self._hop_rate_hz * dt_slice_s))
                else:
                    hop_prob_batch = float(self._hop_prob)
                    hops_batch = int(self.cfg['diffusion_hops'])

                _diag_out = cp.zeros(n_active_batch, dtype=cp.int32)  # TEMP DIAGNOSTIC
                _diag_vals = cp.zeros(n_active_batch * 6, dtype=cp.float32)  # TEMP DIAGNOSTIC
                self._diff_kernel((grid_a,), (block,), (new_pos_active, self.positions_gpu,
                                                          self.gpu_grid.cell_starts, global_sorted_indices,
                                                          np.int32(nx), np.int32(ny), np.int32(nz),
                                                          np.float32(gm_x), np.float32(gm_y), np.float32(gm_z),
                                                          np.float32(csize),
                                                          np.float32(self.cfg['diffusion_radius']), np.int32(hops_batch),
                                                          np.float32(hop_prob_batch), np.float32(self._kB_T),
                                                          np.float32(self._r), np.float32(self._cd),
                                                          np.int32(n_active_batch), np.float32(bw), np.float32(bd), np.uint32(seed),
                                                          _diag_out, _diag_vals,
                                                          np.int32(1 if self.use_eam_neighbor_barrier else 0),
                                                          np.float32(self.eam_e0_eV),
                                                          np.float32(self.eam_dNN_eV),
                                                          np.float32(self.eam_dNNN_eV),
                                                          np.float32(self.eam_nn_cutoff_nm),
                                                          np.float32(self.eam_nnn_cutoff_nm),
                                                          np.int32(self._n_active_at_last_rebuild), np.int32(n_active)))
                # DEAD-CODE REMOVAL (dev, 2026-09-03): both `self._debug_last_diffusion_flags`
                # and `self._debug_last_diffusion_vals` were write-only project-wide (grepped
                # both dev and live copies, zero reads) -- two more unconditional GPU->host
                # syncs (.get()) every diffusion-enabled batch, for data nothing ever consumed.
                # `_diag_out`/`_diag_vals` themselves are left untouched above (still passed
                # into the kernel as real out-parameters) -- only the dead host-copy is removed.

                # --- D1 (2026-09-07): diffusion same-batch-blindness overlap resolution ---
                # The kernel above ran one parallel thread per diffusing atom against the
                # frozen pre-batch snapshot; no thread saw another thread's in-flight hop, so
                # two same-batch atoms can land < collision_diameter apart unnoticed
                # (empirically 97.8-100 % of genuine overlaps are intra-batch --
                # D1_DIFFUSION_KERNEL_CONTACT_RECHECK_DESIGN_20260907.md). Resolve here,
                # deterministically, in deposition (index) order: any post-hop atom within cd
                # of an already-accepted atom -- an established structure atom, or an
                # earlier-index same-batch atom at its accepted position -- is REVERTED to its
                # pre-hop position (that atom simply did not hop this batch). Revert, not
                # lift: raising z to a contact floor would apply an uphill displacement the
                # kernel's own dz>=0||Boltzmann acceptance never tested and would re-introduce
                # the height ratchet, biasing void fraction. O(n_batch * n_batch) + one
                # KD-tree over a thin near-front slab of the structure; n_batch ~= 512.
                if self._diffusion_resolve_same_batch:
                    try:
                        # Two arrays, kept strictly separate (this is what the first attempt
                        # got wrong -- it wrote wrapped coords into the write-back, so
                        # non-hopping atoms the kernel leaves UNwrapped came out wrapped, and
                        # reverted atoms came out unwrapped -> a mixed batch that wrecks the
                        # grid at box 100):
                        #   _wb  = the WRITE-BACK. Never wrapped/clipped. Reverted rows take
                        #          the raw pre-hop position (== "this atom did not hop", which
                        #          is exactly what the kernel does to a non-hopping atom).
                        #   _q   = a QUERY-ONLY copy, x/y wrapped into [0,box), z shifted >=0,
                        #          used solely to find overlaps via a periodic KD-tree.
                        _post = cp.asnumpy(new_pos_active).astype(np.float64)
                        _pre  = cp.asnumpy(_d1_pre_hop).astype(np.float64)
                        _movd = np.any(_post[:, :3] != _pre[:, :3], axis=1)
                        _nrev = 0
                        if _movd.any():
                            _cd2 = (float(self._cd) - 1e-4) ** 2
                            _cdq = float(self._cd) - 1e-4
                            _bwf, _bdf = float(bw), float(bd)
                            _wb = _post.copy()                       # write-back, untouched
                            _est_band = self.positions_gpu[:n_active]
                            _zref = float(_post[:, 2].min()) - float(self._cd) - 0.01
                            _est_band = _est_band[_est_band[:, 2] >= _zref]
                            _est = cp.asnumpy(_est_band[:, :3]).astype(np.float64)
                            # common z shift so every query point has z >= 1 (periodic tree
                            # needs all coords in [0, boxsize); z axis is a non-periodic dummy)
                            _zoff = min(_post[:, 2].min(), _est[:, 2].min() if len(_est) else 0.0) - 1.0
                            def _wrapq(a):
                                w = np.empty_like(a)
                                w[:, 0] = np.mod(a[:, 0], _bwf)
                                w[:, 1] = np.mod(a[:, 1], _bdf)
                                w[:, 2] = a[:, 2] - _zoff
                                return w
                            _q = _wrapq(_post)
                            _zmax = max(_q[:, 2].max(), (_wrapq(_est)[:, 2].max() if len(_est) else 0.0)) + 2.0
                            _est_tree = cKDTree(_wrapq(_est), boxsize=[_bwf, _bdf, _zmax]) if len(_est) else None
                            _acc_pts = _q.copy()                     # accepted query positions, index order
                            for _j in np.nonzero(_movd)[0]:
                                _p = _q[_j]
                                _hit = (_est_tree is not None
                                        and len(_est_tree.query_ball_point(_p, _cdq)) > 0)
                                if not _hit and _j > 0:
                                    _dx = _acc_pts[:_j, 0] - _p[0]; _dx -= _bwf * np.round(_dx / _bwf)
                                    _dy = _acc_pts[:_j, 1] - _p[1]; _dy -= _bdf * np.round(_dy / _bdf)
                                    _dz = _acc_pts[:_j, 2] - _p[2]
                                    _hit = bool(np.any(_dx * _dx + _dy * _dy + _dz * _dz < _cd2))
                                if _hit:
                                    _wb[_j] = _pre[_j]              # revert: raw pre-hop, NOT wrapped
                                    _acc_pts[_j] = _wrapq(_pre[_j:_j + 1])[0]
                                    _nrev += 1
                            if _nrev:
                                new_pos_active = cp.asarray(_wb.astype(np.float32))
                        self._d1_reverts_last_batch = _nrev
                        self._d1_reverts_total += _nrev
                    except Exception as _d1_exc:
                        self._log(f"[D1][WARN] same-batch overlap resolution skipped this "
                                  f"batch ({type(_d1_exc).__name__}: {_d1_exc}); "
                                  f"positions left as the kernel produced them")

                new_pos[active_bool] = new_pos_active   # write back to full array

        # 5. Direct Insertion — only insert active atoms (state-based, not z-based)
        valid_mask = mask_active.astype(cp.bool_)
        valid_pos  = new_pos[valid_mask]
        n_valid    = int(valid_mask.sum())

        if n_valid > 0:
            # ACTIVE-SLAB-ONLY: new atoms append to the GPU slab at [n_active:].
            # The slab capacity (not the full structure) is what must fit on GPU.
            if self.n_active + n_valid > self.capacity:
                new_cap = max(int(self.capacity * 1.5), self.n_active + n_valid)
                self._log(f"[VRAM] Slab capacity expansion: {self.capacity:,} → {new_cap:,} "
                          f"slab atoms (~{new_cap * 12 / 1024**2:.0f} MB VRAM for slab positions)")
                old = self.positions_gpu
                self.positions_gpu = cp.zeros((new_cap, 3), dtype=cp.float32)
                self.positions_gpu[:self.n_active] = old[:self.n_active]
                del old
                self.capacity = new_cap
                if GPU_AVAILABLE:
                    try:
                        cp.get_default_memory_pool().free_all_blocks()
                    except Exception:
                        pass

            self.positions_gpu[self.n_active : self.n_active + n_valid] = valid_pos
            self.n_active += n_valid
            self.n_atoms  += n_valid

            # Optimized height update: inspect only the new valid batch
            batch_max_h = float(valid_pos[:, 2].max())
            if batch_max_h > self.max_height:
                self.max_height = batch_max_h

        self.atoms_since_deposit += n
        
        return n

    def deposit_batch(self, n: int = 512) -> int:
        if GPU_AVAILABLE and self._kernel: return self.deposit_batch_gpu(n)
        return self.deposit_batch_cpu(n)

    def deposit_batch_cpu(self, n: int = 512) -> int:
        # Simplified fallback for small tests only
        bw, bd = self.cfg['box_width'], self.cfg['box_depth']
        x0s = np.random.uniform(0, bw, n)
        y0s = np.random.uniform(0, bd, n)
        nominal_flux_dir = self._compute_helical_flux_dir(self.max_height)
        angular_dirs = self._compute_angular_flux_samples(nominal_flux_dir, self.max_height)
        for i in range(n):
            self.flux_dir = angular_dirs[i % len(angular_dirs)]
            pos = self._find_landing_position(x0s[i], y0s[i])
            if pos is not None:
                # Diffusion handled poorly in fallback (skip for speed).
                # Active-slab-only: append to the slab at [n_active].
                idx = self.n_active
                if idx >= self.capacity: break
                self.positions_gpu[idx] = (cp.asarray(pos) if GPU_AVAILABLE else pos)
                self.n_active += 1
                self.n_atoms  += 1
        return n

    # ------------------------------------------------------------------
    # Substrate initialisation (flat sphere layer)
    # ------------------------------------------------------------------

    def initialise_substrate(self):
        """
        Create a flat monolayer of touching spheres as substrate seed.
        Equivalent to the cube-seed in vpython code.py.
        """
        bw = self.cfg['box_width']
        bd = self.cfg['box_depth']
        r  = self._r
        spacing = float(self.cfg.get('substrate_spacing', 2.0 * r))

        xs = np.arange(r, bw, spacing)
        ys = np.arange(r, bd, spacing)
        XX, YY = np.meshgrid(xs, ys)
        XF = XX.flatten().astype(np.float32)
        YF = YY.flatten().astype(np.float32)
        ZF = np.full_like(XF, r)

        self.positions   = np.column_stack([XF, YF, ZF]).astype(np.float32)
        self.n_atoms     = len(self.positions)
        self.n_seed_atoms = self.n_atoms
        self.max_height  = float(r)

        # ACTIVE-SLAB-ONLY: the substrate IS the active slab. Nothing frozen yet.
        self.frozen_positions_cpu = np.empty((0, 3), dtype=np.float32)
        self.active_indices = None
        if self.n_atoms > self.capacity:
            self.capacity = int(self.n_atoms * 1.5)
            self.positions_gpu = (cp.zeros((self.capacity, 3), dtype=cp.float32)
                                  if GPU_AVAILABLE
                                  else np.zeros((self.capacity, 3), dtype=np.float32))
        if GPU_AVAILABLE:
            self.positions_gpu[:self.n_atoms] = cp.asarray(self.positions)
        else:
            self.positions_gpu[:self.n_atoms] = self.positions
        self.n_active = self.n_atoms
        self.last_compaction_height = self.max_height
        self._grid_needs_rebuild = True

        self._rebuild_tree()
        self._log(f"[V3] Substrate: {self.n_atoms:,} seed spheres "
                  f"({bw:.0f}×{bd:.0f} nm, spacing={spacing} nm)")

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    def run_simulation(self, n_particles: Optional[int] = None,
                       xyz_path: str = 'trajectory.xyz',
                       lmp_path: str = 'high_fidelity_traj.lammpstrj'):
        """
        Main deposition loop.

        Parameters
        ----------
        n_particles : total number of atoms to attempt; if None, run until
                      target_height is reached.
        xyz_path    : output .xyz file path
        lmp_path    : output LAMMPS trajectory path (for glad_analytics.py)
        """
        # self._open_trajectory_files(xyz_path, lmp_path) # Disabled in 100% H5 Strategy
        target = self.cfg['target_height']
        n_total = n_particles if n_particles else int(1e9)

        self._log("=" * 65)
        self._log(f"[V3] SIMULATION START")
        self._log(f"     Target height : {target:.0f} nm")
        self._log(f"     Initial atoms : {self.n_atoms:,}")
        self._log(f"     Batch size    : {self.cfg.get('batch_size', 512)}")
        self._log("=" * 65)

        attempted   = 0
        t_last_log  = time.time()
        t_last_ckpt = time.time()
        last_height_ckpt = self.last_checkpoint_height
        batch_counter = 0
        checkpoint_particle_counter = 0
        # Telemetry: print CORE_HEIGHT every N batches (not every single batch).
        # Tune via config key 'telemetry_interval' (default 10).
        _telemetry_interval = max(1, int(self.cfg.get('telemetry_interval', 10)))

        while attempted < n_total and self.max_height < target:
            # --- PHASE 0: PARTICLE LIMIT ---
            if self.cfg.get('max_particles') and attempted >= self.cfg['max_particles']:
                self._log(f"[LIMIT] Particle limit reached ({attempted:,}). Stopping.")
                break
            # --- PHASE 1: THERMAL SAFETY (throttled sampler — ~0 ms overhead) ---
            temp = self._thermal_monitor.temperature()
            throttle_temp = float(self.cfg.get('thermal_throttle_temp', 84))
            critical_temp = float(self.cfg.get('thermal_critical_temp', 87))
            if temp > throttle_temp:
                self._log(f"[THERMAL] High temp ({temp}°C). Throttling 10s...")
                time.sleep(10)
            if temp > critical_temp:
                self._log(f"[CRITICAL] Emergency Stop ({temp}°C). Saving...")
                self._rebuild_tree()
                self.save_checkpoint(emergency=True)
                sys.exit(1)

            # --- PHASE 2: PAUSE SIGNAL ---
            if os.path.exists(_PAUSE_FILE):
                self._log(f"[SIGNAL] Pause detected ({_PAUSE_FILE}). Saving checkpoint ...")
                self._rebuild_tree()
                self.save_checkpoint(emergency=True)   # synchronous — must be on disk before exit
                try:
                    os.remove(_PAUSE_FILE)
                except OSError:
                    pass
                self._log("[SIGNAL] Safe pause complete. Exiting.")
                return

            batch_size = self.cfg.get('batch_size', 512)
            batch_n = min(batch_size, n_total - attempted)
            
            t_batch_start = time.time()
            deposited_this_batch = self.deposit_batch(batch_n)
            t_batch_elapsed = time.time() - t_batch_start
            
            attempted += batch_n
            batch_counter += 1
            checkpoint_particle_counter += batch_n
            
            # --- PHASE 3: ORCHESTRATOR TELEMETRY ---
            batch_speed = batch_n / t_batch_elapsed if t_batch_elapsed > 0 else 0
            n_miss = self._last_batch_miss_count
            miss_rate = n_miss / batch_n if batch_n > 0 else 0.0

            # Miss-rate alarm: if > 99% of rays miss for 100 consecutive batches,
            # the flux direction or geometry is catastrophically wrong → stop now.
            if miss_rate > 0.99:
                self._consecutive_high_miss += 1
                if self._consecutive_high_miss >= 100:
                    self._log(
                        f"[FATAL] {self._consecutive_high_miss} consecutive batches with "
                        f">99% ray misses (miss_rate={miss_rate:.3f}). "
                        "Check flux direction, alpha convention, and ray-origin offset. "
                        "Saving emergency checkpoint and aborting."
                    )
                    self.save_checkpoint(emergency=True)
                    self.log_fh.close()
                    sys.exit(2)
            else:
                self._consecutive_high_miss = 0

            if batch_counter % _telemetry_interval == 0:
                nx_t, ny_t, nz_t = self.gpu_grid.grid_dims if self.gpu_grid.grid_dims else (1, 1, 1)
                grid_cells = nx_t * ny_t * nz_t
                vram_used_mb = 0
                if GPU_AVAILABLE:
                    try:
                        free_b, total_b = cp.cuda.runtime.memGetInfo()
                        vram_used_mb = (total_b - free_b) // (1024 ** 2)
                    except Exception:
                        pass
                print(
                    f"CORE_HEIGHT: {self.max_height:.6f} | N_TOTAL: {self.n_atoms} | "
                    f"GRID_CELLS: {grid_cells} | MISS_RATE: {miss_rate:.3f} | "
                    f"BATCH_SPEED: {batch_speed:.1f} | GPU_TEMP: {temp} | "
                    f"VRAM_MB: {vram_used_mb}",
                    flush=True
                )

            # Rebuild grid is handled INSIDE deposit_batch_gpu
            # Rebuild tree (CPU) is moved to 10k atoms frequency to avoid bottleneck
            if (self.n_atoms - self._tree_n) >= 10000:
                self._rebuild_tree() 

            # --- Progress log every 10 s ---
            now = time.time()
            if now - t_last_log >= 10.0:
                elapsed  = now - self.start_time
                progress = min(100.0, self.max_height / target * 100)
                speed    = attempted / elapsed if elapsed > 0 else 0
                eta_s    = (target - self.max_height) / max(self.max_height, 0.001) * elapsed
                eta_str  = str(timedelta(seconds=int(eta_s)))
                self._log(
                    f"[V3] {progress:.1f}%  h={self.max_height:.2f}/{target:.0f} nm  "
                    f"N={self.n_atoms:,}  speed={speed:.0f} rays/s  ETA={eta_str}"
                )
                t_last_log = now

            # --- Trajectory frame every `write_interval` batch ---
            if batch_counter % self.write_interval == 0:
                self._write_xyz_frame()
                self._write_lammps_frame()
                self._frame_idx += 1

            # --- Checkpoint every `checkpoint_height_interval` nm ---------
            h_delta = self.max_height - last_height_ckpt
            if h_delta >= self.cfg['checkpoint_height_interval']:
                self.save_checkpoint()
                last_height_ckpt = self.max_height

            # --- Also checkpoint by time (fixed key — not nested in a sub-dict) ---
            if now - t_last_ckpt >= self.cfg.get('checkpoint_time_interval', 600.0):
                self.save_checkpoint()
                t_last_ckpt = now

            # --- Also checkpoint by particle count ---
            if checkpoint_particle_counter >= self.cfg.get('checkpoint_particle_interval', 20_000_000):
                self.save_checkpoint()
                checkpoint_particle_counter = 0

        # Final save (synchronous — must be on disk before reporting complete)
        self._write_xyz_frame()
        self._write_lammps_frame()
        self.save_checkpoint(emergency=True)   # drains queue + writes synchronously
        self._log(f"[V3] SIMULATION END. Final height = {self.max_height:.2f} nm, Atoms = {self.n_atoms:,}")

        elapsed = time.time() - self.start_time
        self._log("=" * 65)
        self._log(f"[V3] SIMULATION COMPLETE")
        self._log(f"     Final height  : {self.max_height:.2f} nm")
        self._log(f"     Total atoms   : {self.n_atoms:,}")
        self._log(f"     Rays attempted: {attempted:,}")
        self._log(f"     Elapsed time  : {timedelta(seconds=int(elapsed))}")
        self._log("=" * 65)

        # --- PHYSICS VALIDATION GATE ---
        val = self.validate_physics()
        if val.get('status') == 'FAIL':
            self._log("[WARNING] Physics validation FAILED. "
                      "Results saved but marked RESEARCH_REQUIRED. "
                      "Review contact geometry (COLLISION_DIAMETER), alpha convention, "
                      "and ray-origin offset before using this data for PINN training.")

        # Close files
        if self.xyz_file:
            self.xyz_file.close()
        if self.lmp_file:
            self.lmp_file.close()
        # --- PHASE 4: OPTIONAL POST-PROCESSING ---
        if self.max_height >= target and self.cfg.get('auto_postprocess', False):
            self._log("[SHIELD] Target reached. Triggering post-processing sequence...")
            scripts = ["export_final_results.py", "extract_z_map.py", "final_analytics.py"]
            script_dir = os.path.dirname(os.path.abspath(__file__))
            for s in scripts:
                spath = os.path.join(script_dir, s)
                if os.path.exists(spath):
                    self._log(f"[AUTO] Launching {s}...")
                    try:
                        subprocess.Popen([sys.executable, spath], cwd=script_dir)
                    except Exception as e:
                        self._log(f"[ERROR] Could not start {s}: {e}")
                else:
                    self._log(f"[WARNING] Script {s} not found at {spath}")
        elif self.max_height >= target:
            self._log("[POST] Target reached. Auto post-processing disabled; run validated exporters manually.")

        self.log_fh.close()

    # ------------------------------------------------------------------
    # Physics Validation Gate (run at end of simulation)
    # ------------------------------------------------------------------

    def validate_physics(self) -> dict:
        """
        Measure column tilt angle beta and compare to GLAD analytical estimates.
        Also measures porosity and pitch (if helical data available).

        Tangent rule: tan(beta) = tan(alpha) / 2.
        Tait-style estimate: beta = alpha - asin((1 - cos(alpha)) / 2).
        Here alpha and beta are angles from the substrate normal.

        Returns a dict with measured vs expected values and pass/fail status.
        Gate thresholds:
          - beta error < 15 deg  (generous for noisy atomistic data)
          - porosity is reported as a bead-model diagnostic unless
            require_porosity_gate=True is set in the config
        """
        self._log("[VALIDATE] Running physics validation gate ...")
        results = {}

        alpha_deg = float(self.cfg.get('alpha', 85.0))
        alpha_rad = np.radians(alpha_deg)
        beta_tangent_deg = np.degrees(np.arctan(np.tan(alpha_rad) / 2.0))
        beta_tait_deg = np.degrees(alpha_rad - np.arcsin((1.0 - np.cos(alpha_rad)) / 2.0))
        results['alpha_deg']      = alpha_deg
        results['beta_tangent_deg'] = beta_tangent_deg
        results['beta_tait_deg']  = beta_tait_deg
        if alpha_deg > 60.0:
            results['angle_rule_status'] = 'QUALITATIVE_HIGH_GLANCING_ANGLE'
            self._log("[VALIDATE] Angle rules  : tangent/Tait are qualitative references "
                      f"at alpha={alpha_deg:.1f} deg; material trapping, diffusion, "
                      "and flux spread can shift beta.")

        # Need positions on CPU
        if self.n_atoms < 100:
            self._log("[VALIDATE] Too few atoms to validate.")
            results['status'] = 'INSUFFICIENT_DATA'
            return results

        pos = self._all_positions_host()  # (N, 3) float32 — frozen(host) + active slab

        # ── Column tilt via linear regression of z vs x ────────────────────
        # For a tilted column, X increases linearly with Z: X = X0 + Z * tan(beta)
        # We estimate the mean drift: dx/dz = tan(beta_measured)
        # Use only atoms above 10% of max height (exclude seed layer)
        turns_observed = self.max_height / self.pitch if self.pitch > 0 else 0.0
        results['turns_observed'] = turns_observed
        # 2026-07-07: was `self.pitch < 1.0e11 and turns_observed > 0.25`. The extra
        # turns_observed>0.25 clause let the straight X-vs-Z regression run whenever a
        # rotating (finite-pitch) deposit was still under a quarter turn -- but the
        # tangent/Tait formulas compared against are only valid for a NON-rotating beam
        # (self.pitch >= 1e11 sentinel). Any real rotation, even <0.25 turns, already
        # curves the trajectory enough to make a straight-line fit meaningless: the P1
        # h=25nm/pitch=150nm sub-series (turns_observed=0.167 for all 4 box sizes) gave
        # beta_measured=55.8/17.3/69.3/48.0 deg (L=150/250/350/450) -- a 52deg spread at
        # IDENTICAL alpha/pitch/height, varying only with box width, which has no
        # physical mechanism to change the mean column tilt this much. That's noise from
        # fitting a straight line to a partially-rotated spiral, not a real per-L tilt
        # effect -- it produced one spurious FAIL (L=250, nearest_error=40.58deg) and
        # three spuriously-lucky PASSes (L=150/350/450, which landed within 15deg of a
        # reference by chance, not because the measurement was valid). Any run with a
        # genuinely rotating substrate (finite pitch) should skip this check regardless
        # of turns_observed, same as the already-correct h50nm/h100nm runs in this
        # campaign (turns=0.33/0.67, already > the old 0.25 cutoff).
        is_rotating_helical = self.pitch < 1.0e11
        if is_rotating_helical:
            results['beta_measured_deg'] = None
            results['beta_pass'] = None
            results['beta_status'] = 'SKIPPED_HELICAL'
            self._log("[VALIDATE] Column tilt : skipped direct X-vs-Z regression "
                      f"because this run is helical ({turns_observed:.2f} turns). "
                      "Use the straight benchmark for beta, or an unwrapped helix validator.")
        else:
            z_min_frac = 0.1 * self.max_height
            mask_upper = pos[:, 2] > z_min_frac
            if mask_upper.sum() < 50:
                self._log("[VALIDATE] Insufficient atoms above 10% height for tilt measurement.")
                results['status'] = 'INSUFFICIENT_DATA'
                return results

            pos_upper = pos[mask_upper]
            # Bin by Z into 20 slices, compute mean X per slice.
            z_vals = pos_upper[:, 2]
            x_vals = pos_upper[:, 0]
            z_bins = np.linspace(z_vals.min(), z_vals.max(), 21)
            z_centers, x_means = [], []
            for i in range(20):
                in_bin = (z_vals >= z_bins[i]) & (z_vals < z_bins[i+1])
                if in_bin.sum() > 10:
                    z_centers.append((z_bins[i] + z_bins[i+1]) / 2.0)
                    x_means.append(x_vals[in_bin].mean())

            if len(z_centers) >= 3:
                z_arr = np.array(z_centers)
                x_arr = np.array(x_means)
                # Linear fit: x = a + b*z  =>  b = tan(beta)
                b, a = np.polyfit(z_arr, x_arr, 1)
                beta_measured_deg = np.degrees(np.arctan(abs(b)))
                results['beta_measured_deg'] = beta_measured_deg
                results['beta_tangent_deg']  = beta_tangent_deg
                results['beta_tait_deg']     = beta_tait_deg
                beta_err_tangent = abs(beta_measured_deg - beta_tangent_deg)
                beta_err_tait = abs(beta_measured_deg - beta_tait_deg)
                results['beta_error_tangent_deg'] = beta_err_tangent
                results['beta_error_tait_deg'] = beta_err_tait
                results['beta_error_deg'] = min(beta_err_tangent, beta_err_tait)
                beta_pass = bool(results['beta_error_deg'] < 15.0)
                results['beta_pass'] = beta_pass
                self._log(f"[VALIDATE] Column tilt : beta_measured={beta_measured_deg:.2f} deg "
                          f"| beta_tangent={beta_tangent_deg:.2f} deg "
                          f"| beta_Tait={beta_tait_deg:.2f} deg "
                          f"| nearest_error={results['beta_error_deg']:.2f} deg "
                          f"| {'PASS' if beta_pass else 'WARN'} (informational only, see note below)")
            else:
                results['beta_measured_deg'] = None
                self._log("[VALIDATE] Not enough Z-bins for tilt regression.")

        # ── Porosity estimate ──────────────────────────────────────────────
        # Porosity = 1 - (volume occupied by deposited atoms) / (total box volume)
        # Exclude the artificial substrate seed from the film porosity estimate.
        # Approximate: each deposited atom occupies a sphere of radius AUTHOR_RADIUS.
        n_film_atoms = max(0, self.n_atoms - int(getattr(self, 'n_seed_atoms', 0)))
        results['film_atoms'] = n_film_atoms
        growth_pass = bool(n_film_atoms > 0 and self.max_height > (2.0 * float(self._r)))
        results['growth_pass'] = growth_pass
        atom_volume = (4.0/3.0) * np.pi * float(self._r)**3
        box_volume  = self.cfg['box_width'] * self.cfg['box_depth'] * self.max_height
        if box_volume > 0:
            porosity = 1.0 - (n_film_atoms * atom_volume) / box_volume
            results['porosity'] = porosity
            por_pass = bool(0.55 <= porosity <= 0.92)
            results['porosity_pass'] = por_pass
            require_porosity_gate = bool(self.cfg.get('require_porosity_gate', False))
            results['porosity_gate_required'] = require_porosity_gate
            por_status = 'PASS' if por_pass else ('FAIL' if require_porosity_gate else 'WARN')
            self._log(f"[VALIDATE] Porosity      : {100*porosity:.1f}% "
                      f"| bead-sphere estimate, calibration needed "
                      f"| {por_status}")

        # ── Overall gate ───────────────────────────────────────────────────
        # 2026-08-27: beta_pass/beta_error_deg is DELIBERATELY EXCLUDED from the pass/fail
        # gate (it used to gate via beta_ok). Root cause: this method fits a single global
        # straight line to mean-X-per-Z-bin across ALL atoms/columns. On real multi-column
        # growth, different columns drift laterally in different directions (seed placement,
        # PBC wrapping, coalescence), so averaging X across columns at a given Z cancels out
        # most of the true per-column tilt instead of measuring it -- the more real columns,
        # the worse the cancellation. This has now produced a wrong near-0deg/near-90deg
        # reading (vs. a trusted phase-correlation cross-check on the voxelized density grid,
        # 05_INVERSE_FRAMEWORK/observation/extract_observables.py:_compute_column_tilt_phase)
        # in every documented run at alpha=85 with diffusion/sticking/reemission mechanisms
        # active: box=100/150/200 FLOORFIX runs (internal 2.12/1.16/7.32deg vs trusted
        # 81.60/82.16/87.47deg), the sticking+reemission-only ablation (0.31 vs 65.41deg), and
        # the combined sticking+reemission+diffusion campaign (3.43 vs 81.54deg) -- see
        # SESSION_LOG.md 2026-08-26/2026-08-27 entries for the full trail. It has never once
        # produced a correct number in that regime, so treating it as a hard gate produces
        # chronic false FAILs / false RESEARCH_REQUIRED on scientifically usable runs. The
        # number is still computed and logged above for visibility (a real bug producing an
        # exact 0deg/NaN would still show up for a human to notice), but a mismatch here no
        # longer fails the run. glad_v3_core.py (engine layer) is architecturally forbidden
        # from importing the trusted observation-layer method directly (see
        # 05_INVERSE_FRAMEWORK/EXECUTION_CONTEXT.json layer_imports: engine cannot_import
        # observation, and vice versa) -- so any beta-mismatch flagged here must be manually
        # cross-checked post-hoc with the trusted method before being treated as a real
        # result; see PROTOCOLS/09_VALIDATION_PROTOCOL.md for that checklist step.
        beta_ok = True
        porosity_ok = results.get('porosity_pass', False) or not bool(results.get('porosity_gate_required', False))
        passed = beta_ok and growth_pass and porosity_ok
        if passed and results.get('beta_status') == 'SKIPPED_HELICAL':
            results['status'] = 'PASS_HELICAL_GEOMETRY_ONLY'
        else:
            results['status'] = 'PASS' if passed else 'FAIL'
        self._log(f"[VALIDATE] Overall gate  : {results['status']}")
        if not passed:
            self._log("[VALIDATE] RESEARCH_REQUIRED — check contact geometry, alpha convention, and flux offset.")
        elif results.get('porosity_pass') is False:
            self._log("[VALIDATE] Porosity is outside the uncalibrated bead-model reference window; "
                      "accepted as report-only, not a hard physics gate.")
        if results.get('beta_pass') is False:
            self._log("[VALIDATE] NOTE: beta mismatch above is informational only (not gating) -- "
                      "cross-check with the trusted phase-correlation method on the checkpoint "
                      "before treating it as a real result. See PROTOCOLS/09_VALIDATION_PROTOCOL.md.")
        return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GLAD V3.0 — Full 3D Bead-Spring Simulator"
    )
    parser.add_argument(
        '--config', default='glad_config.yaml',
        help='YAML configuration file (default: glad_config.yaml)'
    )
    parser.add_argument(
        '--load_v3', default=None,
        help='Load a V3 HDF5 checkpoint (e.g. checkpoints/checkpoint_V3_READY.h5)'
    )
    parser.add_argument(
        '--n_particles', type=int, default=None,
        help='Max particles to attempt (default: run until target_height)'
    )
    parser.add_argument(
        '--kdtree_interval', type=int, default=None,
        help='Atoms between cKDTree rebuilds'
    )
    parser.add_argument(
        '--target_height', type=float, default=None,
        help='Override target height (nm)'
    )
    parser.add_argument(
        '--force_dynamic', action='store_true',
        help='Ignore checkpoint height and recalculate from VRAM'
    )
    parser.add_argument(
        '--out', default='trajectory.xyz',
        help='Output .xyz trajectory path (default: trajectory.xyz)'
    )
    parser.add_argument(
        '--lmp', default='high_fidelity_traj.lammpstrj',
        help='Output LAMMPS trajectory path'
    )
    parser.add_argument(
        '--fresh', action='store_true',
        help='Ignore existing checkpoint and start fresh (re-initialise substrate)'
    )
    parser.add_argument(
        '--allow_legacy_resume', action='store_true',
        help='Allow loading pre-v3.2 checkpoints only for forensic comparison'
    )
    parser.add_argument(
        '--batch_size', type=int, default=None,
        help='Override batch size'
    )
    parser.add_argument(
        '--write_interval', type=int, default=None,
        help='Override write interval (batches)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--random_seed', type=int, default=None,
        help='RNG seed (int>0). Overrides YAML simulation.random_seed. '
             'If unset (or 0), RNGs use OS entropy (independent but not reproducible).'
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  GLAD V3.0 — Full 3D Bead-Spring Simulator")
    print("=" * 65)

    # Load config
    cfg = load_config(args.config)
    if args.batch_size:
        cfg['batch_size'] = args.batch_size
    if args.write_interval:
        cfg['write_interval'] = args.write_interval
    if args.verbose:
        cfg['verbose'] = True
    if args.allow_legacy_resume:
        cfg['allow_legacy_resume'] = True

    # ── RNG SEED PROPAGATION (added for publication reproducibility) ─────
    # YAML key:  simulation.random_seed
    # CLI override: --random_seed
    # Behaviour:
    #   - if explicit seed (any int, including 0): seed Python random, NumPy, CuPy
    #     → deterministic and reproducible (0 is a valid, explicit seed value)
    #   - if None (key absent from both CLI and YAML): leave RNGs unseeded
    #     → runs use OS entropy (non-reproducible) — distinct realizations each run
    #
    # RESUME-STAGE SEED OFFSET (2026-08-30): no checkpoint field persists RNG
    # state, so a process that RESUMES from an existing checkpoint (crash-
    # retry, or a deliberate periodic-restart pattern such as
    # run_gridfix_queue_staged.sh) previously re-applied the SAME base seed
    # as a fresh start -- replaying the identical np.random/cp.random draw
    # sequence (e.g. ray-origin x,y sampling, `origins = cp.random.uniform(...)`
    # at deposit_batch_gpu) at the start of every resumed stage instead of
    # continuing it. Found while validating the periodic-restart fix for
    # P1_GRIDFIX_CAMPAIGN_V1 -- caught before any real campaign data was
    # generated under the bug (verified via the launch log: zero stage
    # restarts had occurred on the live run before this fix landed). Fix:
    # on a genuine resume, offset the base seed by a persistent per-run
    # stage counter so each resumed stage draws a distinct, still fully
    # deterministic/reproducible sub-seed instead of replaying stage 1. A
    # run that never resumes (the overwhelming majority of this project's
    # existing campaigns) is completely unaffected -- offset stays 0.
    sim_cfg = cfg.get('simulation', cfg)
    cli_seed = getattr(args, 'random_seed', None)
    yaml_seed = sim_cfg.get('random_seed', None) if isinstance(sim_cfg, dict) else None
    chosen_seed = cli_seed if cli_seed not in (None, 0) else yaml_seed
    if chosen_seed is not None:
        try: chosen_seed = int(chosen_seed)
        except (TypeError, ValueError): chosen_seed = None

    _is_resume = (not args.fresh) and any(
        os.path.exists(os.path.join(cfg['checkpoint_dir'], _n))
        for _n in ('checkpoint_v3_A.h5', 'checkpoint_v3_B.h5', 'checkpoint_v3_emergency.h5',
                   cfg.get('checkpoint_filename', 'checkpoint_v3.h5'))
    )
    _seed_stage = 0
    if _is_resume:
        _counter_path = os.path.join(cfg['checkpoint_dir'], '.rng_stage_counter')
        try:
            _seed_stage = (int(open(_counter_path).read().strip()) + 1) if os.path.exists(_counter_path) else 1
        except Exception:
            _seed_stage = 1
        try:
            os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
            with open(_counter_path, 'w') as _f:
                _f.write(str(_seed_stage))
        except Exception:
            pass

    if chosen_seed is not None:
        effective_seed = chosen_seed if _seed_stage == 0 else (chosen_seed * 1_000_003 + _seed_stage) % (2**31 - 1)
        import random as _py_random
        _py_random.seed(effective_seed)
        np.random.seed(effective_seed)
        try:
            import cupy as _cp_for_seed
            _cp_for_seed.random.seed(effective_seed)
            print(f"[SEED] Seeded Python.random + NumPy + CuPy with random_seed = {effective_seed} "
                  f"(base={chosen_seed}, resume_stage={_seed_stage})")
        except Exception as _seed_err:
            print(f"[SEED] Seeded Python.random + NumPy with random_seed = {effective_seed} "
                  f"(base={chosen_seed}, resume_stage={_seed_stage})  (CuPy not seeded: {_seed_err})")
        cfg['random_seed_used'] = effective_seed
        cfg['random_seed_base'] = chosen_seed
        cfg['random_seed_resume_stage'] = _seed_stage
    else:
        print("[SEED] No random_seed provided — RNGs use OS entropy (non-reproducible)."
              " Set simulation.random_seed in YAML for reproducible runs.")
        cfg['random_seed_used'] = None

    # Create simulator
    # Override config with CLI
    if args.kdtree_interval:
        cfg['kdtree_rebuild_interval'] = args.kdtree_interval
    if args.target_height:
        cfg['target_height'] = args.target_height
    
    sim = GLADV3Simulator(cfg)
    sim.force_dynamic = args.force_dynamic

    # Load state
    loaded = False
    if not args.fresh:
        if args.load_v3:
            loaded = sim.load_v3_checkpoint(args.load_v3)
            if not loaded:
                print("[V3] Requested checkpoint was not loaded. Exiting instead of silently starting fresh.")
                sys.exit(1)
        else:
            # Try default V3 checkpoint. _do_save_checkpoint() never writes the
            # plain 'checkpoint_v3.h5' name -- it always rotates between
            # 'checkpoint_v3_A.h5' / 'checkpoint_v3_B.h5' (see checkpoint.py
            # I/O architecture doc). Without this fallback, any resume that
            # omits --load_v3 silently falls through to a fresh substrate
            # init instead of resuming -- found 2026-07-04 after box=250nm
            # discarded a 10.2M-atom checkpoint this way.
            candidates = [
                os.path.join(cfg['checkpoint_dir'],
                              cfg.get('checkpoint_filename', 'checkpoint_v3.h5')),
                os.path.join(cfg['checkpoint_dir'], 'checkpoint_v3_A.h5'),
                os.path.join(cfg['checkpoint_dir'], 'checkpoint_v3_B.h5'),
            ]
            existing = [p for p in candidates if os.path.exists(p)]
            if existing:
                default_v3 = max(existing, key=os.path.getmtime)
                loaded = sim.load_v3_checkpoint(default_v3)

    if not loaded:
        print("[V3] No checkpoint loaded — initialising substrate seed layer")
        sim.initialise_substrate()

    # Run
    sim.run_simulation(
        n_particles=args.n_particles,
        xyz_path=args.out,
        lmp_path=args.lmp
    )

    print("[V3] Done.")


if __name__ == "__main__":
    main()
