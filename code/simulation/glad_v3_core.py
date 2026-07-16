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

try:
    import cupy as cp
    # 2026-07-03: switched from a plain device-memory pool to CUDA Unified
    # (managed) memory. This lets allocations transparently page out to host
    # RAM when VRAM (4GB on this GPU) is exhausted, instead of hard-failing.
    # Verified working standalone (cupy 14.1.0): allocated 5.6GB > 4GB VRAM
    # via managed memory. Does not change simulation physics -- only where
    # the data physically resides; may be slower than pure VRAM when paging.
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
        'rotation_rpm': None,
        'checkpoint_height_interval': 50.0,
        'checkpoint_particle_interval': 20_000_000,
        'checkpoint_dir': './checkpoints',
        'checkpoint_filename': 'checkpoint_v3.h5',
        'allow_legacy_resume': False,
        'substrate_spacing': 1.0,
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
        'activation_energy': 0.041,  # eV
        'use_arrhenius': True,
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

    if 'performance' in raw:
        cfg['use_gpu'] = raw['performance'].get('use_gpu', False)
        cfg['batch_size'] = raw['performance'].get('batch_size', 512)
        cfg['cell_size'] = raw['performance'].get('cell_size', cfg['cell_size'])
        cfg['kdtree_rebuild_interval'] = raw['performance'].get('kdtree_rebuild_interval', 1000000)
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
        cfg['activation_energy']        = raw['physics'].get('activation_energy', cfg['activation_energy'])
        cfg['use_arrhenius']            = raw['physics'].get('use_arrhenius', cfg['use_arrhenius'])
        cfg['pitch']                    = float(raw['physics'].get('pitch', PITCH))
        cfg['growth_rate']              = float(raw['physics'].get('growth_rate', GROWTH_RATE))
        cfg['active_window_height']     = raw['physics'].get('active_window_height', cfg['active_window_height'])
        cfg['resume_active_window_height'] = raw['physics'].get('resume_active_window_height', cfg['resume_active_window_height'])
        cfg['thermal_throttle_temp']    = raw['physics'].get('thermal_throttle_temp', cfg['thermal_throttle_temp'])
        cfg['thermal_critical_temp']    = raw['physics'].get('thermal_critical_temp', cfg['thermal_critical_temp'])
        if 'author_radius_nm' in raw['physics']:
            cfg['author_radius_nm'] = float(raw['physics']['author_radius_nm'])
        angular_raw = raw['physics'].get('angular_distribution', {})
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

    cfg.setdefault('author_radius_nm', float(AUTHOR_RADIUS))
    cfg['_source_yaml'] = os.path.abspath(yaml_path)
    return cfg


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
    float*        t_results      // [n_rays]
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
    unsigned int  seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rays) return;
    
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

    for (int h = 0; h < hops; h++) {
        if (next_rand() > hop_prob) continue;
        float angle = next_rand() * 6.2831853f;
        float r = next_rand() * diff_radius;
        float tx = fmodf(pos.x + r * cosf(angle), box_size.x);
        if (tx < 0) tx += box_size.x;
        float ty = fmodf(pos.y + r * sinf(angle), box_size.y);
        if (ty < 0) ty += box_size.y;

        float tz = r_nm;   // floor = r (was hardcoded 0.1f for legacy r=0.10 nm)
        int gx = floorf((tx - grid_min.x) / cell_size);
        int gy = floorf((ty - grid_min.y) / cell_size);

        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int nx = (gx + dx % grid_dims.x + grid_dims.x) % grid_dims.x;
                int ny = (gy + dy % grid_dims.y + grid_dims.y) % grid_dims.y;
                for (int nz = grid_dims.z - 1; nz >= 0; nz--) {
                    int c_idx = (nx * grid_dims.y + ny) * grid_dims.z + nz;
                    int start = cell_starts[c_idx];
                    int end = cell_starts[c_idx + 1];
                    if (start < end) {
                        for (int k = start; k < end; k++) {
                            float z_atom = all_positions[sorted_indices[k]].z;
                            // collision_diameter = 2*r_nm: atom center lands at full contact distance above existing
                            if (z_atom + collision_diameter > tz) tz = z_atom + collision_diameter;
                        }
                    }
                }
            }
        }
        float dz = tz - pos.z;
        if (dz >= 0 || next_rand() < expf(-(fabsf(dz)*0.1f) / kB_T)) {
            pos.x = tx; pos.y = ty; pos.z = tz;
        }
    }
    new_atoms[idx] = pos;
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
        for _k in ('alpha', 'pitch', 'batch_size', 'author_radius_nm'):
            if _k not in self.cfg:
                raise ValueError(f"[GLAD FATAL] Missing required physics parameter: '{_k}'")
        self._r  = np.float32(self.cfg['author_radius_nm'])
        self._cd = np.float32(2.0 * self.cfg['author_radius_nm'])
        self.kB = 8.617333262145e-5
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
        # Rebuild only every `_grid_rebuild_delta` new active atoms (or after
        # compaction).  Between rebuilds, the cached grid is used — new atoms
        # added since last rebuild won't cast shadows, but they represent < 0.5%
        # of the active slab and the physics error is negligible.
        self._cached_global_sorted_indices = None
        self._cached_grid_params = None
        self._n_active_at_last_rebuild = 0
        self._grid_rebuild_delta = max(512, int(cfg.get('batch_size', 512)) * 4)
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

        self._shadow_range = cfg['box_width']
        cell_size_cfg = float(cfg.get('cell_size', 0.4))
        self.gpu_grid = GPUGrid(cell_size=cell_size_cfg, box_dims=(cfg['box_width'], cfg['box_depth']))
        self._kernel = None
        self._diff_kernel = None

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
                kernel_source = (
                    RAY_SPHERE_KERNEL_SOURCE
                    if self.cfg.get('enable_surface_diffusion', False)
                    else RAY_SPHERE_ONLY_KERNEL_SOURCE
                )
                mod = cp.RawModule(code=kernel_source)
                self._kernel = mod.get_function('RAY_SPHERE_KERNEL')
                if self.cfg.get('enable_surface_diffusion', False):
                    self._diff_kernel = mod.get_function('DIFFUSION_KERNEL')
                    self._log("[V3] [OK] CUDA Kernels (Ballistic + Diffusion) compiled")
                else:
                    self._log("[V3] [OK] CUDA Kernel (Ballistic only) compiled")
            except Exception as e:
                self._log(f"[V3] [!] CUDA Compilation failed: {e}")

        # Throttled GPU thermal monitor (polls nvidia-smi every 5 s, not per-batch)
        self._thermal_monitor = GPUThermalMonitor(interval_s=5.0)

        if cfg.get('use_gpu', False) and not GPU_AVAILABLE:
            print("[FATAL] use_gpu: true but CuPy/CUDA initialization failed!", flush=True)
            sys.exit(1)

        signal.signal(signal.SIGINT, self._signal_handler)
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
            self._log(f"     Surface diffusion: ON | Ea={self.cfg.get('activation_energy', 0.0):.3f} eV | "
                      f"hop_prob={self._hop_prob:.3e}")
            if self._hop_prob < 1e-5:
                self._log("     [NOTE] Hop probability is extremely small; this run is effectively ballistic.")
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
        self._log("[V3] SIGINT — draining checkpoint queue, then saving emergency ...")
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
                                             self._cd, t_results))
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
                                                   self._cd, t_i))
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
            mask_active[reject] = 0   # K_stick: state transition, NOT z mutation
        else:
            self._last_batch_sticking_reject = 0

        # 4. Parallel Diffusion Kernel — K_diffuse : S_active → S_active (domain-restricted)
        if self.cfg['enable_surface_diffusion']:
            active_bool = mask_active.astype(cp.bool_)
            n_active_batch = int(active_bool.sum())
            if n_active_batch > 0:
                new_pos_active = new_pos[active_bool]   # restrict to S_active domain
                seed = np.random.randint(0, 1000000)
                grid_a = (n_active_batch + block - 1) // block
                self._diff_kernel((grid_a,), (block,), (new_pos_active, self.positions_gpu,
                                                          self.gpu_grid.cell_starts, global_sorted_indices,
                                                          nx, ny, nz, gm_x, gm_y, gm_z, csize,
                                                          float(self.cfg['diffusion_radius']), int(self.cfg['diffusion_hops']),
                                                          float(self._hop_prob), float(self._kB_T),
                                                          float(self._r), float(self._cd),
                                                          n_active_batch, float(bw), float(bd), np.uint32(seed)))
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
                          f"| {'PASS' if beta_pass else 'FAIL'}")
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
        beta_ok = (results.get('beta_pass') is True) or (results.get('beta_status') == 'SKIPPED_HELICAL')
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
    sim_cfg = cfg.get('simulation', cfg)
    cli_seed = getattr(args, 'random_seed', None)
    yaml_seed = sim_cfg.get('random_seed', None) if isinstance(sim_cfg, dict) else None
    chosen_seed = cli_seed if cli_seed not in (None, 0) else yaml_seed
    if chosen_seed is not None:
        try: chosen_seed = int(chosen_seed)
        except (TypeError, ValueError): chosen_seed = None
    if chosen_seed is not None:
        import random as _py_random
        _py_random.seed(chosen_seed)
        np.random.seed(chosen_seed)
        try:
            import cupy as _cp_for_seed
            _cp_for_seed.random.seed(chosen_seed)
            print(f"[SEED] Seeded Python.random + NumPy + CuPy with random_seed = {chosen_seed}")
        except Exception as _seed_err:
            print(f"[SEED] Seeded Python.random + NumPy with random_seed = {chosen_seed}  "
                  f"(CuPy not seeded: {_seed_err})")
        cfg['random_seed_used'] = chosen_seed
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
