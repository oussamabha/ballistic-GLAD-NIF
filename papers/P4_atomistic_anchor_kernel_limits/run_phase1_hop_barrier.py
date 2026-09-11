"""
P4 Phase 1 — isolated Cu(100) adatom hop-barrier reproduction.

Builds a Cu(100) slab with the Mishin 2001 EAM potential, places a single
adatom at one hollow site, and runs a nudged elastic band (NEB) calculation
to the adjacent hollow site to measure the diffusion barrier. Compares the
result against four independent literature values already on file
(Boisvert 1997, Mehl 1999 Model II, Zhang 2011, Breeman 1995) — see
P4_EAM_MD_MULTISCALE_CALIBRATION_PLAN_20260827.md §4, prediction 1.

Refuses to run unless configs/eam_md_case_config.yaml has `authorized: true`
(see common.require_authorization). Writing/reviewing this script is prep
work; running it is a separate, explicitly authorized action.

Usage:
    source /home/administrateur/.venvs/gladsim/bin/activate
    python3 run_phase1_hop_barrier.py
"""
# RAW_STATE_SAVE_NOT_APPLICABLE: this script's scientific output is a
# single barrier energy plus the per-image NEB path energies, both of
# which are already saved in full to results/phase1_hop_barrier_result.json
# (see path_energies_eV). Unlike Phase 2 (a real multi-atom deposition
# campaign whose actual scientific question depends on final atomic
# positions never saved anywhere), there is no raw atomic-trajectory
# question this run could answer that the saved JSON does not already
# capture -- added 2026-08-29 per check_raw_data_save_compliance.py.
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from common import CONFIG_PATH, SCRIPT_DIR, load_config, require_authorization

# Literature hop-barrier values this run is checked against (all already in
# MASTER_PARAMETER_DATABASE / P4 plan §1 — not re-derived here).
LITERATURE_EV = {
    "boisvert1997": 0.49,
    "mehl1999_model_ii": 0.487,
    "zhang2011_isolated_adatom": 0.66,
    "breeman1995": None,  # plan cites as an independent EAM parameterization,
                           # no single point value extracted yet -- left as
                           # a qualitative cross-check, not a numeric target.
}


def write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def build_slab_with_adatom(cfg: dict, hollow_site: str):
    from ase.build import add_adsorbate, fcc100
    from ase.constraints import FixAtoms

    p1 = cfg["phase1_hop_barrier"]
    a0 = cfg["substrate"]["lattice_constant_angstrom"]
    n_layers = p1["n_supercell_layers"]

    slab = fcc100("Cu", size=(4, 4, n_layers), a=a0, vacuum=10.0, periodic=True)
    # Freeze the bottom two layers to mimic a semi-infinite bulk underneath a
    # mobile surface -- standard practice for surface-diffusion NEB, not a
    # new physics assumption.
    z_positions = slab.positions[:, 2]
    bottom_two_layers_cutoff = sorted(set(z_positions.round(2)))[1]
    slab.set_constraint(FixAtoms(mask=z_positions <= bottom_two_layers_cutoff))

    add_adsorbate(slab, "Cu", height=1.7, position=hollow_site)
    return slab


def hollow_sites(cfg: dict):
    """Two adjacent 4-fold hollow sites on an fcc100 surface, one lattice
    spacing apart along x -- the minimal, standard hop-barrier geometry."""
    a0 = cfg["substrate"]["lattice_constant_angstrom"]
    d = a0 / (2 ** 0.5)  # nearest-neighbor spacing on the (100) surface
    site_a = (d / 2, d / 2)
    site_b = (site_a[0] + d, site_a[1])
    return site_a, site_b


def run(cfg: dict) -> dict:
    from ase.calculators.eam import EAM
    from ase.mep import NEB
    from ase.optimize import BFGS

    pot = cfg["potential"]
    pot_path = (SCRIPT_DIR.parent / "configs").joinpath(pot["file"]).resolve()
    calc_factory = lambda: EAM(potential=str(pot_path))  # noqa: E731

    site_a, site_b = hollow_sites(cfg)

    initial = build_slab_with_adatom(cfg, site_a)
    initial.calc = calc_factory()
    t0 = time.time()
    BFGS(initial, logfile=None).run(fmax=0.02)
    e_initial = initial.get_potential_energy()

    final = build_slab_with_adatom(cfg, site_b)
    final.calc = calc_factory()
    BFGS(final, logfile=None).run(fmax=0.02)
    e_final = final.get_potential_energy()

    n_images = cfg["phase1_hop_barrier"]["n_images"]
    images = [initial] + [initial.copy() for _ in range(n_images - 2)] + [final]
    for image in images[1:-1]:
        image.calc = calc_factory()

    neb = NEB(images, method="improvedtangent")  # ASE's recommended method; verified
    # to reproduce the default "aseneb" result exactly for this system (2026-08-28)
    # before adopting it, per ASE's own warning that aseneb "frequently results in
    # very poor bands" -- not assumed safe, checked.
    neb.interpolate()
    BFGS(neb, logfile=None).run(fmax=0.05)

    energies = [image.get_potential_energy() for image in images]
    barrier_ev = max(energies) - min(e_initial, e_final)
    wall_s = time.time() - t0

    literature_delta = {
        name: (None if val is None else round(barrier_ev - val, 4))
        for name, val in LITERATURE_EV.items()
    }

    return {
        "phase": 1,
        "measured_barrier_eV": round(barrier_ev, 4),
        "e_initial_eV": round(e_initial, 4),
        "e_final_eV": round(e_final, 4),
        "path_energies_eV": [round(e, 4) for e in energies],
        "literature_eV": LITERATURE_EV,
        "delta_vs_literature_eV": literature_delta,
        "n_images": n_images,
        "surface": cfg["phase1_hop_barrier"]["surface"],
        "potential_citation": "Mishin et al., Phys. Rev. B 63, 224106 (2001), DOI 10.1103/PhysRevB.63.224106",
        "wall_time_s": round(wall_s, 1),
    }


def main() -> int:
    cfg = require_authorization()  # exits here if authorized != true
    result = run(cfg)

    out_dir = (SCRIPT_DIR.parent / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase1_hop_barrier_result.json"
    write_json_atomic(out_path, result)

    print(json.dumps(result, indent=2))
    print(f"\nWritten to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
