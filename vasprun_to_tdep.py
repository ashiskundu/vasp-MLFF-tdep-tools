#!/usr/bin/env python3

import argparse
import bisect
import glob
import itertools
import os
import re
import sys
import xml.etree.ElementTree as ET

DEFAULT_PATTERN = "./samples*/sample.*/vasprun.xml"


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------

def _trailing_int(path):
    """Numeric key from the digits immediately before /vasprun.xml, e.g.
    .../sample.12/vasprun.xml -> 12. Falls back to 0 if none found."""
    m = re.search(r"(\d+)(?=[\\/]vasprun\.xml$)", path)
    return int(m.group()) if m else 0


def discover_vasprun_files(pattern):
    """Find vasprun.xml files matching `pattern`, grouped by the directory
    two levels above each file (e.g. .../samples_3/sample.12/vasprun.xml ->
    group .../samples_3), groups visited in reverse-sorted order, files
    within a group sorted by their own trailing number."""
    matches = glob.glob(pattern)
    if not matches:
        return [], []

    groups = {}
    for path in matches:
        base_dir = os.path.dirname(os.path.dirname(path)) or "."
        groups.setdefault(base_dir, []).append(path)

    ordered_base_dirs = sorted(groups.keys(), reverse=True)
    all_files = []
    for base_dir in ordered_base_dirs:
        all_files.extend(sorted(groups[base_dir], key=_trailing_int))
    return all_files, ordered_base_dirs


# ----------------------------------------------------------------------
# Cheap metadata reads
# ----------------------------------------------------------------------

def extract_number_of_atoms(vasprun_file):
    tree = ET.parse(vasprun_file)
    root = tree.getroot()
    atoms_tag = root.find(".//atominfo/atoms")
    if atoms_tag is not None and atoms_tag.text and atoms_tag.text.strip().isdigit():
        return int(atoms_tag.text.strip())
    raise ValueError(
        f"could not find the number of atoms in <atominfo> in {vasprun_file}"
    )


def resolve_num_atoms(vasprun_files, override):
    if override is not None:
        return override
    n0 = extract_number_of_atoms(vasprun_files[0])
    for f in vasprun_files[1:]:
        n = extract_number_of_atoms(f)
        if n != n0:
            sys.exit(
                f"error: inconsistent atom count across input files: "
                f"{vasprun_files[0]} has {n0} atoms, {f} has {n}. "
                f"Pass --num-atoms to override this check if that's expected."
            )
    return n0


def count_totalsc_entries(vasprun_file):
    """Fast, pure-Python count of ionic steps: one 'totalsc' <time> tag per
    completed ionic step. Used only to plan the discard/stride selection
    before the real (ElementTree) extraction pass -- the real pass re-derives
    the true count from what it actually parses and will warn on mismatch."""
    count = 0
    with open(vasprun_file, "r", errors="replace") as fh:
        for line in fh:
            if "totalsc" in line:
                count += 1
    return count


# ----------------------------------------------------------------------
# Frame selection (discard-start / stride over the combined sequence)
# ----------------------------------------------------------------------

def compute_global_selection(n_per_file, discard_start, stride):
    if stride < 1:
        raise ValueError("--stride must be >= 1")
    if discard_start < 0:
        raise ValueError("--discard-start must be >= 0")
    total = sum(n_per_file)
    boundaries = list(itertools.accumulate(n_per_file))
    selected_per_file = [[] for _ in n_per_file]
    for g in range(discard_start, total, stride):
        file_idx = bisect.bisect_right(boundaries, g)
        start_of_file = boundaries[file_idx - 1] if file_idx > 0 else 0
        selected_per_file[file_idx].append(g - start_of_file)
    return selected_per_file, total


# ----------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------

def extract_positions(vasprun_file, num_atoms, keep_local_indices, output_file):
    tree = ET.parse(vasprun_file)
    root = tree.getroot()
    positions = [
        pos for structure in root.findall(".//structure")
        if structure.get("name") not in ("primitive_cell", "initialpos", "finalpos")
        for pos in structure.findall("varray[@name='positions']/v")
    ]
    if not positions:
        print(f"  no relevant positions found in {vasprun_file}")
        return 0

    total_positions = len(positions)
    if total_positions % num_atoms != 0:
        print(f"  warning: {vasprun_file} has {total_positions} position "
              f"entries, not a multiple of num_atoms={num_atoms}; trailing "
              f"partial step ignored")
    n_steps = total_positions // num_atoms

    written = 0
    with open(output_file, "a") as outfile:
        for local_idx in keep_local_indices:
            if local_idx >= n_steps:
                print(f"  warning: requested step {local_idx} but "
                      f"{vasprun_file} only has {n_steps} position steps; "
                      f"skipping")
                continue
            start = local_idx * num_atoms
            for j in range(start, start + num_atoms):
                v = [float(x) for x in positions[j].text.split()]
                outfile.write(f"{v[0]: >10.8f} {v[1]: >10.8f} {v[2]: >10.8f}\n")
            written += 1
    return n_steps, written


def extract_forces(vasprun_file, num_atoms, keep_local_indices, output_file):
    tree = ET.parse(vasprun_file)
    root = tree.getroot()
    forces = root.findall(".//varray[@name='forces']/v")
    if not forces:
        print(f"  no forces found in {vasprun_file}")
        return 0, 0

    total_forces = len(forces)
    if total_forces % num_atoms != 0:
        print(f"  warning: {vasprun_file} has {total_forces} force entries, "
              f"not a multiple of num_atoms={num_atoms}; trailing partial "
              f"step ignored")
    n_steps = total_forces // num_atoms

    written = 0
    with open(output_file, "a") as outfile:
        for local_idx in keep_local_indices:
            if local_idx >= n_steps:
                print(f"  warning: requested step {local_idx} but "
                      f"{vasprun_file} only has {n_steps} force steps; "
                      f"skipping")
                continue
            start = local_idx * num_atoms
            for j in range(start, start + num_atoms):
                v = [float(x) for x in forces[j].text.split()]
                outfile.write(f"{v[0]: >10.7f} {v[1]: >10.7f} {v[2]: >10.7f}\n")
            written += 1
    return n_steps, written


def extract_stress_energy(vasprun_files, selected_per_file, output_file):
    """Single pass, line-based (vasprun.xml can be huge -- this avoids a
    second full ElementTree parse). Tracks a LOCAL per-file step counter to
    decide, via `selected_per_file`, whether each completed ionic step gets
    written, and a GLOBAL output row counter so infile.stat rows number
    continuously 1..N across all files, matching the row order of the
    concatenated infile.positions/infile.forces."""
    output_row = 0
    per_file_step_counts = []

    with open(output_file, "w") as outfile:
        for file_idx, vasprun_file in enumerate(vasprun_files):
            keep_set = set(selected_per_file[file_idx])
            local_step_idx = -1  # becomes 0 on the first completed step

            with open(vasprun_file, "r", errors="replace") as f:
                lines = f.readlines()

            in_stress_block = False
            temp_block = []
            energy_values = {}

            for line in lines:
                if "stress" in line and "<varray" in line:
                    in_stress_block = True
                    temp_block = []

                if in_stress_block:
                    temp_block.append(line.strip())

                if "</time>" in line and "totalsc" in line:
                    in_stress_block = False
                    local_step_idx += 1

                    if temp_block:
                        stress_values = []
                        for entry in temp_block:
                            if "<v>" in entry and "</v>" in entry:
                                values = [float(x) for x in
                                          entry.replace("<v>", "").replace("</v>", "").split()]
                                stress_values.append(values)

                        if len(stress_values) == 3 and local_step_idx in keep_set:
                            output_row += 1
                            xx, xy, xz = stress_values[0]
                            yx, yy, yz = stress_values[1]
                            zx, zy, zz = stress_values[2]

                            xx, yy, zz = xx * -0.1, yy * -0.1, zz * -0.1
                            xy, xz, yz = xy * -0.1, xz * -0.1, yz * -0.1
                            stress_avg = (xx + yy + zz) / 3.0

                            outfile.write(f"{output_row:7d} {output_row - 1:9.3f} ")
                            outfile.write(f"{energy_values.get('e_fr_energy', 0):23.15e} ")
                            outfile.write(f"{energy_values.get('e_wo_entrp', 0):23.15e} ")
                            outfile.write(f"{-energy_values.get('e_0_energy', 0):23.15e} ")
                            outfile.write(f"{0.0:15.9f} {stress_avg:15.9f} ")
                            outfile.write(f"{xx:15.9f} {yy:15.9f} {zz:15.9f} "
                                          f"{yz:15.9f} {xz:15.9f} {xy:15.9f}\n")
                        elif len(stress_values) != 3:
                            print(f"  warning: invalid stress matrix for "
                                  f"local step {local_step_idx} in "
                                  f"{vasprun_file}")
                    else:
                        print(f"  warning: no stress block captured before "
                              f"totalsc close for local step {local_step_idx} "
                              f"in {vasprun_file}")

                    energy_values = {}  # don't leak into the next step
                    continue

                if '<i name="e_fr_energy">' in line:
                    energy_values["e_fr_energy"] = float(line.split(">")[1].split("<")[0])
                elif '<i name="e_wo_entrp">' in line:
                    energy_values["e_wo_entrp"] = float(line.split(">")[1].split("<")[0])
                elif '<i name="e_0_energy">' in line:
                    energy_values["e_0_energy"] = float(line.split(">")[1].split("<")[0])

            per_file_step_counts.append(local_step_idx + 1)

    return output_row, per_file_step_counts


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--temperature", type=float, default=300.0,
                    help="temperature in K, written to infile.meta (used by "
                         "TDEP for the free energy). Default: 300.0")
    p.add_argument("--pattern", default=DEFAULT_PATTERN,
                    help=f"glob pattern locating vasprun.xml files "
                         f"(default: {DEFAULT_PATTERN!r})")
    p.add_argument("--num-atoms", type=int, default=None,
                    help="override the atom count instead of auto-detecting "
                         "it from <atominfo> in the first (and, by default, "
                         "every) matched vasprun.xml")
    p.add_argument("--discard-start", type=int, default=0, metavar="N",
                    help="drop the first N frames of the combined sequence "
                         "across all matched files (equilibration). Default: 0")
    p.add_argument("--stride", type=int, default=1, metavar="K",
                    help="keep every Kth frame after discarding "
                         "(decorrelation/subsampling). Default: 1 (keep all)")
    p.add_argument("--timestep-fs", type=float, default=1.0,
                    help="timestep in fs written to infile.meta (informational "
                         "only; not currently used by TDEP downstream of this "
                         "script). Default: 1.0")
    p.add_argument("--outdir", default=".",
                    help="directory to write infile.* into (default: current "
                         "directory)")
    p.add_argument("--dry-run", action="store_true",
                    help="discover files and print the frame-count/selection "
                         "summary, but don't write any infile.* output")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out_positions = os.path.join(args.outdir, "infile.positions")
    out_forces = os.path.join(args.outdir, "infile.forces")
    out_stat = os.path.join(args.outdir, "infile.stat")
    out_meta = os.path.join(args.outdir, "infile.meta")

    if not args.dry_run:
        for path in (out_positions, out_forces, out_stat, out_meta):
            if os.path.exists(path):
                os.remove(path)
                print(f"{path} removed (will be rewritten).")

    # --- discovery ---
    all_vasprun_files, base_dirs = discover_vasprun_files(args.pattern)
    if not all_vasprun_files:
        sys.exit(f"error: no files matched pattern {args.pattern!r}")
    print(f"found {len(all_vasprun_files)} vasprun.xml file(s) across "
          f"{len(base_dirs)} group(s): {base_dirs}")

    # --- num_atoms ---
    num_atoms = resolve_num_atoms(all_vasprun_files, args.num_atoms)
    print(f"num_atoms = {num_atoms} "
          f"({'user override' if args.num_atoms is not None else 'auto-detected, checked across all files'})")

    # --- fast per-file frame counts, then global discard/stride selection ---
    n_per_file_fast = [count_totalsc_entries(f) for f in all_vasprun_files]
    print(f"frames per file (fast count): {n_per_file_fast}")
    try:
        selected_per_file, total_frames = compute_global_selection(
            n_per_file_fast, args.discard_start, args.stride)
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    n_selected_fast = sum(len(s) for s in selected_per_file)
    print(f"total frames found: {total_frames}")
    print(f"discard-start={args.discard_start}, stride={args.stride} -> "
          f"{n_selected_fast} frame(s) selected (before any per-file "
          f"consistency re-checks during extraction)")

    if args.dry_run:
        print("--dry-run: stopping before writing any output.")
        return

    if n_selected_fast == 0:
        sys.exit("error: selection produced 0 frames -- check "
                  "--discard-start/--stride against the frame counts above")

    # --- positions ---
    print("extracting positions...")
    for file_idx, vasprun_file in enumerate(all_vasprun_files):
        if not selected_per_file[file_idx]:
            continue
        n_steps, written = extract_positions(
            vasprun_file, num_atoms, selected_per_file[file_idx], out_positions)
        print(f"  {vasprun_file}: {written}/{n_steps} steps written")
    print(f"infile.positions done -> {out_positions}")

    # --- forces ---
    print("extracting forces...")
    for file_idx, vasprun_file in enumerate(all_vasprun_files):
        if not selected_per_file[file_idx]:
            continue
        n_steps, written = extract_forces(
            vasprun_file, num_atoms, selected_per_file[file_idx], out_forces)
        print(f"  {vasprun_file}: {written}/{n_steps} steps written")
    print(f"infile.forces done -> {out_forces}")

    # --- stress/energy ---
    print("extracting stress/energy...")
    total_written, per_file_step_counts = extract_stress_energy(
        all_vasprun_files, selected_per_file, out_stat)
    for vasprun_file, fast_n, real_n in zip(all_vasprun_files, n_per_file_fast,
                                              per_file_step_counts):
        if fast_n != real_n:
            print(f"  warning: {vasprun_file}: fast totalsc count was "
                  f"{fast_n}, actual completed steps found while writing "
                  f"was {real_n} -- selection for this file was planned "
                  f"against the fast count, double check the output")
    print(f"infile.stat done -> {out_stat} ({total_written} row(s))")

    # --- meta ---
    # total_timesteps mirrors infile.stat's actual row count -- that's the
    # number TDEP needs to match infile.positions/infile.forces row counts.
    with open(out_meta, "w") as f:
        f.write(f"{num_atoms:>10}     # N atoms\n")
        f.write(f"{total_written:>10}     # N timesteps after discard-start/stride\n")
        f.write(f"{args.timestep_fs:>10}     # timestep in fs (currently not used)\n")
        f.write(f"{args.temperature:>10}     # temperature in K (only for free energy)\n")
    print(f"infile.meta done -> {out_meta}")

    if total_written != n_selected_fast:
        print(f"warning: infile.stat wrote {total_written} rows but the "
              f"planned selection was {n_selected_fast} frames -- see the "
              f"per-file warnings above for why (likely a file with fewer "
              f"actual steps than its fast totalsc count suggested)")


if __name__ == "__main__":
    main()
