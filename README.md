Extract TDEP input files (infile.positions, infile.forces, infile.stat,
infile.meta) from one or more VASP vasprun.xml files.

Works for both:
  - many independent vasprun.xml files, each a single-point or short
    calculation (e.g. samples_X/sample.N/vasprun.xml from an active-learning
    / ML training-set workflow), and
  - one (or a few) long AIMD vasprun.xml, each containing many ionic steps.
Pure standard library -- no ASE, no numpy, no external tools required
(a subprocess/grep dependency in the original version has been replaced
with a portable pure-Python line scan).

Frame model
-----------
Every ionic step found across every matched vasprun.xml file (positions +
forces + one energy/stress record each) is treated as one "frame" in a
single combined sequence, in the file order established by --pattern (files
are grouped by their directory two levels above vasprun.xml, those groups
are visited in REVERSE sorted order -- matching the original script's
"samples_prev before samples_N" convention -- and files within a group are
sorted by the trailing number in their own directory name, e.g. sample.12).

--discard-start N drops the first N frames of that COMBINED sequence (not
per file) -- appropriate for a single long MD trajectory (equilibration) and
also a reasonable way to drop an initial batch of many independent samples.
--stride K then keeps every Kth frame of what's left (decorrelation /
subsampling). If your independent samples each need their OWN per-file
equilibration discard instead (many short MD runs, each starting from a
different structure), that is a different semantics than what's implemented
here -- ask for it explicitly if that's what you need.

Usage:
    python vasprun_to_tdep.py                       # temperature defaults to 300.0 K
    python vasprun_to_tdep.py --temperature 250.0
    python vasprun_to_tdep.py --pattern "./samples*/sample.*/vasprun.xml"
    python vasprun_to_tdep.py --discard-start 500 --stride 5
    python vasprun_to_tdep.py --dry-run

