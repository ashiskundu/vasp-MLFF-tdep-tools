Extract TDEP input files (infile.positions, infile.forces, infile.stat,
infile.meta) from one or more VASP vasprun.xml files.

Works for both sTDEP and mdTDEP:
  - many independent vasprun.xml files, each a single-point or short
    calculation (e.g. samples_X/sample.N/vasprun.xml from an active-learning
    / ML training-set workflow), and
  - one (or a few) long AIMD vasprun.xml, each containing many ionic steps.
Pure standard library -- no ASE, no numpy, no external tools required
(a subprocess/grep dependency in the original version has been replaced
with a portable pure-Python line scan).

Usage:sTDEP

    python vasprun_to_tdep.py                       # temperature defaults to 300.0 K
    
    python vasprun_to_tdep.py --temperature 250.0
    
    python vasprun_to_tdep.py --pattern "./samples*/sample.*/vasprun.xml"
    
    python vasprun_to_tdep.py --discard-start 500 --stride 5
    
    python vasprun_to_tdep.py --dry-run

Usage:mdTDEP 

    python vasprun_to_tdep.py --pattern ./vasprun.xml --discard-start 2000 --stride 10 --timestep-fs 1.0
    
    python vasprun_to_tdep.py --pattern ./vasprun.xml --discard-start 2000
