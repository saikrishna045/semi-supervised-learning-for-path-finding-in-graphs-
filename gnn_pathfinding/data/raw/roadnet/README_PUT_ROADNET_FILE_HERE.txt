PUT THE ROADNET DATASET FILE IN THIS FOLDER
===========================================

Download from SNAP (free, no signup):
    https://snap.stanford.edu/data/roadNet-PA.html      -> roadNet-PA.txt.gz
    (also available: roadNet-CA.txt.gz, roadNet-TX.txt.gz - same format, just bigger)

Either file name works, compressed or not:
    data/raw/roadnet/roadNet-PA.txt.gz       (you do NOT have to unzip it)
    data/raw/roadnet/roadNet-PA.txt

The scripts pick up the first data file in this folder automatically.
This README file is ignored. Then run:

    python scripts/run_all.py
