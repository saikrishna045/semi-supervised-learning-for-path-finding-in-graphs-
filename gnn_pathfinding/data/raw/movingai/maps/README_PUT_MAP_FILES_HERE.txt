PUT MOVINGAI .map FILES IN THIS FOLDER
======================================

Download from https://movingai.com/benchmarks/grids.html
Pick a set (e.g. Dragon Age: Origins "dao", or Mazes), download the "maps" zip and unzip it
so the .map files sit directly in this folder:

    data/raw/movingai/maps/arena.map
    data/raw/movingai/maps/den312d.map

Scenario (.scen) files go in the sibling folder:  data/raw/movingai/scen/
Then run:   python scripts/run_all.py --kind movingai --map arena.map
