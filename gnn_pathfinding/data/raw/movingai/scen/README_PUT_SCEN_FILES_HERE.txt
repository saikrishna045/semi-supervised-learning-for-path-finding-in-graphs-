PUT MOVINGAI .scen FILES IN THIS FOLDER
=======================================

From the same page as the maps (https://movingai.com/benchmarks/grids.html) download the
"scenarios" zip and unzip it here, keeping the original names:

    data/raw/movingai/scen/arena.map.scen

If a scenario file named <map name>.scen exists, the pipeline uses its start/goal pairs
(and checks our Dijkstra costs against the optimal lengths stored in it).
If there is none, random pairs are generated instead - the scenario files are optional.
