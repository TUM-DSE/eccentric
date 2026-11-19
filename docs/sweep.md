# The sweep (how the dataset is produced)

The dataset is produced by `sweep/`, which generates one job per circuit or decode, runs them on a cluster
(`python -m sweep.run_job <run_dir> <job_id>`, each until 100 logical errors or its time budget) and collects
them (`python -m sweep.collect <run dirs> <out>`; `python -m sweep.stats` then updates the README badges).
Each row records the commit that produced it. `validation/` holds the cluster checks of circuits, code and
circuit distances, and decoders.

The table's columns and statuses: [dataset.md](dataset.md). The slices: `sweep/slices.py`.
