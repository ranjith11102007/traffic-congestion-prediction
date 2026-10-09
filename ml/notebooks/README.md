# Notebooks

Exploratory analysis lives here. Two constraints apply to anything committed in
this directory:

1. **No generated traffic data.** A notebook may load a real dataset or one of the
   prepared frames under `ml/data/processed`. It must not fabricate observations to
   make a plot or a metric look meaningful.
2. **Reproducible from the repo.** Prefer importing `ml.preprocessing` over
   copy-pasting transformations, so the notebook and the training pipeline cannot
   drift apart.

Notebook outputs are stripped before committing (see `.gitignore`) to keep the
repository readable. Re-run cells locally to regenerate any figures.

Planned notebooks:

- `01_exploration.ipynb` — distributions, missingness, congestion-class balance
- `02_baseline_random_forest.ipynb` — first working model and its error analysis
- `03_model_comparison.ipynb` — side-by-side comparison of candidate models