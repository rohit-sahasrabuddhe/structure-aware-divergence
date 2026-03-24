# structure-aware-divergence

This repository accompanies Sahasrabuddhe and Lambiotte (2026): https://arxiv.org/abs/2603.22237.

We provide software to implement the methods proposed in the paper and include notebooks to replicate the results.

`tools` contains the code for our structure-aware methods.
- `core.py` contains core tools for computing entropies, divergences, and Bregman information.
- `clustering.py` implements clustering with a k-means-style algorithm.

`outputs` contains files that the results in the paper are based on.
- `clustering_results.pkl` for the synthetic clustering experiment
- `timetrial_M50_A2.pkl` for the time trial experiment
- `ew_occupations` contains intermediate files and data reported in the paper

`data` contains the empirical datasets we use.
- `rutor_glacier` for the Rutor glacier experiment from the `adiv` R package: https://cran.r-project.org/web/packages/adiv/index.html, originally published by Caccianiga et al.: https://doi.org/10.1111/j.0030-1299.2006.14107.x
- `ew_occupations` 
    - 2021 Census of England and Wales, originally from Nomis: https://www.nomisweb.co.uk/datasets/c2021ts064
    - Occupations to skills mapping from ONET: https://www.onetcenter.org/dictionary/30.2/excel/skills.html
    - SOC2020 to ONET occupation codes from NFER: https://www.nfer.ac.uk/key-topics-expertise/education-to-employment/the-skills-imperative-2035/resources/systematic-mapping-of-soc-2020/


The notebooks in the main directory contain code to reproduce our experiments.
- `synthetic_clustering.ipynb` for the synthetic clustering experiment
- `synthetic_timetrial.ipynb` for the time trial against OT
- `rutor_glacier.ipynb` for the Rutor glacier experiment
- `occupations_preprocessing.ipynb` for identifying the tradable occupaitons in England and Wales
- `occupations_analysis.ipynb` for the experiments reported in the paper

Our software is written in Python 3.12.8. In addition to the standard libraries, we use:
- `numpy 1.26.4`
- `pandas 2.2.3`
- `scipy 1.13.1` (for distances and correlations)
- `sklearn 1.5.2` (for Adjusted Mutual Information)
- `geopandas 1.1.1` (for geographic data)
- `matplotlib 3.10.0` (for visualisations)
- `seaborn 0.13.2` (for visualisations)
- `ot 0.9.5` (for Optimal Transport methods)
Our core methods rely only on `numpy`. We saved some results with `pickle 4.0`.

We thank the teams publishin and maintaining the data and developing open source software libraries.
