# Eleyele Reservoir Flood Modeling: A Hybrid Machine Learning–System Dynamics Framework

## Research overview

This repository is the research compendium for a flood-vulnerability study of Eleyele Reservoir and the Ona River catchment in Ibadan, Nigeria. The study develops a sequential hybrid framework for a data-scarce tropical basin: an Attention-LSTM first evaluates how well daily satellite and reanalysis data represent streamflow dynamics, after which a process-based system dynamics model examines the routing of short-duration extreme rainfall through the catchment–reservoir system.

The machine-learning and system-dynamics components serve different purposes. The Attention-LSTM is used as a diagnostic model of the information contained in daily forcing data; it is not treated as a substitute for event-scale hydraulic or hydrological simulation. The routing component therefore uses temporally disaggregated design storms and explicit reservoir mass balance to investigate flood response at a five-minute computational time step.

## Study objectives

The study aims to:

1. construct a consistent daily hydroclimatic dataset from NASA POWER meteorological data and GEOGloWS reanalysis streamflow;
2. assess the predictive skill and limitations of an explainable Attention-LSTM for daily peak-flow estimation;
3. derive 5-, 10-, 20-, and 50-year design rainfall events using Gumbel frequency analysis and the Alternating Block Method;
4. simulate catchment response, reservoir storage, and spillway discharge under alternative runoff and antecedent-storage conditions; and
5. reconstruct the August 2011 Ibadan flood event as a scenario-based assessment.

## Methodological framework

| Phase | Analysis | Main output |
| --- | --- | --- |
| 1 | Temporal alignment of NASA POWER meteorology and GEOGloWS streamflow, followed by antecedent-rainfall and flow-lag feature engineering | Fused hydroclimatic dataset |
| 2 | Encoder–decoder Attention-LSTM, chronological train/validation/test partitioning, performance assessment, and SHAP interpretation | Diagnostic daily streamflow model |
| 3 | Annual-maximum rainfall extraction, Gumbel frequency analysis, and temporal disaggregation | Design-storm hyetographs |
| 4 | Linear catchment routing and level-pool reservoir mass balance with broad-crested-weir outflow | Flood-routing and sensitivity results |
| 5 | Scenario reconstruction using reported rainfall characteristics and assumed antecedent conditions | August 2011 event hydrograph |

The routing experiments use a Kirpich-based catchment lag and test multiple runoff coefficients and initial reservoir-storage levels. Model parameters and scenario assumptions are stated in the relevant notebooks and scripts and should be interpreted in light of the limited availability of local gauge and reservoir-operation records.

## Data sources

- **Meteorological forcing:** NASA Prediction of Worldwide Energy Resources (POWER) daily point data.
- **Streamflow:** GEOGloWS historical simulation/reanalysis for the Ona River reach used in this study.
- **Event information and reservoir parameters:** values compiled from the sources documented in the manuscript and the event-reconstruction notebook.

The fused dataset is provided as [`Eleyele_HydroMet_Master.csv`](Eleyele_HydroMet_Master.csv). Derived design storms are available in [`Hydrological/Design_Storm_Hyetographs.csv`](Hydrological/Design_Storm_Hyetographs.csv). Users should consult the original data providers for product definitions, limitations, and terms of use.

## Research files

The five notebooks correspond to the analytical sequence reported in the study:

1. [`01_Data_Fusion_and_Feature_Engineering.ipynb`](Notebook/01_Data_Fusion_and_Feature_Engineering.ipynb)
2. [`02_Machine_Learning_Benchmarking.ipynb`](Notebook/02_Machine_Learning_Benchmarking.ipynb)
3. [`03_Stochastic_Design_Storms.ipynb`](Notebook/03_Stochastic_Design_Storms.ipynb)
4. [`04_System_Dynamics_Reservoir_Routing.ipynb`](Notebook/04_System_Dynamics_Reservoir_Routing.ipynb)
5. [`05_August_2011_Event_Reconstruction.ipynb`](Notebook/05_August_2011_Event_Reconstruction.ipynb)

Equivalent standalone implementations and supporting utilities are in [`Python Code/`](Python%20Code/). Source and derived tabular data are organized under [`Meteorological/`](Meteorological/) and [`Hydrological/`](Hydrological/). The study-area map is provided as [`Figure_1_Study_Area.png`](Figure_1_Study_Area.png).

Large geospatial rasters, archives, QGIS-generated files, and downloaded literature are excluded from version control to keep the research compendium lightweight and to respect redistribution constraints.

## Reproducing the analysis

Use Python 3 with Jupyter and install the libraries imported by the notebooks and scripts, including `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit-learn`, `torch`, and `shap`. From the repository root, run the notebooks in numerical order. The scripts assume that input paths are resolved relative to that root directory.

Some notebooks write generated figures and result tables to folders such as `EDA/`, `Hydrological/`, and `System_Dynamics/`. Numerical results can vary slightly across Python, PyTorch, and hardware versions; the machine-learning implementation sets NumPy and PyTorch random seeds to improve repeatability.

## Manuscript and citation

The associated manuscript is currently under peer review and is not included in this public repository. This repository should be treated as supporting research material, not as the version of record. Citation information, including the article title, journal, DOI, and preferred software/data citation, will be added when the manuscript record becomes publicly available.

If this repository is referenced during a double-blind review, follow the target journal's anonymization policy; a public repository that identifies the authors may compromise reviewer blinding.

## Authors and acknowledgements

- **Lead researcher:** Mubarak Abiodun Adisa
- **Supervisor:** Dr Adesogan
- **Affiliation:** Department of Civil Engineering, University of Ibadan, Nigeria

The authors acknowledge the NASA POWER project and the GEOGloWS initiative for providing the hydroclimatic products used in this research.

## Scope and limitations

This material supports research and reproducibility. It is not an operational flood-warning system, a dam-safety certification, or a substitute for field-calibrated hydrological and hydraulic assessment. Results should be interpreted as model-based estimates conditioned on the stated data products, parameter choices, and scenario assumptions.
