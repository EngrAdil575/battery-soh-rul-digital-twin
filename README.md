# Deep Learning-Based State of Health and Remaining Useful Life Prediction for Lithium-Ion Batteries: A Leave-One-Cell-Out Comparison of LSTM and Hybrid CNN-LSTM

Dataset, code, and output files for the conference paper:

> **Deep Learning-Based State of Health and Remaining Useful Life Prediction for Lithium-Ion Batteries: A Leave-One-Cell-Out Comparison of LSTM and Hybrid CNN-LSTM**
> Muhammad Waiz Wais, Roya Mangal, Muhammad Sadiq Orakzai, Adil Zohaib, Sohail Khalid, Haris Sheh Zad.

This repository is the reproducibility record for the paper. It holds the battery cells used in the study, the analysis notebook, and the figures and tables reported in the paper.

## What the paper does

The paper predicts State of Health (SOH) and Remaining Useful Life (RUL) of lithium-ion cells from cycle-level voltage, current, temperature, and discharge-time features, and compares a standalone LSTM against a hybrid CNN-LSTM that adds one-dimensional causal convolutional layers before the LSTM. Both models are evaluated under leave-one-cell-out (LOCO) cross-validation on the four NASA cells: each cell in turn is held out while the others are used for fitting, so every reported number is for a cell the model never saw during training. End of life is taken at 80% SOH. Each model and feature set is run over five seeds, and results are reported as mean ± standard deviation (Table II), with and without the raw cycle index as an input. Coulombic-efficiency analysis and polynomial energy correction from an existing digital-twin baseline are reproduced on cell B0005, and a standalone offline digital twin replays cell B0018. The per-cell and per-seed numbers behind every figure and table are included under `outputs/`.

## Repository layout

```
battery-soh-rul-digital-twin/
├── data/                              NASA battery cells used in the study (see data/DATASET_README.txt)
│   ├── B0005.mat  B0006.mat  B0007.mat  B0018.mat   The four cells (all held out in turn under LOCO)
│   └── DATASET_README.txt
├── code/
│   ├── Battery_Digital_Twin_Notebook.ipynb   Google Colab notebook (full pipeline)
│   ├── Battery_Digital_Twin_AllCells.py      The notebook's code cells as a single script
│   └── requirements.txt
├── outputs/
│   ├── figures/                       Figs 1–5 (PNG, plus a vector PDF of Fig. 1)
│   ├── tables/                        Tables I–III (CSV, plus All_Tables.xlsx)
│   ├── results_data/                  Per-cycle, per-cell, and per-seed data behind the figures and tables
│   └── OUTPUTS_README.txt             File-by-file description mapped to the paper
├── Offline_Digital_Twin_Battery_Health_Estimation.html   Standalone offline digital twin (cell B0018; open in a browser)
├── CITATION.cff
└── LICENSE                            MIT (code). The NASA dataset is public-domain U.S. Government work.
```

## Dataset

The battery cells are from the NASA Prognostics Center of Excellence (PCoE) battery aging dataset:

> B. Saha and K. Goebel, "Battery data set," NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA, 2007.

Only the four cells used in the paper are included (`B0005`, `B0006`, `B0007`, `B0018`) as the original `.mat` files. The data are U.S. Government work in the public domain. The full repository and the other aging-ARC sets are available from NASA PCoE; see `data/DATASET_README.txt` for details.

## Reproducing the results

Open `code/Battery_Digital_Twin_Notebook.ipynb` in Google Colab and run the cells in order. Cell 1 locates the `.mat` files in Google Drive (or extracts the NASA ZIP); point it at the four cells in `data/`. The notebook builds the cycle-level dataset (EOL at 80% SOH), reproduces the baseline diagnostics on B0005 (Coulombic efficiency and energy correction), runs the leave-one-cell-out comparison of the LSTM and the hybrid CNN-LSTM over five seeds with and without the cycle input, writes Tables I–III and the per-seed data, produces Figures 2–5, and generates the offline digital twin for B0018.

`code/Battery_Digital_Twin_AllCells.py` is the same code as a single script. Results on a GPU can differ slightly between runs because GPU training is not bit-for-bit deterministic; the five-seed mean ± standard deviation in Table II captures this spread.

Dependencies are in `code/requirements.txt`: numpy, pandas, scipy, scikit-learn, tensorflow, and matplotlib. The digital twin is a standalone HTML file and needs no Python.

## Outputs

`outputs/figures/` holds Figures 1–5, `outputs/tables/` holds Tables I–III (CSV and `All_Tables.xlsx`), and `outputs/results_data/` holds the per-cycle, per-cell, and per-seed data behind them (including `LOCO_predictions.csv` and `LOCO_per_seed.csv`). `outputs/OUTPUTS_README.txt` maps each file to the paper.

## Offline digital twin

`Offline_Digital_Twin_Battery_Health_Estimation.html` is a standalone, offline digital twin for the test cell B0018. It needs no server or internet connection — open the file in any web browser.

Move the slider across the cycles to replay the cell's ageing. The page shows the predicted SOH and RUL, the measured-versus-predicted SOH trend, and the per-cycle health state, which moves from **healthy** toward a **warning** state as the predicted SOH falls with cycling.

## Citation

If you use this code or data, please cite the conference paper (see `CITATION.cff`) and the NASA PCoE battery dataset (Saha and Goebel, 2007) above.

## License

The code in this repository is released under the MIT License (see `LICENSE`). The NASA battery data in `data/` are U.S. Government work in the public domain, attributed to the NASA Prognostics Center of Excellence.
