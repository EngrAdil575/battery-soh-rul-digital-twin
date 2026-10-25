# Deep Learning-Based State of Health and Remaining Useful Life Prediction for Lithium-Ion Batteries Using a Hybrid CNN-LSTM Architecture

Dataset, code, and output files for the conference paper:

> **Deep Learning-Based State of Health and Remaining Useful Life Prediction for Lithium-Ion Batteries Using a Hybrid CNN-LSTM Architecture**
> Muhammad Waiz Wais, Roya Mangal, Muhammad Sadiq Orakzai, Adil Zohaib, Sohail Khalid, Haris Sheh Zad.

This repository is the reproducibility record for the paper. It holds the battery cells used in the study, the analysis notebook, and the figures and tables reported in the paper.

## What the paper does

The paper proposes a hybrid deep-learning architecture that couples one-dimensional causal convolutional layers with an LSTM network to jointly predict State of Health (SOH) and Remaining Useful Life (RUL) from cycle-level voltage, current, temperature, and discharge-time features. The model is trained on NASA cells B0005, B0006, and B0007 and tested on the unseen cell B0018, against a standalone LSTM trained under identical settings. On B0018 the hybrid CNN-LSTM reaches an SOH RMSE of 0.0180 (R² = 0.9286) and lowers the RUL RMSE from 73.29 to 43.47 cycles relative to the LSTM. Coulombic-efficiency analysis, polynomial energy correction, and L-BFGS-B SOC optimization from an existing digital-twin baseline are adapted to the same dataset, and a software interface displays predicted SOH, RUL, and per-cycle health status.

## Repository layout

```
battery-soh-rul-digital-twin/
├── data/                              NASA battery cells used in the study (see data/DATASET_README.txt)
│   ├── B0005.mat  B0006.mat  B0007.mat    Training cells
│   ├── B0018.mat                          Unseen test cell
│   └── DATASET_README.txt
├── code/
│   ├── Battery_Digital_Twin_Notebook.ipynb   Google Colab notebook (full pipeline)
│   ├── Battery_Digital_Twin_AllCells.py      The notebook's code cells as a single script
│   └── requirements.txt
├── outputs/
│   ├── figures/                       Figs 1–5 (300–600 dpi PNG)
│   ├── tables/                        Tables I–II (CSV and xlsx)
│   ├── Paper_Graphs_Data/             Per-figure plot data (CSV), vector PDFs, and All_Graphs_Data.xlsx
│   └── OUTPUTS_README.txt
├── Offline_Digital_Twin_Battery_Health_Estimation.html   Standalone offline digital-twin dashboard (open in a browser)
├── CITATION.cff
└── LICENSE                            MIT (code). The NASA dataset is public-domain U.S. Government work.
```

## Dataset

The battery cells are from the NASA Prognostics Center of Excellence (PCoE) battery aging dataset:

> B. Saha and K. Goebel, "Battery data set," NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA, 2007.

Only the four cells used in the paper are included (`B0005`, `B0006`, `B0007`, `B0018`) as the original `.mat` files. The data are U.S. Government work in the public domain. The full repository and the other aging-ARC sets are available from NASA PCoE; see `data/DATASET_README.txt` for details.

## Reproducing the results

Open `code/Battery_Digital_Twin_Notebook.ipynb` in Google Colab and run the cells in order. Cell 1 locates the `.mat` files in Google Drive (or extracts the NASA ZIP); point it at the four cells in `data/`. The notebook builds the cycle-level dataset, runs the baseline diagnostics on B0005 (Coulombic efficiency, energy correction, SOC optimization), trains the LSTM and the hybrid CNN-LSTM over five seeds, writes the metrics of Table I, produces Figures 2–3, and launches the digital-twin interface (Figures 4–5).

`code/Battery_Digital_Twin_AllCells.py` is the same code as a single script. Results on a GPU can differ slightly between runs because GPU training is not bit-for-bit deterministic.

Dependencies are in `code/requirements.txt`: numpy, pandas, scipy, scikit-learn, tensorflow, matplotlib, and gradio (for the interface).

## Outputs

`outputs/figures/` holds Figures 1–5, `outputs/tables/` holds Tables I–II (CSV and `Tables_I_II.xlsx`), and `outputs/Paper_Graphs_Data/` holds the per-figure plot data and vector PDFs. `outputs/OUTPUTS_README.txt` maps each file to the paper.

## Offline digital twin

`Offline_Digital_Twin_Battery_Health_Estimation.html` is a standalone, offline digital-twin dashboard for the unseen test cell B0018. It needs no server or internet connection — open the file in any web browser.

Battery health is read from two places at once: the trend and diagnostic graphs, and the colour of the battery cells. Drag or scroll the horizontal battery-cycle bar at the top of the page to the right to step through the ageing cycles. As the cycle increases, the predicted State of Health falls and the cells shift from **green** (healthy) toward **orange** (warning), so the degradation and the point at which the battery enters a warning state can be seen directly from the colour change. The dashboard also shows the predicted SOH and RUL, the measured-versus-AI SOH trend, the baseline diagnostics, and the status of every cell in the fleet at the selected cycle.

## Citation

If you use this code or data, please cite the conference paper (see `CITATION.cff`) and the NASA PCoE battery dataset (Saha and Goebel, 2007) above.

## License

The code in this repository is released under the MIT License (see `LICENSE`). The NASA battery data in `data/` are U.S. Government work in the public domain, attributed to the NASA Prognostics Center of Excellence.
