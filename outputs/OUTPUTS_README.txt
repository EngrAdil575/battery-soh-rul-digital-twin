Output files of the revised paper
"Deep Learning-Based State of Health and Remaining Useful Life Prediction for Lithium-Ion Batteries:
 A Leave-One-Cell-Out Comparison of LSTM and Hybrid CNN-LSTM"
All files come from one run of Revised_Paper_Code.ipynb (Google Colab, T4 GPU).

tables/
  Table_I_cells_EOL_CE.csv            Table I   cells, observed end of life (80% SOH), Coulombic efficiency
  Table_II_LOCO_RMSE.csv              Table II  leave-one-cell-out RMSE, mean +/- std over 5 seeds, with/without Cycle input
  Table_III_scope_comparison.csv      Table III scope comparison with battery digital-twin studies
  All_Tables.xlsx                     Tables I-III in one Excel file (one sheet each)

figures/
  Fig1_DT_architecture.pdf / .png     Fig. 1  digital-twin architecture
  Fig2a_CE_Gaussian_B0005.png         Fig. 2(a) Coulombic efficiency with Gaussian fit (B0005)
  Fig2b_Energy_Correction_B0005.png   Fig. 2(b) energy-imbalance correction (B0005)
  Fig3_LOCO_SOH_RUL.png               Fig. 3  actual vs predicted SOH and RUL for every held-out cell
  Fig4_Digital_Twin_B0018_cycle60.png Fig. 4  offline digital twin, cell B0018, cycle 60 (warning state)
  Fig5_Seed_Variability.png           Fig. 5  per-seed RMSE for each held-out cell

results_data/
  cycle_level_dataset.csv             one row per discharge cycle: features, SOH, RUL (EOL = 80% SOH)
  LOCO_per_seed.csv                   metrics of every model, feature set, held-out cell and seed
  LOCO_ensemble_per_cell.csv          metrics of the 5-seed ensembles per held-out cell
  LOCO_predictions.csv                actual and predicted SOH/RUL per cycle (data behind Fig. 3)
  Fig2a_CE_values_B0005.csv           data behind Fig. 2(a)
  Fig2b_energy_correction_B0005.csv   data behind Fig. 2(b)

digital_twin/
  Digital_Twin_B0018.html             standalone digital twin: open in any browser, move the slider over the cycles
