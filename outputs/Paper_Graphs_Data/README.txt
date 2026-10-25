Graph data of the paper 'AI-Enabled Digital Twin for Predictive Maintenance and Health Prognostics of Battery Energy Storage Systems'.

All_Graphs_Data.xlsx : all graph data, one sheet per graph.
FigX_*.csv : data of one graph; first column = x-axis, other columns = curves named as in the legend.
FigX_*.pdf / .png : the graph (PDF is vector, PNG is 300 dpi).
Fig1: framework diagram (.drawio is editable at app.diagrams.net). Fig4/Fig5: Digital Twin screenshots.

Fig2a_CE_histogram.csv: histogram bars of Fig. 2(a). Columns: Coulombic Efficiency (bin start), Coulombic Efficiency (bin end), CE Distribution (probability density)
Fig2a_CE_gaussian_fit.csv: Gaussian curve of Fig. 2(a); Mean CE = 1.0002, std = 0.0169. Columns: Coulombic Efficiency, Gaussian Fit
Fig2a_CE_values.csv: the 29 valid CE values used for Fig. 2(a). Columns: Charge cycle, Discharge cycle, Coulombic Efficiency
Fig2b_Energy_Correction.csv: curves of Fig. 2(b). Columns: Valid Cycle Pair, Cumulative Energy Imbalance (Wh), 3rd-Degree Polynomial Fit (Wh), Corrected Imbalance (Wh)
Fig3a_Actual_vs_Predicted_SOH.csv: curves of Fig. 3(a). Columns: Cycle, Actual SOH (%), LSTM (%), Hybrid CNN-LSTM (%)
Fig3b_Actual_vs_Predicted_RUL.csv: curves of Fig. 3(b). Columns: Cycle, Actual RUL (cycles), LSTM (cycles), Hybrid CNN-LSTM (cycles)
Table1_Metrics_LSTM_vs_Hybrid.csv: values of Table I. Columns: Model, Target, MSE, RMSE, MAE, MAPE (%), R2
