# Battery Digital Twin - all code cells exported from Battery_Digital_Twin_Notebook.ipynb
# Hybrid CNN-LSTM for joint SOH and RUL prediction (NASA B0005/6/7 train, B0018 test).

# ===== Cell 1 =====
import zipfile, os, glob, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from google.colab import drive

drive.mount("/content/drive")

# Find the NASA battery .mat files anywhere in Drive; extract the zip only if they are not there yet
ROOT = "/content/drive/MyDrive"
OUT = "/content/drive/MyDrive/FYP_NASA_Battery/paper_results/revised"
os.makedirs(OUT, exist_ok=True)

def find_b0005():
    hits = glob.glob(os.path.join(ROOT, "**", "B0005.mat"), recursive=True)
    return os.path.dirname(hits[0]) if hits else None

battery_path = find_b0005()
if battery_path is None:
    zips = [z for z in glob.glob(os.path.join(ROOT, "**", "*.zip"), recursive=True) if "attery" in z]
    if not zips:
        raise FileNotFoundError("Upload the NASA '5. Battery Data Set' zip (or BatteryAgingARC-FY08Q4.zip) to MyDrive/FYP_NASA_Battery")
    out = os.path.join(ROOT, "FYP_NASA_Battery", "NASA_Battery_Data")
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(out)
    for inner in glob.glob(os.path.join(out, "**", "*FY08Q4*.zip"), recursive=True):
        with zipfile.ZipFile(inner) as z:
            z.extractall(os.path.join(os.path.dirname(inner), "BatteryAgingARC-FY08Q4"))
    battery_path = find_b0005()

BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
print("Battery files found in:", battery_path)
print(sorted(f for f in os.listdir(battery_path) if f.endswith(".mat")))
print("Results folder:", OUT)

# ===== Cell 2 =====
EOL_FRAC = 0.80
rows = []
for name in BATTERIES:
    cyc = loadmat(os.path.join(battery_path, f"{name}.mat"))[name][0, 0]["cycle"]
    k = 0
    for i in range(cyc.shape[1]):
        c = cyc[0, i]
        if c["type"][0] != "discharge":
            continue
        k += 1
        d = c["data"][0, 0]
        v, cu = d["Voltage_measured"].flatten(), d["Current_measured"].flatten()
        t, tm = d["Temperature_measured"].flatten(), d["Time"].flatten()
        rows.append({"Battery": name, "Cycle": k, "Record": i + 1,
                     "Capacity_Ah": float(d["Capacity"].flatten()[0]),
                     "Voltage_mean": v.mean(), "Voltage_min": v.min(), "Voltage_max": v.max(),
                     "Current_mean": cu.mean(), "Temperature_mean": t.mean(), "Temperature_max": t.max(),
                     "Discharge_time_s": tm[-1] - tm[0]})
df = pd.DataFrame(rows)

eol_rows = []
for b, g in df.groupby("Battery"):
    q0 = g["Capacity_Ah"].iloc[0]
    df.loc[g.index, "SOH"] = g["Capacity_Ah"] / q0
    eol = int(g.loc[g["Capacity_Ah"] <= EOL_FRAC * q0, "Cycle"].iloc[0])
    df.loc[g.index, "EOL_cycle"] = eol
    df.loc[g.index, "RUL"] = np.maximum(eol - g["Cycle"], 0)
    eol_rows.append({"Cell": b, "Cycles": len(g), "Q0 (Ah)": round(q0, 3), "EOL (80%)": eol,
                     "Min SOH (%)": round(100 * g["Capacity_Ah"].min() / q0, 1)})
eol_table = pd.DataFrame(eol_rows)
df.to_csv(os.path.join(OUT, "cycle_level_dataset.csv"), index=False)
print(df.shape)
print(eol_table.to_string(index=False))

# ===== Cell 3 =====
def calculate_charge_capacity(current, time):
    current, time = np.asarray(current).flatten(), np.asarray(time).flatten()
    current_mid = (current[:-1] + current[1:]) / 2
    return np.sum(np.abs(current_mid) * np.diff(time)) / 3600          # Ah

def coulombic_efficiency(name):
    cyc = loadmat(os.path.join(battery_path, f"{name}.mat"))[name][0, 0]["cycle"]
    rows = []
    for i in range(cyc.shape[1] - 1):
        if cyc[0, i]["type"][0] == "charge" and cyc[0, i + 1]["type"][0] == "discharge":
            ch, dis = cyc[0, i]["data"][0, 0], cyc[0, i + 1]["data"][0, 0]
            q_ch = calculate_charge_capacity(ch["Current_charge"], ch["Time"])
            q_dis = float(np.asarray(dis["Capacity"]).flatten()[0])
            rows.append({"Charge_Cycle": i + 1, "Discharge_Cycle": i + 2, "Q_charge_Ah": q_ch,
                         "Q_discharge_Ah": q_dis, "CE": q_dis / q_ch if q_ch > 0 else np.nan})
    ce = pd.DataFrame(rows)
    return cyc, ce, ce[(ce["Q_charge_Ah"] > 1.0) & (ce["Q_charge_Ah"] < 2.5)].copy()

ce_rows = []
for name in BATTERIES:
    _, ce_all, ce_ok = coulombic_efficiency(name)
    ce_rows.append({"Cell": name, "CE pairs": len(ce_ok), "Mean CE": round(ce_ok["CE"].mean(), 4),
                    "Std CE": round(ce_ok["CE"].std(ddof=0), 4)})
table_I = eol_table.merge(pd.DataFrame(ce_rows), on="Cell")
table_I.to_csv(os.path.join(OUT, "Table_I_cells_EOL_CE.csv"), index=False)
print("Table I\n", table_I.to_string(index=False))

# B0005 is used for the detailed diagnostics below (Fig. 2)
cycles, pairs_df, ce_valid = coulombic_efficiency("B0005")

# ===== Cell 4 =====
from scipy.stats import norm

ce_values = ce_valid["CE"].dropna().values
gaussian_mean, gaussian_std = norm.fit(ce_values)
print("Gaussian mean CE:", round(gaussian_mean, 4), " std:", round(gaussian_std, 4))

sample_size = 5
sample_means = np.array([np.mean(ce_values[i:i + sample_size])
                         for i in range(0, len(ce_values), sample_size)
                         if len(ce_values[i:i + sample_size]) == sample_size])
print("CLT: sample size", sample_size, "->", len(sample_means), "sample means;",
      "mean", round(sample_means.mean(), 4), " std", round(sample_means.std(), 4))

# ===== Cell 5 =====
def calculate_energy(voltage, current, time):
    voltage, current, time = (np.asarray(a).flatten() for a in (voltage, current, time))
    dt = np.diff(time)
    power_mid = (voltage[:-1] * current[:-1] + voltage[1:] * current[1:]) / 2
    return np.sum(power_mid * dt) / 3600                     # Wh

energy_pairs = []
for _, pair in pairs_df.iterrows():
    ch = cycles[0, int(pair["Charge_Cycle"]) - 1]["data"][0, 0]
    dis = cycles[0, int(pair["Discharge_Cycle"]) - 1]["data"][0, 0]
    e_ch = abs(calculate_energy(ch["Voltage_measured"], ch["Current_measured"], ch["Time"]))
    e_dis = abs(calculate_energy(dis["Voltage_measured"], dis["Current_measured"], dis["Time"]))
    energy_pairs.append({"Charge_Cycle": int(pair["Charge_Cycle"]), "Discharge_Cycle": int(pair["Discharge_Cycle"]),
                         "Charge_Energy_Wh": e_ch, "Discharge_Energy_Wh": e_dis, "Energy_Imbalance_Wh": e_ch - e_dis})
energy_pair_df = pd.DataFrame(energy_pairs)

energy_valid = energy_pair_df[energy_pair_df["Charge_Cycle"].isin(set(ce_valid["Charge_Cycle"]))].copy()
energy_valid["Cumulative_Imbalance_Wh"] = energy_valid["Energy_Imbalance_Wh"].cumsum()
x = np.arange(len(energy_valid))
coefficients = np.polyfit(x, energy_valid["Cumulative_Imbalance_Wh"].values, 3)
energy_valid["Fitted_Imbalance_Wh"] = np.poly1d(coefficients)(x)
energy_valid["Corrected_Imbalance_Wh"] = energy_valid["Cumulative_Imbalance_Wh"] - energy_valid["Fitted_Imbalance_Wh"]
print("Polynomial coefficients (a, b, c, d):", coefficients)

# ===== Cell 6 =====
from scipy.optimize import minimize

Cr = 2.0                                                      # rated capacity (Ah)
cycle_data = []
for _, pair in pairs_df.iterrows():
    for cycle_number, mode in [(int(pair["Charge_Cycle"]), "charge"), (int(pair["Discharge_Cycle"]), "discharge")]:
        data = cycles[0, cycle_number - 1]["data"][0, 0]
        current = np.asarray(data["Current_measured"]).flatten()
        time = np.asarray(data["Time"]).flatten()
        dt = np.diff(time)
        current_mid = (current[:-1] + current[1:]) / 2
        charge = np.abs(current_mid) * dt / 3600
        total = charge.sum()
        if total <= 0:
            continue
        reference = np.cumsum(charge) / total if mode == "charge" else 1 - np.cumsum(charge) / total
        cycle_data.append({"mode": mode, "current": current_mid, "dt": dt, "reference": reference})

def objective_final(params):
    nc, dc = params
    errors = []
    for item in cycle_data:
        coefficient = nc if item["mode"] == "charge" else dc
        soc0 = 0.0 if item["mode"] == "charge" else 1.0
        predicted = soc0 + np.cumsum(coefficient * np.abs(item["current"]) * item["dt"] / (3600 * Cr))
        errors.extend(item["reference"] - predicted)
    return np.mean(np.square(errors))

result_final = minimize(objective_final, x0=[1.0, 1.0], method="L-BFGS-B", bounds=[(0.0, 2.0), (0.0, 2.0)])
print("Optimization successful:", result_final.success)
print("Optimal nc:", result_final.x[0], " Optimal dc:", result_final.x[1])
print("Minimum MSE:", result_final.fun, " Iterations:", result_final.nit)

# ===== Cell 7 =====
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEQ = 20
N_SEEDS = 5
MODELS = ["LSTM", "Hybrid CNN-LSTM"]
FEATS_ALL = ["Cycle", "Voltage_mean", "Voltage_min", "Voltage_max", "Current_mean",
             "Temperature_mean", "Temperature_max", "Discharge_time_s"]
FEATURE_SETS = {"with Cycle": FEATS_ALL, "without Cycle": FEATS_ALL[1:]}

def windows(g, feats, xs, ys):
    X = xs.transform(g[feats].values); y = ys.transform(g[["SOH", "RUL"]].values)
    n = len(g) - SEQ + 1
    return np.array([X[i:i + SEQ] for i in range(n)]), np.array([y[i + SEQ - 1] for i in range(n)])

def build(kind, nfeat):
    inp = keras.Input((SEQ, nfeat))
    if kind == "LSTM":
        x = layers.LSTM(64)(inp)
    else:
        x = layers.Conv1D(64, 3, padding="causal", activation="relu")(inp)
        x = layers.Conv1D(64, 3, padding="causal", activation="relu")(x)
        x = layers.LSTM(64)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    m = keras.Model(inp, layers.Dense(2)(x))
    m.compile(optimizer="adam", loss="mse")
    return m

def scores(y, p):
    return {"RMSE": np.sqrt(mean_squared_error(y, p)), "MAE": mean_absolute_error(y, p), "R2": r2_score(y, p)}

# ===== Cell 8 =====
per_seed, ensemble, preds = [], [], []
for test_b in BATTERIES:
    train_bs = [b for b in BATTERIES if b != test_b]
    tr_df, te_df = df[df.Battery.isin(train_bs)], df[df.Battery == test_b]
    for fname, feats in FEATURE_SETS.items():
        xs = StandardScaler().fit(tr_df[feats].values)
        ys = StandardScaler().fit(tr_df[["SOH", "RUL"]].values)
        Xtr, Ytr, Xva, Yva = [], [], [], []
        for b in train_bs:
            Xb, Yb = windows(df[df.Battery == b], feats, xs, ys)
            cut = len(Xb) - int(round(0.2 * len(Xb)))
            Xtr.append(Xb[:cut - (SEQ - 1)]); Ytr.append(Yb[:cut - (SEQ - 1)])     # gap: no shared cycles
            Xva.append(Xb[cut:]); Yva.append(Yb[cut:])
        Xtr, Ytr, Xva, Yva = map(np.concatenate, (Xtr, Ytr, Xva, Yva))
        Xte, Yte = windows(te_df, feats, xs, ys)
        act = ys.inverse_transform(Yte)
        te_cyc = te_df["Cycle"].values[SEQ - 1:]
        upto = te_cyc <= int(te_df["EOL_cycle"].iloc[0])                          # RUL scored up to EOL
        for kind in MODELS:
            seed_preds = []
            for s in range(N_SEEDS):
                random.seed(s); np.random.seed(s); tf.random.set_seed(s)
                m = build(kind, len(feats))
                m.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=100, batch_size=16, verbose=0, shuffle=True,
                      callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)])
                p = ys.inverse_transform(m.predict(Xte, verbose=0)); seed_preds.append(p)
                per_seed.append({"Test cell": test_b, "Features": fname, "Model": kind, "Seed": s,
                                 **{f"SOH {k}": v for k, v in scores(act[:, 0], p[:, 0]).items()},
                                 **{f"RUL {k}": v for k, v in scores(act[upto, 1], p[upto, 1]).items()}})
                tf.keras.backend.clear_session()
            pe = np.mean(seed_preds, axis=0)
            ensemble.append({"Test cell": test_b, "Features": fname, "Model": kind,
                             **{f"SOH {k}": v for k, v in scores(act[:, 0], pe[:, 0]).items()},
                             **{f"RUL {k}": v for k, v in scores(act[upto, 1], pe[upto, 1]).items()}})
            preds.append(pd.DataFrame({"Test cell": test_b, "Features": fname, "Model": kind, "Cycle": te_cyc,
                                       "Actual SOH": act[:, 0], "Pred SOH": pe[:, 0],
                                       "Actual RUL": act[:, 1], "Pred RUL": pe[:, 1]}))
            print(f"{test_b} | {fname:13s} | {kind:15s} | SOH RMSE {ensemble[-1]['SOH RMSE']:.4f} | RUL RMSE {ensemble[-1]['RUL RMSE']:.2f}")

per_seed, ensemble, preds = pd.DataFrame(per_seed), pd.DataFrame(ensemble), pd.concat(preds, ignore_index=True)
per_seed.to_csv(os.path.join(OUT, "LOCO_per_seed.csv"), index=False)
ensemble.to_csv(os.path.join(OUT, "LOCO_ensemble_per_cell.csv"), index=False)
preds.to_csv(os.path.join(OUT, "LOCO_predictions.csv"), index=False)

# ---------- Table II: per-seed RMSE (mean ± std) per test cell + ensemble mean over cells ----------
def fmt(m, s, scale): return f"{m*scale:.2f} ± {s*scale:.2f}"
cols = [(f, k) for k in MODELS for f in FEATURE_SETS]
table_II = []
for b in BATTERIES:
    row = {"Test cell": b}
    for f, k in cols:
        g = per_seed[(per_seed["Test cell"] == b) & (per_seed.Features == f) & (per_seed.Model == k)]
        row[f"SOH RMSE x1e-2 | {k} | {f}"] = fmt(g["SOH RMSE"].mean(), g["SOH RMSE"].std(), 100)
        row[f"RUL RMSE | {k} | {f}"] = fmt(g["RUL RMSE"].mean(), g["RUL RMSE"].std(), 1)
    table_II.append(row)
row = {"Test cell": "Ensemble, mean over cells"}
for f, k in cols:
    g = ensemble[(ensemble.Features == f) & (ensemble.Model == k)]
    row[f"SOH RMSE x1e-2 | {k} | {f}"] = fmt(g["SOH RMSE"].mean(), g["SOH RMSE"].std(), 100)
    row[f"RUL RMSE | {k} | {f}"] = fmt(g["RUL RMSE"].mean(), g["RUL RMSE"].std(), 1)
table_II.append(row)
table_II = pd.DataFrame(table_II)
table_II.to_csv(os.path.join(OUT, "Table_II_LOCO_RMSE.csv"), index=False)

summary = ensemble.groupby(["Features", "Model"])[["SOH RMSE", "SOH R2", "RUL RMSE", "RUL R2"]].mean()
pd.set_option("display.float_format", lambda v: f"{v:.4f}")
print("\nTable II\n", table_II.T.to_string(header=False))
print("\nEnsemble, mean over the 4 held-out cells\n", summary)

# ===== Cell 9 =====
# =====================================================================================================
# 5B. Leave-One-Cell-Out with EXACTLY the settings of the accepted paper (Fold 4 = original experiment)
#   Same as the accepted paper: Cycle = NASA record index, EOL = 70% of initial capacity (last record if
#   not reached), 8 features, 20-cycle windows, LSTM(64) / CNN-LSTM (2x Conv1D 64, k=3), Adam, MSE, batch 16,
#   max 100 epochs, validation_split 0.2, EarlyStopping(patience 10), 5 seeds -> mean prediction.
#   Outputs (paper_results/revised/LOCO_original):
#     Table o_A: unseen-cell RMSE / MAE / MAPE for SOH and RUL, LSTM vs CNN-LSTM, + average
#     Table o_B: proposed CNN-LSTM on training cells (seen) vs the unseen cell, for every fold
#     Figure : actual vs predicted SOH for all 4 cells as unseen test cells (2 x 2)
#   Run time on a T4 GPU: about 30-40 minutes.
# =====================================================================================================
import os, random, numpy as np, pandas as pd, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

LOCO_OUT = os.path.join(OUT, "LOCO_original"); os.makedirs(LOCO_OUT, exist_ok=True)

# ---------- dataset with the accepted paper's definitions (built from the Section 2 table) ----------
df_orig = df.copy()
df_orig["Cycle"] = df_orig["Record"]                         # NASA record index, as in the accepted paper
for b, g in df_orig.groupby("Battery"):
    q0 = g["Capacity_Ah"].iloc[0]
    below = g.index[g["Capacity_Ah"] <= 0.70 * q0]
    eol = df_orig.loc[below[0], "Cycle"] if len(below) else g["Cycle"].iloc[-1]
    df_orig.loc[g.index, "RUL"] = np.maximum(eol - g["Cycle"], 0)

o_FEATS = ["Cycle", "Voltage_mean", "Voltage_min", "Voltage_max",
         "Current_mean", "Temperature_mean", "Temperature_max", "Discharge_time_s"]
o_SEQ, o_N_SEEDS = 20, 5
o_MODELS = {"LSTM": "LSTM", "Hybrid CNN-LSTM": "CNN-LSTM"}

def o_seqs(g, o_xs, o_ys):
    X = o_xs.transform(g[o_FEATS].values); y = o_ys.transform(g[["SOH", "RUL"]].values)
    n = len(g) - o_SEQ + 1
    return np.array([X[i:i + o_SEQ] for i in range(n)]), np.array([y[i + o_SEQ - 1] for i in range(n)])

def o_build(o_kind):
    inp = keras.Input((o_SEQ, len(o_FEATS)))
    if o_kind == "LSTM":
        x = layers.LSTM(64)(inp)
    else:
        x = layers.Conv1D(64, 3, padding="causal", activation="relu")(inp)
        x = layers.Conv1D(64, 3, padding="causal", activation="relu")(x)
        x = layers.LSTM(64)(x)
    x = layers.Dropout(0.2)(x); x = layers.Dense(32, activation="relu")(x)
    m = keras.Model(inp, layers.Dense(2)(x)); m.compile(optimizer="adam", loss="mse")
    return m

def o_scores(y, p):
    nz = y != 0
    return {"RMSE": np.sqrt(mean_squared_error(y, p)), "MAE": mean_absolute_error(y, p),
            "MAPE (%)": 100 * np.mean(np.abs((p[nz] - y[nz]) / y[nz])), "R2": r2_score(y, p)}

o_unseen_rows, o_seen_rows, o_seed_rows, o_curves = [], [], [], {}
for o_test_b in BATTERIES:
    o_train_bs = [b for b in BATTERIES if b != o_test_b]
    o_tr = df_orig[df_orig.Battery.isin(o_train_bs)]
    o_xs = StandardScaler().fit(o_tr[o_FEATS].values); o_ys = StandardScaler().fit(o_tr[["SOH", "RUL"]].values)
    o_Xtr, o_Ytr = zip(*[o_seqs(df_orig[df_orig.Battery == b], o_xs, o_ys) for b in o_train_bs])
    o_Xtr, o_Ytr = np.concatenate(o_Xtr), np.concatenate(o_Ytr)
    o_evals = {b: o_seqs(df_orig[df_orig.Battery == b], o_xs, o_ys) for b in BATTERIES}
    for o_kind, o_short in o_MODELS.items():
        o_preds = {b: [] for b in BATTERIES}
        for s in range(o_N_SEEDS):
            random.seed(s); np.random.seed(s); tf.random.set_seed(s)
            m = o_build(o_kind)
            m.fit(o_Xtr, o_Ytr, validation_split=0.2, epochs=100, batch_size=16, verbose=0,
                  callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)])
            for b in BATTERIES:
                o_preds[b].append(o_ys.inverse_transform(m.predict(o_evals[b][0], verbose=0)))
            a = o_ys.inverse_transform(o_evals[o_test_b][1]); p = o_preds[o_test_b][-1]
            o_seed_rows.append({"Unseen cell": o_test_b, "Model": o_short, "Seed": s,
                              "SOH RMSE": o_scores(a[:, 0], p[:, 0])["RMSE"], "RUL RMSE": o_scores(a[:, 1], p[:, 1])["RMSE"]})
            tf.keras.backend.clear_session()
        for b in BATTERIES:
            a = o_ys.inverse_transform(o_evals[b][1]); p = np.mean(o_preds[b], axis=0)
            row = {"Unseen cell": o_test_b, "Model": o_short, "Cell": b, "Role": "unseen (test)" if b == o_test_b else "seen (training)",
                   **{f"SOH {k}": v for k, v in o_scores(a[:, 0], p[:, 0]).items()},
                   **{f"RUL {k}": v for k, v in o_scores(a[:, 1], p[:, 1]).items()}}
            (o_unseen_rows if b == o_test_b else o_seen_rows).append(row)
            if b == o_test_b:
                o_curves[(o_test_b, o_short)] = (df_orig[df_orig.Battery == b]["Cycle"].values[o_SEQ - 1:], a[:, 0], p[:, 0])
        print(f"Unseen {o_test_b} | {o_short:8s} | SOH RMSE {o_unseen_rows[-1]['SOH RMSE']:.4f} | RUL RMSE {o_unseen_rows[-1]['RUL RMSE']:.2f}")

o_U = pd.DataFrame(o_unseen_rows); o_S = pd.DataFrame(o_seen_rows); o_SD = pd.DataFrame(o_seed_rows)
o_U.to_csv(os.path.join(LOCO_OUT, "LOCO_unseen_all_metrics.csv"), index=False)
o_S.to_csv(os.path.join(LOCO_OUT, "LOCO_seen_training_cells.csv"), index=False)
o_SD.to_csv(os.path.join(LOCO_OUT, "LOCO_per_seed.csv"), index=False)

# ---------- Table o_A: unseen cell, LSTM vs CNN-LSTM ----------
o_cols = ["SOH RMSE", "SOH MAE", "SOH MAPE (%)", "RUL RMSE", "RUL MAE"]
o_A = o_U.pivot(index="Unseen cell", columns="Model", values=o_cols)
o_A.loc["Average"] = o_A.mean()
o_A = o_A.reindex(columns=pd.MultiIndex.from_product([o_cols, ["LSTM", "CNN-LSTM"]]))
o_A.to_csv(os.path.join(LOCO_OUT, "Table_A_LOCO_unseen.csv"))
o_std = o_SD.groupby(["Unseen cell", "Model"])[["SOH RMSE", "RUL RMSE"]].std().unstack()
o_std.to_csv(os.path.join(LOCO_OUT, "Table_A_seed_std.csv"))

# ---------- Table o_B: proposed CNN-LSTM, seen (training) vs unseen cell ----------
o_B = pd.concat([o_S[o_S.Model == "CNN-LSTM"], o_U[o_U.Model == "CNN-LSTM"]])
o_B = o_B[["Unseen cell", "Cell", "Role", "SOH RMSE", "SOH MAE", "SOH MAPE (%)"]].sort_values(["Unseen cell", "Role", "Cell"])
o_B.to_csv(os.path.join(LOCO_OUT, "Table_B_CNNLSTM_seen_vs_unseen.csv"), index=False)

# ---------- Figure: actual vs predicted SOH for every cell as the unseen cell ----------
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 10, "legend.fontsize": 9})
o_fig, o_ax = plt.subplots(2, 2, figsize=(7.2, 5.2))
for k, b in enumerate(BATTERIES):
    a_ = o_ax[k // 2, k % 2]
    cyc, act, pl = o_curves[(b, "LSTM")]; _, _, pc = o_curves[(b, "CNN-LSTM")]
    a_.plot(cyc, 100 * act, color="black", lw=1.6, label="Actual SOH")
    a_.plot(cyc, 100 * pl, color="tab:blue", ls="--", lw=1.3, marker="s", ms=3.5, mfc="white", markevery=8, label="LSTM")
    a_.plot(cyc, 100 * pc, color="tab:brown", lw=1.3, marker="^", ms=4.5, markevery=(4, 8), label="CNN-LSTM")
    a_.set_title(f"Unseen cell {b}"); a_.grid(True, alpha=0.3)
    a_.set_xlabel("Cycle"); a_.set_ylabel("SOH (%)")
o_ax[0, 0].legend(frameon=False)
o_fig.tight_layout(); o_fig.savefig(os.path.join(LOCO_OUT, "Fig_LOCO_SOH_all_cells.png"), dpi=300); plt.show()

pd.set_option("display.float_format", lambda v: f"{v:.4f}")
print("\nTABLE o_A - unseen test cell (5-seed mean prediction)\n", o_A.round(4).to_string())
print("\nSeed o_std of RMSE\n", o_std.round(4).to_string())
print("\nTABLE o_B - proposed CNN-LSTM: training cells (seen) vs unseen cell\n", o_B.round(4).to_string(index=False))
o_wins = (o_A.loc[BATTERIES, ("SOH RMSE", "CNN-LSTM")] < o_A.loc[BATTERIES, ("SOH RMSE", "LSTM")]).sum()
print(f"\nCNN-LSTM has lower SOH RMSE on {o_wins} of 4 unseen cells; average SOH RMSE  LSTM "
      f"{o_A.loc['Average', ('SOH RMSE', 'LSTM')]:.4f}  vs  CNN-LSTM {o_A.loc['Average', ('SOH RMSE', 'CNN-LSTM')]:.4f}")
print("All files saved to", LOCO_OUT)

# ===== Cell 10 =====
plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "legend.fontsize": 9.5,
                     "xtick.labelsize": 10, "ytick.labelsize": 10})

# Fig. 2(a)
x_fit = np.linspace(ce_values.min(), ce_values.max(), 300)
plt.figure(figsize=(5, 3.1))
plt.hist(ce_values, bins=12, density=True, color="tab:blue", alpha=0.45,
         edgecolor="tab:blue", hatch="//", label="CE Distribution")
plt.plot(x_fit, norm.pdf(x_fit, gaussian_mean, gaussian_std), color="tab:brown", ls="-", lw=1.8,
         marker="^", ms=6, markevery=30, label="Gaussian Fit")
y_top = plt.gca().get_ylim()[1]
plt.plot([gaussian_mean] * 6, np.linspace(0, y_top, 6), color="black", ls="--", lw=1.5,
         marker="o", ms=5, markevery=[1, 2, 3, 4], label=f"Mean CE = {gaussian_mean:.4f}")
plt.ylim(0, y_top)
plt.xlabel("Coulombic Efficiency"); plt.ylabel("Probability Density")
plt.legend(frameon=False); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(OUT, "CE_Gaussian_Distribution_B0005.png"), dpi=300)
plt.show()

# Fig. 2(b)
xe = np.arange(1, len(energy_valid) + 1)
plt.figure(figsize=(5, 3.1))
plt.plot(xe, energy_valid["Cumulative_Imbalance_Wh"], color="black", ls="-", lw=1.8,
         marker="o", ms=5, markevery=(1, 3), label="Cumulative Energy Imbalance")
plt.plot(xe, energy_valid["Fitted_Imbalance_Wh"], color="tab:blue", ls="--", lw=1.5,
         marker="s", ms=5, mfc="white", markevery=3, label="3rd-Degree Polynomial Fit")
plt.plot(xe, energy_valid["Corrected_Imbalance_Wh"], color="tab:brown", ls="-", lw=1.5,
         marker="^", ms=6, markevery=3, label="Corrected Imbalance")
plt.xlabel("Valid Cycle Pair"); plt.ylabel("Energy Imbalance (Wh)")
plt.legend(frameon=False); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(OUT, "Energy_Correction.png"), dpi=300)
plt.show()

ce_valid.to_csv(os.path.join(OUT, "Fig2a_CE_values_B0005.csv"), index=False)
energy_valid.to_csv(os.path.join(OUT, "Fig2b_energy_correction_B0005.csv"), index=False)

# ===== Cell 11 =====
P_all = preds
P_all = P_all[P_all["Features"] == "without Cycle"]
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 9,
                     "xtick.labelsize": 9, "ytick.labelsize": 9})
STY = {"Actual": dict(color="black", ls="-", lw=1.6),
       "LSTM": dict(color="tab:blue", ls="--", lw=1.4, marker="s", ms=4, mfc="white", markevery=8),
       "Hybrid CNN-LSTM": dict(color="tab:brown", ls="-", lw=1.4, marker="^", ms=5, markevery=(4, 8))}
fig, ax = plt.subplots(2, 4, figsize=(12, 5.2))
for c, b in enumerate(BATTERIES):
    eol = int(df.loc[df.Battery == b, "EOL_cycle"].iloc[0])
    for kind in ["LSTM", "Hybrid CNN-LSTM"]:
        g = P_all[(P_all["Test cell"] == b) & (P_all["Model"] == kind)].sort_values("Cycle")
        if kind == "LSTM":
            ax[0, c].plot(g.Cycle, 100 * g["Actual SOH"], label="Actual", **STY["Actual"])
            ax[1, c].plot(g.Cycle[g.Cycle <= eol], g["Actual RUL"][g.Cycle <= eol], label="Actual", **STY["Actual"])
        ax[0, c].plot(g.Cycle, 100 * g["Pred SOH"], label=kind, **STY[kind])
        gr_ = g[g.Cycle <= eol]                                      # RUL is defined (and scored) up to EOL
        ax[1, c].plot(gr_.Cycle, gr_["Pred RUL"], label=kind, **STY[kind])
    for r in (0, 1):
        ax[r, c].axvline(eol, color="gray", ls=":", lw=1.2)
        ax[r, c].grid(True, alpha=0.3)
    ax[0, c].axhline(80, color="gray", ls="-.", lw=0.9)
    ax[0, c].set_title(f"Test cell {b} (EOL at cycle {eol})", fontsize=10)
    ax[1, c].set_xlabel("Discharge cycle")
ax[0, 0].set_ylabel("SOH (%)"); ax[1, 0].set_ylabel("RUL (cycles)")
ax[0, 0].legend(frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "Fig_LOCO_SOH_RUL.png"), dpi=300)
plt.show()

# ===== Cell 12 =====
# Fig. 5 — per-seed RMSE on each held-out cell (five seeds per box)
cfg = [("LSTM", "with Cycle", "LSTM, w/ Cycle"), ("LSTM", "without Cycle", "LSTM, w/o Cycle"),
       ("Hybrid CNN-LSTM", "with Cycle", "CNN-LSTM, w/ Cycle"), ("Hybrid CNN-LSTM", "without Cycle", "CNN-LSTM, w/o Cycle")]
sty = [dict(fc="white", ec="tab:blue", hatch=""), dict(fc="tab:blue", ec="tab:blue", hatch="", alpha=.55),
       dict(fc="white", ec="tab:brown", hatch="///"), dict(fc="tab:brown", ec="tab:brown", hatch="", alpha=.55)]
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8})
fig, ax = plt.subplots(2, 1, figsize=(3.5, 4.3), sharex=True)
for r, (col, lab, sc) in enumerate([("SOH RMSE", "SOH RMSE ($\\times10^{-2}$)", 100), ("RUL RMSE", "RUL RMSE (cycles)", 1)]):
    for k, (m, f, name) in enumerate(cfg):
        data = [per_seed[(per_seed["Test cell"] == c) & (per_seed.Model == m) & (per_seed.Features == f)][col].values * sc
                for c in BATTERIES]
        bp = ax[r].boxplot(data, positions=np.arange(4) * 5 + k, widths=0.8, patch_artist=True,
                           medianprops=dict(color="black", lw=1), flierprops=dict(marker="o", ms=2.5))
        for b in bp["boxes"]:
            b.set_facecolor(sty[k]["fc"]); b.set_edgecolor(sty[k]["ec"]); b.set_hatch(sty[k]["hatch"])
            if "alpha" in sty[k]: b.set_alpha(sty[k]["alpha"])
        for part in ("whiskers", "caps"):
            for l in bp[part]: l.set_color(sty[k]["ec"])
        if r == 0:
            ax[r].bar([np.nan], [np.nan], color=sty[k]["fc"], edgecolor=sty[k]["ec"], hatch=sty[k]["hatch"],
                      alpha=sty[k].get("alpha", 1), label=name)
    ax[r].set_ylabel(lab); ax[r].grid(True, axis="y", alpha=.3)
    for x in [4, 9, 14]: ax[r].axvline(x, color="gray", lw=.5)
ax[1].set_xticks(np.arange(4) * 5 + 1.5); ax[1].set_xticklabels(BATTERIES); ax[1].set_xlabel("Held-out cell")
ax[0].legend(ncol=2, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.32), columnspacing=0.8, handlelength=1.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "Fig_seed_variability.png"), dpi=300, bbox_inches="tight")
plt.show()

# ===== Cell 13 =====
import math
import numpy as np
try:
    import gradio as gr
except ImportError:          # Colab: install once
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gradio'], check=True)
    import gradio as gr

# ---------- data used by the interface: LSTM (no Cycle input), held-out cell B0018 ----------
P = preds[(preds["Test cell"] == "B0018") & (preds["Features"] == "without Cycle") & (preds["Model"] == "LSTM")].sort_values("Cycle")
TW_CYC = P["Cycle"].values.astype(int)
TW_PRED = P[["Pred SOH", "Pred RUL"]].values
TW_ACT = P[["Actual SOH", "Actual RUL"]].values
TW_MODEL = "LSTM (no Cycle input)"
test_df = df[df["Battery"] == "B0018"].copy()
battery_view = test_df.reset_index(drop=True)
RUL_MAX = float(max(TW_ACT[:, 1].max(), 1))
CE_REF = float(gaussian_mean) * 100
CE_STD = float(gaussian_std) * 100
FLEET = {b: df[df["Battery"] == b].reset_index(drop=True) for b in BATTERIES}
STATUS = {"HEALTHY": "#16a34a", "WARNING": "#f59e0b", "CRITICAL": "#dc2626"}
STATUS_TEXT = {"HEALTHY": "#15803d", "WARNING": "#b45309", "CRITICAL": "#b91c1c"}

def health(soh):
    return "HEALTHY" if soh >= 90 else ("WARNING" if soh >= 80 else "CRITICAL")

# ---------- SVG building blocks ----------
def battery_svg(pct, color, w=150, h=260, text=None, sub=None, glow=True):
    """Upright battery; the fill level shows pct (0-100)."""
    pct = max(0.0, min(100.0, pct))
    cap_w, cap_h, pad, stroke = w * 0.36, h * 0.06, w * 0.07, max(3, w * 0.045)
    body_y, body_h = cap_h, h - cap_h
    inner_h = body_h - 2 * pad
    fill_h = inner_h * pct / 100
    fid = f"g{abs(hash((pct, color, w))) % 10**8}"
    t = "" if text is None else f'<text x="{w/2}" y="{body_y + body_h*0.52}" text-anchor="middle" font-size="{w*0.2:.0f}" font-weight="700" fill="{"#fff" if pct >= 55 else "#0f172a"}">{text}</text>'
    s = "" if sub is None else f'<text x="{w/2}" y="{body_y + body_h*0.52 + w*0.16}" text-anchor="middle" font-size="{w*0.085:.0f}" fill="{"#f8fafc" if pct >= 55 else "#475569"}">{sub}</text>'
    return f'''<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" style="{'filter:drop-shadow(0 0 14px ' + color + '66)' if glow else ''}">
      <defs><linearGradient id="{fid}" x1="0" x2="1"><stop offset="0" stop-color="{color}" stop-opacity="0.75"/>
      <stop offset="0.5" stop-color="{color}"/><stop offset="1" stop-color="{color}" stop-opacity="0.75"/></linearGradient></defs>
      <rect x="{(w-cap_w)/2}" y="0" width="{cap_w}" height="{cap_h+stroke}" rx="{cap_h*0.4}" fill="#94a3b8"/>
      <rect x="{stroke/2}" y="{body_y}" width="{w-stroke}" height="{body_h-stroke/2}" rx="{w*0.12}" fill="#f1f5f9" stroke="#94a3b8" stroke-width="{stroke}"/>
      <rect x="{pad+stroke/2}" y="{body_y + pad + inner_h - fill_h}" width="{w-2*pad-stroke}" height="{fill_h}" rx="{w*0.06}" fill="url(#{fid})"/>
      <rect x="{pad+stroke/2}" y="{body_y + pad + inner_h - fill_h}" width="{w-2*pad-stroke}" height="{min(6, fill_h)}" rx="3" fill="#ffffff" opacity="0.25"/>
      {t}{s}</svg>'''

def rul_cells(rul, n=10):
    k = int(math.ceil(n * max(0.0, min(1.0, rul / RUL_MAX))))
    col = "#22c55e" if k > 6 else ("#f59e0b" if k > 3 else "#ef4444")
    cells = "".join(
        f'<div class="cell {"on" if i < k else ""}" style="{"background:" + col + ";box-shadow:0 0 8px " + col + "88" if i < k else ""}"></div>'
        for i in range(n))
    return f'<div class="rulbar">{cells}<div class="rulcap"></div></div>'

def gauge_svg(v, lo=2.5, hi=4.2, w=190):
    frac = max(0.0, min(1.0, (v - lo) / (hi - lo)))
    r, cx, cy = w * 0.4, w / 2, w * 0.5
    def pt(f, rr):
        a = math.pi * (1 - f)
        return cx + rr * math.cos(a), cy - rr * math.sin(a)
    x0, y0 = pt(0, r); x1, y1 = pt(1, r); xv, yv = pt(frac, r)
    nx, ny = pt(frac, r * 0.78)
    ticks = "".join(f'<line x1="{pt(f, r*0.86)[0]:.1f}" y1="{pt(f, r*0.86)[1]:.1f}" x2="{pt(f, r*0.96)[0]:.1f}" y2="{pt(f, r*0.96)[1]:.1f}" stroke="#94a3b8" stroke-width="2"/>' for f in np.linspace(0, 1, 7))
    return f'''<svg width="{w}" height="{w*0.5+22}" viewBox="0 0 {w} {w*0.5+22}">
      <path d="M{x0:.1f},{y0:.1f} A{r},{r} 0 0 1 {x1:.1f},{y1:.1f}" stroke="#e2e8f0" stroke-width="14" fill="none" stroke-linecap="round"/>
      <path d="M{x0:.1f},{y0:.1f} A{r},{r} 0 0 1 {xv:.1f},{yv:.1f}" stroke="#0284c7" stroke-width="14" fill="none" stroke-linecap="round"/>
      {ticks}
      <line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#1e293b" stroke-width="3" stroke-linecap="round"/>
      <circle cx="{cx}" cy="{cy}" r="6" fill="#1e293b"/>
      <text x="{x0:.0f}" y="{cy+16}" text-anchor="middle" font-size="11" fill="#64748b">{lo}</text>
      <text x="{x1:.0f}" y="{cy+16}" text-anchor="middle" font-size="11" fill="#64748b">{hi}</text></svg>'''

def thermo_svg(t, lo=20, hi=45, h=150):
    frac = max(0.0, min(1.0, (t - lo) / (hi - lo)))
    col = "#0284c7" if t < 32 else ("#f59e0b" if t < 38 else "#dc2626")
    tube_h = h - 44; fill = tube_h * frac
    return f'''<svg width="54" height="{h}" viewBox="0 0 54 {h}">
      <rect x="18" y="6" width="18" height="{tube_h+8}" rx="9" fill="#f1f5f9" stroke="#94a3b8" stroke-width="2.5"/>
      <rect x="22" y="{10 + tube_h - fill}" width="10" height="{fill+8}" rx="5" fill="{col}"/>
      <circle cx="27" cy="{h-20}" r="15" fill="{col}" stroke="#94a3b8" stroke-width="2.5"/></svg>'''

def trend_svg(i, w=460, h=230):
    soh_a, soh_p, cyc = TW_ACT[:i+1, 0] * 100, TW_PRED[:i+1, 0] * 100, TW_CYC[:i+1]
    lo, hi = 65, 100
    X = lambda c: 34 + (c - TW_CYC[0]) / max(TW_CYC[-1] - TW_CYC[0], 1) * (w - 44)
    Y = lambda v: 10 + (hi - v) / (hi - lo) * (h - 30)
    pa = " ".join(f"{X(c):.1f},{Y(v):.1f}" for c, v in zip(cyc, soh_a))
    pp = " ".join(f"{X(c):.1f},{Y(v):.1f}" for c, v in zip(cyc, soh_p))
    grid = "".join(f'<line x1="34" x2="{w-10}" y1="{Y(g):.1f}" y2="{Y(g):.1f}" stroke="#e2e8f0"/><text x="28" y="{Y(g)+4:.1f}" text-anchor="end" font-size="10" fill="#64748b">{g}</text>' for g in (70, 80, 90, 100))
    return f'''<svg viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet">{grid}
      <line x1="34" x2="{w-10}" y1="{Y(80):.1f}" y2="{Y(80):.1f}" stroke="#dc2626" stroke-dasharray="5 4"/>
      <text x="{w-12}" y="{Y(80)-4:.1f}" text-anchor="end" font-size="10" fill="#dc2626">EOL 80%</text>
      <polyline points="{pa}" fill="none" stroke="#475569" stroke-width="1.6"/>
      <polyline points="{pp}" fill="none" stroke="#0284c7" stroke-width="2.4"/>
      <circle cx="{X(cyc[-1]):.1f}" cy="{Y(soh_p[-1]):.1f}" r="5" fill="#0284c7"/>
      <text x="34" y="{h-4}" font-size="10" fill="#64748b">cycle {int(TW_CYC[0])}</text>
      <text x="{w-10}" y="{h-4}" text-anchor="end" font-size="10" fill="#64748b">cycle {int(TW_CYC[-1])}</text></svg>'''

def fleet_html(cycle):
    out = []
    for b, d in FLEET.items():
        j = int(np.argmin(np.abs(d["Cycle"].values - cycle)))
        soh = float(d["SOH"].iloc[j]) * 100
        col = STATUS[health(soh)]
        role = "test (twin)" if b == "B0018" else "training"
        out.append(f'<div class="fcell {"me" if b == "B0018" else ""}">{battery_svg(soh, col, 46, 78, glow=False)}'
                   f'<b style="color:#0f172a">{b}</b><span style="color:#0f172a">{soh:.1f}%</span>'
                   f'<small style="color:#475569">{role}</small></div>')
    return "".join(out)

# ---------- page (landscape layout for a single screenshot) ----------
def final_digital_twin(cycle):
    i = int(np.argmin(np.abs(TW_CYC - int(cycle))))
    cyc = int(TW_CYC[i])
    row = battery_view.iloc[int(np.argmin(np.abs(battery_view["Cycle"].values - cyc)))]
    soh = float(TW_PRED[i, 0] * 100); rul = max(0.0, float(TW_PRED[i, 1]))
    st = health(soh); col = STATUS[st]
    return f"""
<div class="dt">
  <div class="top">
    <div class="ttl"><h1>⚡ Battery Digital Twin</h1><p>NASA battery B0018 &nbsp;•&nbsp; cycle {cyc} &nbsp;•&nbsp; AI model: {TW_MODEL}</p></div>
    <div class="status" style="background:{col}1f;border-color:{col};color:{STATUS_TEXT[st]}">● {st}</div>
  </div>
  <div class="grid">
    <div class="panel center soh">
      <div class="ptitle">State of Health</div>
      {battery_svg(soh, col, 120, 200, f"{soh:.1f}%", "AI predicted")}
      <div class="meas">measured {TW_ACT[i,0]*100:.1f}%</div>
      <div class="thr"><div><i style="background:#16a34a"></i><span style="color:#334155">Healthy ≥ 90%</span></div>
        <div><i style="background:#f59e0b"></i><span style="color:#334155">Warning 80–90%</span></div>
        <div><i style="background:#dc2626"></i><span style="color:#334155">Critical &lt; 80%</span></div></div>
    </div>
    <div class="col">
      <div class="panel">
        <div class="ptitle">Remaining Useful Life</div>
        <div class="rulrow"><div class="big" style="color:{STATUS_TEXT[st]}">{rul:.0f}<span> cycles</span></div>{rul_cells(rul)}</div>
        <div class="meas">each cell ≈ {RUL_MAX/10:.0f} cycles &nbsp;•&nbsp; measured {TW_ACT[i,1]:.0f}</div>
      </div>
      <div class="tiles">
        <div class="panel center"><div class="ptitle">Mean voltage</div>{gauge_svg(float(row['Voltage_mean']), w=120)}<div class="val">{row['Voltage_mean']:.3f} V</div></div>
        <div class="panel center flex"><div>{thermo_svg(float(row['Temperature_mean']), h=96)}</div>
          <div><div class="ptitle">Temperature</div><div class="val">{row['Temperature_mean']:.1f} °C</div><div class="meas">max {row['Temperature_max']:.1f} °C</div></div></div>
        <div class="panel center"><div class="ptitle">Current</div><div class="amp">⇣</div><div class="val">{row['Current_mean']:.3f} A</div><div class="meas">discharging</div></div>
        <div class="panel center flex"><div>{battery_svg(float(row['Capacity_Ah'])/2*100, "#06b6d4", 44, 72, glow=False)}</div>
          <div><div class="ptitle">Capacity</div><div class="val">{row['Capacity_Ah']:.3f} Ah</div><div class="meas">of 2.0 Ah rated</div></div></div>
      </div>
    </div>
    <div class="col">
      <div class="panel trend">
        <div class="ptitle">SOH trend — measured vs AI</div>
        {trend_svg(i)}
        <div class="legend"><i style="background:#475569"></i>measured <i style="background:#0284c7"></i>AI predicted</div>
      </div>
      <div class="diag">
        <div><span>Coulombic efficiency</span><strong>{CE_REF:.2f}%</strong><small>Gaussian ref., B0005</small></div>
        <div><span>Energy correction</span><strong>3rd-order poly.</strong><small>baseline method</small></div>
        <div><span>SOC optimization</span><strong>L-BFGS-B</strong><small>baseline method</small></div>
      </div>
    </div>
    <div class="panel fleetp">
      <div class="ptitle">Battery fleet · cycle {cyc}</div>
      <div class="fleet">{fleet_html(cyc)}</div>
    </div>
  </div>
</div>"""

css = """
.gradio-container{background:#f1f5f9 !important;max-width:1600px !important}
.dt{font-family:Inter,Segoe UI,Arial,sans-serif;color:#0f172a;padding:18px 20px;border-radius:18px;
    background:linear-gradient(150deg,#ffffff,#eef4fb);border:1px solid #cbd5e1;box-shadow:0 4px 18px rgba(15,23,42,.06)}
.top{display:flex;justify-content:space-between;align-items:center;gap:16px}
.ttl h1{margin:0;font-size:26px;color:#0369a1}.ttl p{margin:3px 0 0;color:#475569;font-size:14px}
.status{padding:8px 22px;border:2px solid;border-radius:999px;font-weight:800;font-size:18px;letter-spacing:.5px}
.grid{display:grid;grid-template-columns:190px 1.05fr 1.15fr 230px;gap:14px;margin-top:14px;align-items:stretch}
.col{display:flex;flex-direction:column;gap:14px}
.panel{background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;padding:12px 14px;box-shadow:0 1px 4px rgba(15,23,42,.05)}
.panel.center{text-align:center}.panel.flex{display:flex;align-items:center;justify-content:center;gap:10px;text-align:left}
.ptitle{font-size:12px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.6px;margin-bottom:8px}
.meas{font-size:11px;color:#64748b;margin-top:4px}
.rulrow{display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.big{font-size:40px;font-weight:800;line-height:1}.big span{font-size:14px;color:#64748b;font-weight:600}
.val{font-size:19px;font-weight:700;color:#0f172a;margin-top:2px}
.amp{font-size:40px;color:#7c3aed;line-height:1.1}
.tiles{display:grid;grid-template-columns:1fr 1fr;gap:14px;flex:1}
.rulbar{display:flex;align-items:center;gap:5px;padding:6px;border:3px solid #94a3b8;border-radius:10px;width:fit-content;background:#f8fafc}
.rulbar .cell{width:20px;height:32px;border-radius:4px;background:#e2e8f0}
.rulcap{width:7px;height:16px;background:#94a3b8;border-radius:0 4px 4px 0;margin-left:3px;margin-right:-13px}
.trend svg{width:100%;height:auto;display:block}
.legend{font-size:11px;color:#475569;margin-top:2px}.legend i{display:inline-block;width:14px;height:3px;margin:0 5px 3px 8px;vertical-align:middle}
.thr{margin-top:12px;text-align:left;font-size:11px;color:#475569;line-height:1.8;padding-left:8px}
.thr i{display:inline-block;width:9px;height:9px;border-radius:3px;margin-right:7px}
.fleet{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.fcell{display:flex;flex-direction:column;align-items:center;padding:8px 4px;border-radius:10px;font-size:13px;color:#0f172a}
.fcell.me{background:#e0f2fe;border:1px solid #7dd3fc}
.fcell b{margin-top:4px}.fcell span{font-size:16px;font-weight:700}.fcell small{color:#64748b;font-size:10px}
.diag{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.diag div{background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:10px;box-shadow:0 1px 4px rgba(15,23,42,.05)}
.diag span{display:block;color:#64748b;font-size:11px}.diag strong{display:block;font-size:15px;margin-top:3px;color:#0f172a}
.diag small{color:#64748b;font-size:10px}
@media(max-width:1100px){.grid{grid-template-columns:1fr 1fr}}
"""

# always use Gradio's light mode (otherwise it follows the browser's dark mode)
FORCE_LIGHT = """
() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'light') {
    url.searchParams.set('__theme', 'light');
    window.location.href = url.href;
  }
}
"""
css += """
.gradio-container .block, .gradio-container .form{background:#ffffff !important;border-color:#e2e8f0 !important}
.gradio-container .prose h2, .gradio-container .prose{color:#0f172a !important}
.dt .fcell b, .dt .fcell span{color:#0f172a !important}
.dt .fcell small{color:#475569 !important}
.dt .thr, .dt .thr div, .dt .thr span{color:#334155 !important}
.gradio-container input[type=number]{background:#f8fafc !important;color:#0f172a !important}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css, js=FORCE_LIGHT) as final_interface:
    gr.Markdown("## 🔋 AI-Enabled Battery Digital Twin")
    cycle_slider = gr.Slider(minimum=int(TW_CYC[0]), maximum=int(TW_CYC[-1]), value=int(TW_CYC[len(TW_CYC)//2]),
                             step=1, label="Select battery cycle")
    output = gr.HTML()
    cycle_slider.change(final_digital_twin, cycle_slider, output)
    final_interface.load(final_digital_twin, cycle_slider, output)

# which cycles are healthy / warning / critical (to choose the screenshot cycle)
_s = pd.DataFrame({"Cycle": TW_CYC, "Pred SOH (%)": 100 * TW_PRED[:, 0]})
_s["Status"] = _s["Pred SOH (%)"].apply(health)
print(_s.groupby("Status")["Cycle"].agg(["min", "max", "count"]))

final_interface.launch(share=True)

# ===== Cell 14 =====
# Export the Digital Twin as ONE standalone HTML file (opens in any browser, no Python/Colab needed)
# Run AFTER the Digital Twin cell. Saves: FYP_NASA_Battery/paper_results/revised/Digital_Twin_B0018.html
import json, os, html as _html

views = [final_digital_twin(int(c)) for c in TW_CYC]          # one dashboard per test cycle
start = 0                                                       # open at the first test cycle
page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI-Enabled Battery Digital Twin - NASA B0018</title>
<style>
body{{margin:0;background:#f1f5f9;font-family:Inter,Segoe UI,Arial,sans-serif;color:#0f172a}}
.wrap{{max-width:1500px;margin:0 auto;padding:18px}}
.bar{{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:12px 16px;margin-bottom:14px;
      display:flex;align-items:center;gap:14px;flex-wrap:wrap}}
.bar h2{{margin:0;font-size:18px}} .bar label{{font-weight:600;font-size:14px}}
.bar input[type=range]{{flex:1;min-width:200px;accent-color:#4f46e5}}
.bar output{{font-weight:700;min-width:90px}} .note{{font-size:12px;color:#64748b;margin-top:10px}}
{css}
</style></head><body><div class="wrap">
<div class="bar"><h2>🔋 AI-Enabled Battery Digital Twin</h2>
  <label for="cyc">Select battery cycle</label>
  <input id="cyc" type="range" min="0" max="{len(views)-1}" value="{start}" step="1">
  <output id="lbl"></output></div>
<div id="view"></div>
<div class="note">{TW_MODEL} predictions for the unseen NASA battery B0018 ({len(views)} cycles).
Offline software prototype that replays recorded NASA data; end of life = 80% SOH.</div>
</div>
<script>
const VIEWS = {json.dumps(views)};
const CYCLES = {json.dumps([int(c) for c in TW_CYC])};
const s = document.getElementById("cyc"), v = document.getElementById("view"), l = document.getElementById("lbl");
function show() {{ v.innerHTML = VIEWS[s.value]; l.textContent = "Cycle " + CYCLES[s.value]; }}
s.addEventListener("input", show); show();
</script></body></html>"""
path = os.path.join(OUT, "Digital_Twin_B0018.html")
with open(path, "w", encoding="utf-8") as f:
    f.write(page)
print("Saved:", path, f"({os.path.getsize(path)/1e6:.1f} MB) - download it and open in any browser")
try:                                   # Colab: download straight to the computer
    from google.colab import files; files.download(path)
except Exception:
    pass
