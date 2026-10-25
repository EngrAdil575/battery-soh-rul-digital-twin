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
OUT = "/content/drive/MyDrive/FYP_NASA_Battery/paper_results"
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
all_data = []
for name in BATTERIES:
    cycles = loadmat(os.path.join(battery_path, f"{name}.mat"))[name][0, 0]["cycle"]
    records = []
    for i in range(len(cycles[0])):
        c = cycles[0, i]
        if c["type"][0] != "discharge":
            continue
        d = c["data"][0, 0]
        voltage = d["Voltage_measured"].flatten()
        current = d["Current_measured"].flatten()
        temperature = d["Temperature_measured"].flatten()
        time = d["Time"].flatten()
        records.append({
            "Battery": name,
            "Cycle": i + 1,
            "Capacity_Ah": float(d["Capacity"].flatten()[0]),
            "Voltage_mean": np.mean(voltage),
            "Voltage_min": np.min(voltage),
            "Voltage_max": np.max(voltage),
            "Current_mean": np.mean(current),
            "Temperature_mean": np.mean(temperature),
            "Temperature_max": np.max(temperature),
            "Discharge_time_s": time[-1] - time[0],
        })
    battery_df = pd.DataFrame(records)
    initial = battery_df["Capacity_Ah"].iloc[0]
    battery_df["SOH"] = battery_df["Capacity_Ah"] / initial
    eol_rows = battery_df.index[battery_df["Capacity_Ah"] <= initial * 0.70]
    eol_cycle = battery_df.loc[eol_rows[0], "Cycle"] if len(eol_rows) else battery_df["Cycle"].iloc[-1]
    battery_df["RUL"] = np.maximum(eol_cycle - battery_df["Cycle"], 0)
    all_data.append(battery_df)

df = pd.concat(all_data, ignore_index=True)
df.to_csv(os.path.join(OUT, "cycle_level_dataset.csv"), index=False)
print(df.groupby("Battery").size())
print(df.shape)

# ===== Cell 3 =====
cycles = loadmat(os.path.join(battery_path, "B0005.mat"))["B0005"][0, 0]["cycle"]

def calculate_charge_capacity(current, time):
    current, time = np.asarray(current).flatten(), np.asarray(time).flatten()
    dt = np.diff(time)
    current_mid = (current[:-1] + current[1:]) / 2
    return np.sum(np.abs(current_mid) * dt) / 3600          # Ah

# charge -> discharge pairs
pairs = []
for i in range(cycles.shape[1] - 1):
    if cycles[0, i]["type"][0] == "charge" and cycles[0, i + 1]["type"][0] == "discharge":
        pairs.append({"Charge_Cycle": i + 1, "Discharge_Cycle": i + 2})
pairs_df = pd.DataFrame(pairs)

ce_rows = []
for _, pair in pairs_df.iterrows():
    charge_data = cycles[0, int(pair["Charge_Cycle"]) - 1]["data"][0, 0]
    discharge_data = cycles[0, int(pair["Discharge_Cycle"]) - 1]["data"][0, 0]
    q_charge = calculate_charge_capacity(charge_data["Current_charge"], charge_data["Time"])
    q_discharge = float(np.asarray(discharge_data["Capacity"]).flatten()[0])
    ce_rows.append({"Charge_Cycle": int(pair["Charge_Cycle"]), "Discharge_Cycle": int(pair["Discharge_Cycle"]),
                    "Q_charge_Ah": q_charge, "Q_discharge_Ah": q_discharge,
                    "CE": q_discharge / q_charge if q_charge > 0 else np.nan})
ce_df = pd.DataFrame(ce_rows)

ce_valid = ce_df[(ce_df["Q_charge_Ah"] > 1.0) & (ce_df["Q_charge_Ah"] < 2.5)].copy()
print("Charge-discharge pairs:", len(pairs_df))
print("Valid CE pairs:", len(ce_valid))
print("Mean CE:", round(ce_valid["CE"].mean(), 4))

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
from sklearn.preprocessing import StandardScaler

features = ["Cycle", "Voltage_mean", "Voltage_min", "Voltage_max",
            "Current_mean", "Temperature_mean", "Temperature_max", "Discharge_time_s"]
train_df = df[df["Battery"].isin(["B0005", "B0006", "B0007"])].copy()
test_df = df[df["Battery"] == "B0018"].copy()

x_scaler = StandardScaler().fit(train_df[features].values)
y_scaler = StandardScaler().fit(train_df[["SOH", "RUL"]].values)

sequence_length = 20

def make_sequences(dataframe):
    X = x_scaler.transform(dataframe[features].values)
    y = y_scaler.transform(dataframe[["SOH", "RUL"]].values)
    X_seq = [X[i:i + sequence_length] for i in range(len(X) - sequence_length + 1)]
    y_seq = [y[i + sequence_length - 1] for i in range(len(X) - sequence_length + 1)]
    return np.array(X_seq), np.array(y_seq)

print("Training cycles:", len(train_df), " Test cycles:", len(test_df))

# ===== Cell 8 =====
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

N_SEEDS = 5

def set_seed(s):
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)

def build_model(kind):
    inp = keras.Input((sequence_length, len(features)))
    if kind == "LSTM":
        x = layers.LSTM(64)(inp)
    else:                                   # Hybrid CNN-LSTM
        x = layers.Conv1D(64, 3, padding="causal", activation="relu")(inp)
        x = layers.Conv1D(64, 3, padding="causal", activation="relu")(x)
        x = layers.LSTM(64)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    m = keras.Model(inp, layers.Dense(2)(x))
    m.compile(optimizer="adam", loss="mse")
    return m

X_tr, y_tr = zip(*[make_sequences(train_df[train_df["Battery"] == b]) for b in ["B0005", "B0006", "B0007"]])
X_tr, y_tr = np.concatenate(X_tr), np.concatenate(y_tr)
X_te, y_te = make_sequences(test_df)
actual = y_scaler.inverse_transform(y_te)
test_cycles = test_df["Cycle"].iloc[sequence_length - 1:].values

class Ensemble:
    """Mean prediction of the N_SEEDS trained networks."""
    def __init__(self, nets): self.nets = nets
    def predict(self, X, verbose=0): return np.mean([n.predict(X, verbose=0) for n in self.nets], axis=0)

MODEL_NAMES = ["LSTM", "Hybrid CNN-LSTM"]
NETS, HIST, PRED = {}, {}, {}
for kind in MODEL_NAMES:
    nets, hists = [], []
    for s in range(N_SEEDS):
        set_seed(s)
        net = build_model(kind)
        h = net.fit(X_tr, y_tr, validation_split=0.2, epochs=100, batch_size=16, verbose=0,
                    callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)])
        nets.append(net); hists.append(h.history)
    NETS[kind], HIST[kind] = Ensemble(nets), hists[0]
    PRED[kind] = y_scaler.inverse_transform(NETS[kind].predict(X_te))
    print(f"{kind}: trained {N_SEEDS} runs")

def metrics(y, p):
    nz = y != 0                                  # MAPE is undefined where the true value is 0
    return {"MSE": mean_squared_error(y, p), "RMSE": np.sqrt(mean_squared_error(y, p)),
            "MAE": mean_absolute_error(y, p), "MAPE (%)": 100 * np.mean(np.abs((p[nz] - y[nz]) / y[nz])),
            "R2": r2_score(y, p)}

rows = []
for kind in MODEL_NAMES:
    for j, target in enumerate(["SOH", "RUL"]):
        rows.append({"Model": kind, "Target": target, **metrics(actual[:, j], PRED[kind][:, j])})
RESULTS = pd.DataFrame(rows)
RESULTS.to_csv(os.path.join(OUT, "results_LSTM_vs_Hybrid.csv"), index=False)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")
print("\nTable I - Performance on the unseen B0018 battery")
print(RESULTS.to_string(index=False))

BEST = RESULTS[RESULTS.Target == "SOH"].sort_values("RMSE").iloc[0]["Model"]
print(f"\nDigital Twin will use: {BEST}")

# predictions behind Fig. 3
pd.DataFrame({"Cycle": test_cycles,
              "Actual SOH (%)": actual[:, 0] * 100,
              "LSTM SOH (%)": PRED["LSTM"][:, 0] * 100,
              "Hybrid CNN-LSTM SOH (%)": PRED["Hybrid CNN-LSTM"][:, 0] * 100,
              "Actual RUL (cycles)": actual[:, 1],
              "LSTM RUL (cycles)": PRED["LSTM"][:, 1],
              "Hybrid CNN-LSTM RUL (cycles)": PRED["Hybrid CNN-LSTM"][:, 1]}
             ).to_csv(os.path.join(OUT, "predictions_B0018.csv"), index=False)

# ===== Cell 9 =====
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

# ===== Cell 10 =====
STYLE = {
    "Actual":          dict(color="black",      ls="-",  lw=1.8, marker=None),
    "LSTM":            dict(color="tab:blue",   ls="--", lw=1.5, marker="s", ms=5, mfc="white", markevery=8),
    "Hybrid CNN-LSTM": dict(color="tab:brown",  ls="-",  lw=1.5, marker="^", ms=6, markevery=(4, 8)),
}
for j, (name, ylabel, scale) in enumerate([("SOH", "SOH (%)", 100), ("RUL", "RUL (cycles)", 1)]):
    plt.figure(figsize=(5, 3.1))
    plt.plot(test_cycles, actual[:, j] * scale, label=f"Actual {name}", **STYLE["Actual"])
    for kind in MODEL_NAMES:
        plt.plot(test_cycles, PRED[kind][:, j] * scale, label=kind, **STYLE[kind])
    plt.xlabel("Cycle"); plt.ylabel(ylabel)
    plt.legend(frameon=False); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"Actual_vs_Predicted_{name}.png"), dpi=300)
    plt.show()
print("Figures saved to", OUT)

# ===== Cell 11 =====
import math
import numpy as np
try:
    import gradio as gr
except ImportError:          # Colab: install once
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gradio'], check=True)
    import gradio as gr

# ---------- data used by the interface ----------
battery_view = test_df.reset_index(drop=True)
try:                                   # predictions of the model chosen in Section 5
    TW_CYC = np.asarray(test_cycles)
    TW_PRED = np.asarray(PRED[BEST])
    TW_ACT = np.asarray(actual)
    TW_MODEL = BEST
except NameError:                      # fallback: predict every window once
    X_all, y_all = make_sequences(test_df)
    TW_PRED = y_scaler.inverse_transform(model.predict(X_all, verbose=0))
    TW_ACT = y_scaler.inverse_transform(y_all)
    TW_CYC = test_df["Cycle"].values[sequence_length - 1:]
    TW_MODEL = "AI model"
RUL_MAX = float(max(TW_ACT[:, 1].max(), 1))
CE_REF = float(gaussian_mean) * 100
CE_STD = float(gaussian_std) * 100
FLEET = {b: df[df["Battery"] == b].reset_index(drop=True) for b in ["B0005", "B0006", "B0007", "B0018"]}
STATUS = {"HEALTHY": "#16a34a", "WARNING": "#f59e0b", "CRITICAL": "#dc2626"}
STATUS_TEXT = {"HEALTHY": "#15803d", "WARNING": "#b45309", "CRITICAL": "#b91c1c"}

def health(soh):
    return "HEALTHY" if soh >= 85 else ("WARNING" if soh >= 70 else "CRITICAL")

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
      <line x1="34" x2="{w-10}" y1="{Y(70):.1f}" y2="{Y(70):.1f}" stroke="#dc2626" stroke-dasharray="5 4"/>
      <text x="{w-12}" y="{Y(70)-4:.1f}" text-anchor="end" font-size="10" fill="#dc2626">EOL 70%</text>
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
      <div class="thr"><div><i style="background:#16a34a"></i><span style="color:#334155">Healthy ≥ 85%</span></div>
        <div><i style="background:#f59e0b"></i><span style="color:#334155">Warning 70–85%</span></div>
        <div><i style="background:#dc2626"></i><span style="color:#334155">Critical &lt; 70%</span></div></div>
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

final_interface.launch(share=True)
