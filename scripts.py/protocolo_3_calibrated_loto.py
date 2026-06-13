
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

def softmax(logits):
    logits = np.asarray(logits, dtype=float)
    m = np.max(logits)
    exps = np.exp(logits - m)
    return exps / np.sum(exps)

def mean_std(values):
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))

def adaptar_dataset_experimental(df):
    required = ["Topology", "Attack_Type", "PDR_percent", "Avg_Delay_ms", "Throughput_kbps", "Energy_Consumed_J"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Colunas ausentes no dataset: " + ", ".join(missing))
    df = df.copy()
    df["topologyNodes"] = df["Topology"].astype(str).str.extract(r"(\d+)").astype(int)
    df["scenario"] = df["Attack_Type"].astype(str)
    df["PDR"] = df["PDR_percent"].astype(float)
    df["delayMean_ms"] = df["Avg_Delay_ms"].astype(float)
    df["throughput_bps"] = df["Throughput_kbps"].astype(float) * 1000.0
    df["energyMean_J"] = df["Energy_Consumed_J"].astype(float)
    return df

def labels_bins(n_bins):
    if n_bins == 3:
        return ["Low", "Medium", "High"]
    if n_bins == 5:
        return ["VeryLow", "Low", "Medium", "High", "VeryHigh"]
    return [f"B{i+1}" for i in range(n_bins)]

def quantis_monotonicos(v):
    v = np.asarray(v, dtype=float).copy()
    for i in range(1, len(v)):
        if v[i] <= v[i - 1]:
            v[i] = v[i - 1] + 1e-12
    return v

def plot_confusion(cm, labels, title, outpath):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    limiar = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm[i, j] > limiar else "black"
            ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", color=color)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel("Real")
    ax.set_xlabel("Predito")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()

def roc_auc_ovr_macro(y_true, y_score, classes):
    y_true_bin = label_binarize(y_true, classes=classes)
    y_score = np.asarray(y_score, dtype=float)
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(classes)
    return auc(all_fpr, mean_tpr), (all_fpr, mean_tpr, roc_auc, fpr, tpr)

def salvar_roc(y_real, y_proba, classes, title, outpath):
    roc_macro, pack = roc_auc_ovr_macro(y_real, y_proba, classes)
    all_fpr, mean_tpr, roc_auc_ind, fpr_ind, tpr_ind = pack
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(all_fpr, mean_tpr, label=f"macro-average (AUC={roc_macro:.3f})")
    for i, cls in enumerate(classes):
        ax.plot(fpr_ind[i], tpr_ind[i], label=f"{cls} (AUC={roc_auc_ind[i]:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()
    return roc_macro


# PROTOCOL 3: CALIBRATED LOTO
# Train on 3 topologies and use Normal samples from the test topology 
# to calibrate the baseline


def calcular_baseline_por_topologia(df, metricas_base, topo_col="topologyNodes", classe_col="scenario", baseline_label="Normal"):
    baseline_df = df[df[classe_col] == baseline_label].copy()
    if baseline_df.empty:
        raise ValueError(f"Nenhuma amostra {baseline_label} encontrada para calcular baseline.")
    return baseline_df.groupby(topo_col)[metricas_base].agg(["mean", "std"])

def aplicar_baseline(df, baseline, metricas_base, topo_col="topologyNodes"):
    df = df.copy()
    for m in metricas_base:
        medias = baseline[(m, "mean")]
        desvios = baseline[(m, "std")].replace(0, 1e-9)
        df[f"{m}_zbase"] = ((df[m] - df[topo_col].map(medias)) / df[topo_col].map(desvios)).astype(float)
    eps = 1e-9
    df["efficiency_tp_energy"] = df["throughput_bps"] / (df["energyMean_J"] + eps)
    df["delay_over_pdr"] = df["delayMean_ms"] / (df["PDR"] + eps)
    df["energy_over_pdr"] = df["energyMean_J"] / (df["PDR"] + eps)
    df["delay_log"] = np.log1p(df["delayMean_ms"])
    df["energy_log"] = np.log1p(df["energyMean_J"])
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)

def baseline_calibrado_teste(calib_df, topo_teste, metricas_base):
    calib_df = calib_df.copy()
    calib_df["topologyNodes"] = topo_teste
    return calib_df.groupby("topologyNodes")[metricas_base].agg(["mean", "std"])

def discretizar_global_sem_leakage(train_df, test_df, metrics, n_bins=5):
    train_df = train_df.copy()
    test_df = test_df.copy()
    labels = labels_bins(n_bins)
    quantis = np.linspace(0.0, 1.0, n_bins + 1)
    cortes = {m: quantis_monotonicos(train_df[m].quantile(quantis).values) for m in metrics}
    def discretizar(valor, bordas):
        idx = np.searchsorted(bordas[1:-1], valor, side="right")
        idx = max(0, min(idx, len(labels)-1))
        return labels[idx]
    for m in metrics:
        train_df[m+"_d"] = train_df[m].apply(lambda v: discretizar(v, cortes[m]))
        test_df[m+"_d"] = test_df[m].apply(lambda v: discretizar(v, cortes[m]))
    return train_df, test_df, labels

def treinar_cpts(train_df, disc_metrics, estados, classe_col="scenario", laplace=1.0):
    K = len(estados)
    cpts = {}
    for md in disc_metrics:
        counts = train_df.groupby([classe_col, md]).size().unstack(fill_value=0)
        for estado in estados:
            if estado not in counts.columns:
                counts[estado] = 0
        counts = counts[estados]
        cpts[md] = (counts + laplace).div(counts.sum(axis=1) + laplace*K, axis=0)
    return cpts

def inferir(row, classes, disc_metrics, cpts, prior, estados):
    K = len(estados)
    logs = []
    for cls in classes:
        logp = np.log(prior[cls])
        for md in disc_metrics:
            valor = row[md]
            prob = cpts[md].loc[cls, valor] if cls in cpts[md].index else 1.0/K
            logp += np.log(prob)
        logs.append(logp)
    probas = softmax(logs)
    return classes[int(np.argmax(logs))], probas.tolist()

def executar_loto_calibrado(
    file_path="Dataset_Expermental_20Mil.csv",
    n_bins=5,
    laplace=1.0,
    n_calibracao_normal=200,
    seed=42,
    n_test_per_class=795,
    output_dir="dataset_experimental_20mil/resultados_loto_calibrado"
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    df = adaptar_dataset_experimental(pd.read_csv(file_path))
    metricas_originais = ["PDR", "delayMean_ms", "throughput_bps", "energyMean_J"]
    metricas_derivadas = ["PDR_zbase", "delayMean_ms_zbase", "throughput_bps_zbase", "energyMean_J_zbase", "efficiency_tp_energy", "delay_over_pdr", "energy_over_pdr", "delay_log", "energy_log"]
    metrics = metricas_originais + metricas_derivadas
    topologias = sorted(df["topologyNodes"].unique().tolist())
    cenarios = sorted(df["scenario"].unique().tolist())
    prior = {s: 1.0/len(cenarios) for s in cenarios}
    linhas = []
    print("\nRESULTADOS | LOTO CALIBRADO")
    for topo_teste in topologias:
        train_df = df[df["topologyNodes"] != topo_teste].copy()
        test_full_df = df[df["topologyNodes"] == topo_teste].copy()
        normal_idx = test_full_df[test_full_df["scenario"] == "Normal"].index.to_numpy().copy()
        rng.shuffle(normal_idx)
        n_cal = min(n_calibracao_normal, len(normal_idx)//2)
        if n_cal < 5:
            raise ValueError(f"Poucas amostras Normal para calibrar a topologia {topo_teste}.")
        calib_idx = normal_idx[:n_cal]
        calib_df = test_full_df.loc[calib_idx].copy()
        test_df = test_full_df.drop(index=calib_idx).copy()
        baseline_train = calcular_baseline_por_topologia(train_df, metricas_originais)
        train_df = aplicar_baseline(train_df, baseline_train, metricas_originais)
        baseline_test = baseline_calibrado_teste(calib_df, topo_teste, metricas_originais)
        test_df = aplicar_baseline(test_df, baseline_test, metricas_originais)
        train_df, test_df, estados = discretizar_global_sem_leakage(train_df, test_df, metrics, n_bins=n_bins)
        disc_metrics = [m+"_d" for m in metrics]
        cpts = treinar_cpts(train_df, disc_metrics, estados, laplace=laplace)
        y_real, y_pred, y_proba = [], [], []
        for _, row in test_df.iterrows():
            pred, probas = inferir(row, cenarios, disc_metrics, cpts, prior, estados)
            y_real.append(row["scenario"]); y_pred.append(pred); y_proba.append(probas)
        cm = confusion_matrix(y_real, y_pred, labels=cenarios)
        acc = accuracy_score(y_real, y_pred)
        prec = precision_score(y_real, y_pred, labels=cenarios, average="macro", zero_division=0)
        rec = recall_score(y_real, y_pred, labels=cenarios, average="macro", zero_division=0)
        f1m = f1_score(y_real, y_pred, labels=cenarios, average="macro", zero_division=0)
        auc_macro = salvar_roc(y_real, y_proba, cenarios, f"Curva ROC OvR - LOTO calibrado - Teste {topo_teste} nós", out / f"roc_loto_calibrado_teste_{topo_teste}_nos.png")
        plot_confusion(cm, cenarios, f"Matriz de Confusão - LOTO calibrado - Teste {topo_teste} nós", out / f"matriz_confusao_loto_calibrado_teste_{topo_teste}_nos.png")
        linhas.append({"topology_test": topo_teste, "calibration_normal_samples": n_cal, "test_samples_after_calibration": len(test_df), "accuracy": acc, "precision_macro": prec, "recall_macro": rec, "f1_macro": f1m, "roc_auc_macro": auc_macro})
        print(f"Teste {topo_teste} nós | Acc={acc:.4f} | F1={f1m:.4f} | AUC={auc_macro:.4f}")
    resumo = pd.DataFrame(linhas)
    resumo.to_csv(out / "resumo_loto_calibrado.csv", index=False)
    return resumo

if __name__ == "__main__":
    executar_loto_calibrado()
