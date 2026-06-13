
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


# PROTOCOL 1: INTRA-TOPOLOGY
# Train and test within the same topology

def discretizar_por_topologia_sem_leakage(train_df, test_df, metrics, topo_col="topologyNodes", n_bins=5):
    train_df = train_df.copy()
    test_df = test_df.copy()
    labels = labels_bins(n_bins)
    quantis = np.linspace(0.0, 1.0, n_bins + 1)
    cortes = {}
    for topo in sorted(train_df[topo_col].unique()):
        subset = train_df[train_df[topo_col] == topo]
        cortes[topo] = {}
        for m in metrics:
            cortes[topo][m] = quantis_monotonicos(subset[m].quantile(quantis).values)
    def discretizar(valor, bordas):
        idx = np.searchsorted(bordas[1:-1], valor, side="right")
        idx = max(0, min(idx, len(labels)-1))
        return labels[idx]
    for m in metrics:
        train_df[m+"_d"] = train_df.apply(lambda r: discretizar(r[m], cortes[r[topo_col]][m]), axis=1)
        test_df[m+"_d"] = test_df.apply(lambda r: discretizar(r[m], cortes[r[topo_col]][m]), axis=1)
    return train_df, test_df, labels

def treinar_cpts(train_df, disc_metrics, estados, classe_col="scenario", topo_col="topologyNodes", laplace=1.0):
    K = len(estados)
    cpts = {}
    for md in disc_metrics:
        counts = train_df.groupby([classe_col, topo_col, md]).size().unstack(fill_value=0)
        for estado in estados:
            if estado not in counts.columns:
                counts[estado] = 0
        counts = counts[estados]
        cpts[md] = (counts + laplace).div(counts.sum(axis=1) + laplace*K, axis=0)
    return cpts

def inferir(row, classes, disc_metrics, cpts, prior, topo, estados):
    K = len(estados)
    logs = []
    for cls in classes:
        logp = np.log(prior[cls])
        chave = (cls, topo)
        for md in disc_metrics:
            valor = row[md]
            prob = cpts[md].loc[chave, valor] if chave in cpts[md].index else 1.0/K
            logp += np.log(prob)
        logs.append(logp)
    probas = softmax(logs)
    return classes[int(np.argmax(logs))], probas.tolist()

def executar_intra_topologia(
    file_path="Dataset_Expermental_20Mil.csv",
    n_iteracoes=1,
    train_per_combo=200,
    test_per_combo=795,
    n_bins=5,
    seed=42,
    laplace=1.0,
    output_dir="dataset_experimental_20mil/resultados_intra_topologia"
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    df = adaptar_dataset_experimental(pd.read_csv(file_path))
    metrics = ["PDR", "delayMean_ms", "throughput_bps", "energyMean_J"]
    topologias = sorted(df["topologyNodes"].unique().tolist())
    cenarios = sorted(df["scenario"].unique().tolist())

    resultados = {n: {"acc": [], "precision": [], "recall": [], "f1": [], "auc": [], "cm": [], "y": [], "p": []} for n in topologias}

    for _ in range(n_iteracoes):
        train_idx, test_idx = [], []
        for n in topologias:
            for s in cenarios:
                subset = df[(df["topologyNodes"] == n) & (df["scenario"] == s)].index.to_numpy().copy()
                rng.shuffle(subset)
                if len(subset) < train_per_combo + test_per_combo:
                    cut = max(1, len(subset)//2)
                    train_idx.extend(subset[:cut]); test_idx.extend(subset[cut:])
                else:
                    train_idx.extend(subset[:train_per_combo])
                    test_idx.extend(subset[train_per_combo:train_per_combo+test_per_combo])
        train_df = df.loc[train_idx].copy()
        test_df = df.loc[test_idx].copy()
        train_df, test_df, estados = discretizar_por_topologia_sem_leakage(train_df, test_df, metrics, n_bins=n_bins)
        disc_metrics = [m+"_d" for m in metrics]
        cpts = treinar_cpts(train_df, disc_metrics, estados, laplace=laplace)
        prior = {s: 1.0/len(cenarios) for s in cenarios}
        for n in topologias:
            test_n = test_df[test_df["topologyNodes"] == n]
            y_real, y_pred, y_proba = [], [], []
            for _, row in test_n.iterrows():
                pred, probas = inferir(row, cenarios, disc_metrics, cpts, prior, n, estados)
                y_real.append(row["scenario"]); y_pred.append(pred); y_proba.append(probas)
            cm = confusion_matrix(y_real, y_pred, labels=cenarios)
            resultados[n]["acc"].append(accuracy_score(y_real, y_pred))
            resultados[n]["precision"].append(precision_score(y_real, y_pred, labels=cenarios, average="macro", zero_division=0))
            resultados[n]["recall"].append(recall_score(y_real, y_pred, labels=cenarios, average="macro", zero_division=0))
            resultados[n]["f1"].append(f1_score(y_real, y_pred, labels=cenarios, average="macro", zero_division=0))
            auc_macro, _ = roc_auc_ovr_macro(y_real, y_proba, cenarios)
            resultados[n]["auc"].append(auc_macro)
            resultados[n]["cm"].append(cm)
            resultados[n]["y"].extend(y_real)
            resultados[n]["p"].extend(y_proba)

    linhas = []
    print("\nRESULTADOS | INTRA-TOPOLOGIA")
    for n in topologias:
        acc_m, acc_s = mean_std(resultados[n]["acc"])
        pr_m, pr_s = mean_std(resultados[n]["precision"])
        re_m, re_s = mean_std(resultados[n]["recall"])
        f1_m, f1_s = mean_std(resultados[n]["f1"])
        auc_m, auc_s = mean_std(resultados[n]["auc"])
        f1_arr = np.asarray(resultados[n]["f1"])
        cm_rep = resultados[n]["cm"][int(np.argmin(np.abs(f1_arr - f1_m)))]
        plot_confusion(cm_rep, cenarios, f"Matriz de Confusão - Intra-topologia - {n} nós", out / f"matriz_confusao_intra_{n}_nos.png")
        auc_global = salvar_roc(resultados[n]["y"], resultados[n]["p"], cenarios, f"Curva ROC OvR - Intra-topologia - {n} nós", out / f"roc_intra_{n}_nos.png")
        linhas.append({"topologyNodes": n, "accuracy_mean": acc_m, "accuracy_std": acc_s, "precision_macro_mean": pr_m, "precision_macro_std": pr_s, "recall_macro_mean": re_m, "recall_macro_std": re_s, "f1_macro_mean": f1_m, "f1_macro_std": f1_s, "roc_auc_macro_mean": auc_m, "roc_auc_macro_std": auc_s, "roc_auc_global": auc_global})
        print(f"{n} nós | Acc={acc_m:.4f} | F1={f1_m:.4f} | AUC={auc_global:.4f}")
    resumo = pd.DataFrame(linhas)
    resumo.to_csv(out / "resumo_intra_topologia.csv", index=False)
    return resumo

if __name__ == "__main__":
    executar_intra_topologia()
