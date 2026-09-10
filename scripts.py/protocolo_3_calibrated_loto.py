"""
Protocol 3 — Calibrated Leave-One-Topology-Out (LOTO), 4-metric version
Paper 14 / SIoT 2026

Scientific rule implemented here:
- Train on N-1 topologies.
- The held-out topology is never used for model training.
- Use ONLY Ncal trusted Normal samples from the held-out topology to estimate
  local baseline mean/std.
- Remove those calibration samples from the final test set.
- Use exactly four calibrated evidences:
    PDR_zbase, delayMean_ms_zbase, throughput_bps_zbase, energyMean_J_zbase
- Learn the 5-state discretization thresholds ONLY from transformed training data.
- No additional engineered/derived features are used.
- Repeat the calibration-subset selection over 30 seeds to quantify variability.

Expected 30-seed mean accuracies on Dataset_Expermental_20Mil.csv:
36 -> ~0.9633
49 -> ~0.9864
64 -> ~0.9996
100 -> ~0.9983
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

DEFAULT_SEEDS = list(range(29)) + [42]

def softmax(logits):
    logits = np.asarray(logits, dtype=float)
    m = np.max(logits)
    exps = np.exp(logits - m)
    return exps / np.sum(exps)

def adaptar_dataset_experimental(df):
    required = [
        "Topology",
        "Attack_Type",
        "PDR_percent",
        "Avg_Delay_ms",
        "Throughput_kbps",
        "Energy_Consumed_J",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Colunas ausentes no dataset: " + ", ".join(missing))

    df = df.copy()
    df["topologyNodes"] = (
        df["Topology"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    )
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

def calcular_baseline_por_topologia(
    df,
    metricas_base,
    topo_col="topologyNodes",
    classe_col="scenario",
    baseline_label="Normal",
):
    """
    Baseline de TREINO:
    calcula média/desvio usando somente amostras Normal das topologias de treino.
    """
    baseline_df = df[df[classe_col] == baseline_label].copy()
    if baseline_df.empty:
        raise ValueError(
            f"Nenhuma amostra {baseline_label} encontrada para calcular baseline."
        )
    return baseline_df.groupby(topo_col)[metricas_base].agg(["mean", "std"])

def baseline_calibrado_teste(calib_df, topo_teste, metricas_base):
    """
    Baseline do ALVO:
    usa somente as Ncal amostras Normal confiáveis separadas para calibração.
    """
    calib_df = calib_df.copy()
    calib_df["topologyNodes"] = topo_teste
    return calib_df.groupby("topologyNodes")[metricas_base].agg(["mean", "std"])

def aplicar_baseline_4metricas(df, baseline, metricas_base, topo_col="topologyNodes"):
  
    df = df.copy()

    for m in metricas_base:
        medias = baseline[(m, "mean")]
        desvios = baseline[(m, "std")].replace(0, 1e-9)

        df[f"{m}_zbase"] = (
            (df[m] - df[topo_col].map(medias))
            / df[topo_col].map(desvios)
        ).astype(float)

    return df.replace([np.inf, -np.inf], np.nan).fillna(0)

def discretizar_global_sem_leakage(train_df, test_df, metrics, n_bins=5):
    """
    Aprende os cortes/quantis somente no conjunto de TREINO transformado
    e aplica os mesmos cortes ao conjunto de TESTE.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    labels = labels_bins(n_bins)
    quantis = np.linspace(0.0, 1.0, n_bins + 1)

    cortes = {
        m: quantis_monotonicos(train_df[m].quantile(quantis).values)
        for m in metrics
    }

    def discretizar(valor, bordas):
        idx = np.searchsorted(bordas[1:-1], valor, side="right")
        idx = max(0, min(idx, len(labels) - 1))
        return labels[idx]

    for m in metrics:
        train_df[m + "_d"] = train_df[m].apply(
            lambda v: discretizar(v, cortes[m])
        )
        test_df[m + "_d"] = test_df[m].apply(
            lambda v: discretizar(v, cortes[m])
        )

    return train_df, test_df, labels

def treinar_cpts(
    train_df,
    disc_metrics,
    estados,
    classe_col="scenario",
    laplace=1.0,
):
    """
    Estima CPTs P(M_i | C) para as evidências discretizadas.
    Laplace smoothing evita probabilidades nulas.
    """
    K = len(estados)
    cpts = {}

    for md in disc_metrics:
        counts = (
            train_df.groupby([classe_col, md])
            .size()
            .unstack(fill_value=0)
        )

        for estado in estados:
            if estado not in counts.columns:
                counts[estado] = 0

        counts = counts[estados]
        cpts[md] = (counts + laplace).div(
            counts.sum(axis=1) + laplace * K,
            axis=0,
        )

    return cpts


def inferir(row, classes, disc_metrics, cpts, prior, estados):
    K = len(estados)
    logs = []

    for cls in classes:
        logp = np.log(prior[cls])

        for md in disc_metrics:
            valor = row[md]
            if cls in cpts[md].index:
                prob = cpts[md].loc[cls, valor]
            else:
                prob = 1.0 / K

            logp += np.log(prob)

        logs.append(logp)

    probas = softmax(logs)
    pred = classes[int(np.argmax(logs))]
    return pred, probas.tolist()


def roc_auc_ovr_macro(y_true, y_score, classes):
    """
    Macro ROC-AUC OvR.
    """
    y_true_bin = label_binarize(y_true, classes=classes)
    y_score = np.asarray(y_score, dtype=float)

    fpr, tpr = {}, {}

    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])

    all_fpr = np.unique(
        np.concatenate([fpr[i] for i in range(len(classes))])
    )

    mean_tpr = np.zeros_like(all_fpr)

    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= len(classes)
    return auc(all_fpr, mean_tpr)

def executar_uma_seed(
    df,
    seed,
    n_bins=5,
    laplace=1.0,
    n_calibracao_normal=200,
):
    """
    Executa o Calibrated LOTO para as quatro topologias com uma seed.
    A seed controla SOMENTE a seleção aleatória das amostras Normal
    de calibração do ambiente-alvo.
    """
    rng = np.random.default_rng(seed)

    metricas_originais = [
        "PDR",
        "delayMean_ms",
        "throughput_bps",
        "energyMean_J",
    ]

    # IMPORTANTE: exatamente quatro evidências entram na BN calibrada.
    metrics = [
        "PDR_zbase",
        "delayMean_ms_zbase",
        "throughput_bps_zbase",
        "energyMean_J_zbase",
    ]

    topologias = sorted(df["topologyNodes"].unique().tolist())
    cenarios = sorted(df["scenario"].unique().tolist())
    prior = {s: 1.0 / len(cenarios) for s in cenarios}

    linhas = []

    for topo_teste in topologias:
        # 1) Separação estrita por topologia
        train_df = df[df["topologyNodes"] != topo_teste].copy()
        test_full_df = df[df["topologyNodes"] == topo_teste].copy()

        # 2) Calibração usa SOMENTE amostras Normal do alvo
        normal_idx = (
            test_full_df[test_full_df["scenario"] == "Normal"]
            .index.to_numpy()
            .copy()
        )
        rng.shuffle(normal_idx)

        n_cal = min(
            n_calibracao_normal,
            len(normal_idx) // 2,
        )

        if n_cal < 5:
            raise ValueError(
                f"Poucas amostras Normal para calibrar topologia {topo_teste}."
            )

        calib_idx = normal_idx[:n_cal]
        calib_df = test_full_df.loc[calib_idx].copy()

        # 3) As amostras de calibração são removidas do teste
        test_df = test_full_df.drop(index=calib_idx).copy()

        # 4) Baseline das topologias de treino: Normal-only
        baseline_train = calcular_baseline_por_topologia(
            train_df,
            metricas_originais,
        )
        train_df = aplicar_baseline_4metricas(
            train_df,
            baseline_train,
            metricas_originais,
        )

        # 5) Baseline local da topologia-alvo: Ncal Normal-only
        baseline_test = baseline_calibrado_teste(
            calib_df,
            topo_teste,
            metricas_originais,
        )
        test_df = aplicar_baseline_4metricas(
            test_df,
            baseline_test,
            metricas_originais,
        )

        # 6) Discretização aprende cortes SOMENTE no treino
        train_df, test_df, estados = discretizar_global_sem_leakage(
            train_df,
            test_df,
            metrics,
            n_bins=n_bins,
        )

        disc_metrics = [m + "_d" for m in metrics]

        # 7) CPTs somente com dados de treino
        cpts = treinar_cpts(
            train_df,
            disc_metrics,
            estados,
            laplace=laplace,
        )

        # 8) Inferência no held-out test
        y_real, y_pred, y_proba = [], [], []

        for _, row in test_df.iterrows():
            pred, probas = inferir(
                row,
                cenarios,
                disc_metrics,
                cpts,
                prior,
                estados,
            )
            y_real.append(row["scenario"])
            y_pred.append(pred)
            y_proba.append(probas)

        acc = accuracy_score(y_real, y_pred)
        prec = precision_score(
            y_real,
            y_pred,
            labels=cenarios,
            average="macro",
            zero_division=0,
        )
        rec = recall_score(
            y_real,
            y_pred,
            labels=cenarios,
            average="macro",
            zero_division=0,
        )
        f1m = f1_score(
            y_real,
            y_pred,
            labels=cenarios,
            average="macro",
            zero_division=0,
        )
        auc_macro = roc_auc_ovr_macro(
            y_real,
            y_proba,
            cenarios,
        )

        linhas.append(
            {
                "seed": seed,
                "topology": topo_teste,
                "calibration_normal_samples": n_cal,
                "test_samples_after_calibration": len(test_df),
                "accuracy": acc,
                "precision_macro": prec,
                "recall_macro": rec,
                "f1_macro": f1m,
                "roc_auc_macro": auc_macro,
            }
        )

    return pd.DataFrame(linhas)


def intervalo_95(media, sd, n):
    """
    IC95% usando t de Student com df=29 para n=30.
    t_(0.975,29) ~= 2.04523.
    """
    t_crit = 2.045229642132703
    half = t_crit * sd / np.sqrt(n)
    return media - half, media + half


def resumir_30_seeds(raw_df):
    linhas = []

    for topo, g in raw_df.groupby("topology"):
        n = len(g)

        acc_mean = g["accuracy"].mean()
        acc_sd = g["accuracy"].std(ddof=1)
        acc_lo, acc_hi = intervalo_95(acc_mean, acc_sd, n)

        f1_mean = g["f1_macro"].mean()
        f1_sd = g["f1_macro"].std(ddof=1)
        f1_lo, f1_hi = intervalo_95(f1_mean, f1_sd, n)

        auc_mean = g["roc_auc_macro"].mean()
        auc_sd = g["roc_auc_macro"].std(ddof=1)
        auc_lo, auc_hi = intervalo_95(auc_mean, auc_sd, n)

        linhas.append(
            {
                "topology": int(topo),
                "accuracy_mean": acc_mean,
                "accuracy_sd": acc_sd,
                "accuracy_ci_low": max(0.0, acc_lo),
                "accuracy_ci_high": min(1.0, acc_hi),
                "f1_macro_mean": f1_mean,
                "f1_macro_sd": f1_sd,
                "f1_macro_ci_low": max(0.0, f1_lo),
                "f1_macro_ci_high": min(1.0, f1_hi),
                "auc_macro_mean": auc_mean,
                "auc_macro_sd": auc_sd,
                "auc_macro_ci_low": max(0.0, auc_lo),
                "auc_macro_ci_high": min(1.0, auc_hi),
            }
        )

    return pd.DataFrame(linhas).sort_values("topology")


def executar_loto_calibrado_4metricas(
    file_path="Dataset/Dataset_Expermental_20Mil.csv",
    n_bins=5,
    laplace=1.0,
    n_calibracao_normal=200,
    seeds=None,
    output_dir="Results/calibrated_loto_4metrics",
):
    if seeds is None:
        seeds = DEFAULT_SEEDS

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = adaptar_dataset_experimental(pd.read_csv(file_path))

    resultados = []

    print("\nCALIBRATED LOTO — 4 METRICS ONLY")
    print(f"Seeds: {len(seeds)} | Ncal={n_calibracao_normal} | bins={n_bins}")

    for seed in seeds:
        r = executar_uma_seed(
            df=df,
            seed=seed,
            n_bins=n_bins,
            laplace=laplace,
            n_calibracao_normal=n_calibracao_normal,
        )
        resultados.append(r)

    raw = pd.concat(resultados, ignore_index=True)
    summary = resumir_30_seeds(raw)

    raw_path = out / "calibrated_loto_4metrics_30seeds_raw.csv"
    summary_path = out / "calibrated_loto_4metrics_30seeds_summary.csv"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nRESUMO FINAL")
    for _, row in summary.iterrows():
        print(
            f"{int(row['topology'])} nós | "
            f"Acc={row['accuracy_mean']:.4f} ± {row['accuracy_sd']:.4f} | "
            f"F1={row['f1_macro_mean']:.4f} ± {row['f1_macro_sd']:.4f} | "
            f"AUC={row['auc_macro_mean']:.4f} ± {row['auc_macro_sd']:.4f}"
        )

    print(f"\nRaw:     {raw_path}")
    print(f"Summary: {summary_path}")

    return raw, summary


if __name__ == "__main__":
    executar_loto_calibrado_4metricas()
