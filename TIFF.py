# -*- coding: utf-8 -*-
"""
Automated MRI Feature Extraction and Machine Learning Pipeline for Olive Oil Content Estimation (TIFF 32-bit Format).

This script extracts morphological (flesh area, pit area, flesh-to-pit ratio) and pixel-level 
intensity statistics from 32-bit TIFF MRI slices, and trains Random Forest (RF) and Partial Least Squares (PLS) 
regression models using nested cross-validation grouped by physical tree.
"""

from itertools import product
import os
from glob import glob
from PIL import Image

import imageio.v3 as iio
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

# ============================
# CONFIGURATION
# ============================

IMAGES_DIR = r"MRI_IMAGES"
MASKS_FLESH_DIR = r"FLESH_MASKS"
MASKS_PIT_DIR = r"PIT_MASKS"
OIL_FILE = r"SOXHLET_OIL_CONTENT"

OUTPUT_OLIVES = "resultados_tiff32_por_oliva.csv"
OUTPUT_TREES = "resultados_tiff32_por_arbol.csv"
OUTPUT_PIXELS = "intensidades_pixeles_pulpa_tiff.csv"
OUTPUT_FEATURES = "features_por_arbol_tiff.csv"
EXCEL_OUTPUT = "resultados_modelo_randomforest_tiff_final_permutacion.xlsx"

N_ESTIMATORS = 500
RANDOM_STATE = 42
N_REPEATS_PERM = 50
MAX_PLS_COMP = 6
FEATS = ["mean", "std", "median", "p05", "p25", "p75", "p95", "n_pixeles"]

# Agrupación lógica de variables para Grouped Permutation Importance
FEATURE_GROUPS = {
    "Tendencia_Central": ["mean", "median"],
    "Extremos_Superiores": ["p75", "p95"],
    "Extremos_Inferiores": ["p05", "p25"],
    "Dispersión": ["std"],
    "Tamaño": ["n_pixeles"]
}

# Hyperparameter grid for Random Forest nested cross-validation
RF_GRID = {
    "min_samples_leaf": [1, 3, 5],
    "max_features": [1.0, 0.5, 0.33],
    "max_depth": [None, 5],
}
RF_DEFAULTS = {"min_samples_leaf": 1, "max_features": 1.0, "max_depth": None}


# ============================
# HELPER FUNCTIONS
# ============================

def load_tiff32(path):
    """Load 32-bit TIFF image file and return float32 numpy array."""
    img = iio.imread(path)
    return img.astype(np.float32)


def load_and_resize_mask(mask_path, target_shape):
    """Load binary mask image, resize to target shape, and return boolean array."""
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((target_shape[1], target_shape[0]), Image.NEAREST)
    return np.array(mask) > 0


def extract_tree_name(name):
    """Extract standard tree identifier from slice file name."""
    parts = name.split("_")
    if "arbol" not in parts:
        return "desconocido"
    idx = parts.index("arbol")
    num = parts[idx + 1]
    mes = parts[idx + 2]
    if parts[idx + 3].isdigit() and len(parts[idx + 3]) == 1:
        sufijo = "_" + parts[idx + 3]
        año = parts[idx + 4]
    else:
        sufijo = ""
        año = parts[idx + 3]
    return f"arbol_{num}_{mes}{sufijo}_{año}"


def add_groups(df):
    """Extract tree ID and harvest season for grouped cross-validation splits."""
    parts = df["arbol"].str.split("_")
    df["arbol_id"] = parts.str[1]
    df["cosecha"] = parts.str[-1]
    return df


def cv_metrics(y, pred):
    """Calculate standard regression metrics (R2, MAE, RMSE)."""
    return {
        "R2": r2_score(y, pred),
        "MAE": mean_absolute_error(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
    }


# ============================
# MACHINE LEARNING FIT & PREDICT
# ============================

def rf_fit_predict(Xtr, ytr, gtr, Xte):
    """Fit Random Forest regressor with nested GroupKFold hyperparameter tuning."""
    n_groups = pd.Series(np.asarray(gtr)).nunique()
    combos = [dict(zip(RF_GRID.keys(), v)) for v in product(*RF_GRID.values())]
    best = RF_DEFAULTS

    if n_groups >= 3:
        inner = GroupKFold(n_splits=min(5, n_groups))
        rmses = {}

        for i, params in enumerate(combos):
            errs = []
            for itr, ite in inner.split(Xtr, ytr, gtr):
                m = RandomForestRegressor(
                    n_estimators=N_ESTIMATORS,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    **params
                )
                m.fit(Xtr.iloc[itr], ytr.iloc[itr])
                errs.append(ytr.iloc[ite].values - m.predict(Xtr.iloc[ite]))

            rmses[i] = np.sqrt(np.mean(np.concatenate(errs)**2))

        best = combos[min(rmses, key=rmses.get)]

    m = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **best
    )
    m.fit(Xtr, ytr)

    info = (
        f"min_samples_leaf={best['min_samples_leaf']}; "
        f"max_features={best['max_features']}; "
        f"max_depth={best['max_depth']}"
    )

    return m.predict(Xte), info


def pls_fit_predict(Xtr, ytr, gtr, Xte):
    """Fit Partial Least Squares regressor with nested GroupKFold component tuning."""
    max_c = min(MAX_PLS_COMP, Xtr.shape[1], len(Xtr) - 1)
    best_c = 1
    n_groups = pd.Series(np.asarray(gtr)).nunique()

    if max_c > 1 and n_groups >= 3:
        inner = GroupKFold(n_splits=min(5, n_groups))
        rmses = {}

        for c in range(1, max_c + 1):
            errs = []
            for itr, ite in inner.split(Xtr, ytr, gtr):
                m = PLSRegression(n_components=c, scale=True)
                m.fit(Xtr.iloc[itr], ytr.iloc[itr])
                errs.append(ytr.iloc[ite].values - m.predict(Xtr.iloc[ite]).ravel())

            rmses[c] = np.sqrt(np.mean(np.concatenate(errs)**2))

        best_c = min(rmses, key=rmses.get)

    m = PLSRegression(n_components=best_c, scale=True)
    m.fit(Xtr, ytr)
    return m.predict(Xte).ravel(), f"n_comp={best_c}"


# ============================
# CROSS-VALIDATION & PERMUTATION IMPORTANCE
# ============================

def run_cv(df, feats, splitter, groups, label="", fit_predict=rf_fit_predict):
    """Run cross-validation loop and gather out-of-fold predictions and fold metrics."""
    X, y = df[feats], df["grasa"]
    inner_groups = df["arbol_id"]

    oof = np.full(len(df), np.nan)
    folds = []

    for k, (tr, te) in enumerate(splitter.split(X, y, groups)):
        pred, info = fit_predict(X.iloc[tr], y.iloc[tr], inner_groups.iloc[tr], X.iloc[te])
        oof[te] = pred

        met = cv_metrics(y.iloc[te], pred)
        met["info"] = info
        met["fold"] = k + 1
        met["n_train_samples"] = len(tr)
        met["n_test_samples"] = len(te)
        met["n_train_groups"] = inner_groups.iloc[tr].nunique()
        met["n_test_groups"] = inner_groups.iloc[te].nunique()
        met["grupos_test"] = ",".join(sorted(set(groups.iloc[te].astype(str))))
        folds.append(met)

        print(
            f"[{label}] fold {k+1} | Train: {len(tr)} obs, {met['n_train_groups']} grp | "
            f"Test: {len(te)} obs, {met['n_test_groups']} grp ({met['grupos_test']}) | "
            f"R2={met['R2']:.3f} | MAE={met['MAE']:.3f} | RMSE={met['RMSE']:.3f} | {info}"
        )

    overall = cv_metrics(y, oof)
    print(
        f"[{label}] OOF -> R2={overall['R2']:.3f} MAE={overall['MAE']:.3f} RMSE={overall['RMSE']:.3f}\n"
    )

    return overall, pd.DataFrame(folds), oof


def compute_rf_cv_permutation_importance(X, y, groups, inner_groups_full, splitter):
    """Compute out-of-fold grouped permutation feature importance across external CV folds for Random Forest."""
    fold_means = []
    group_names = list(FEATURE_GROUPS.keys())

    for tr, te in splitter.split(X, y, groups):
        Xtr, ytr, gtr = X.iloc[tr], y.iloc[tr], inner_groups_full.iloc[tr]
        Xte, yte = X.iloc[te], y.iloc[te]

        n_groups = pd.Series(np.asarray(gtr)).nunique()
        combos = [dict(zip(RF_GRID.keys(), v)) for v in product(*RF_GRID.values())]
        best = RF_DEFAULTS

        if n_groups >= 3:
            inner = GroupKFold(n_splits=min(5, n_groups))
            rmses = {}
            for i, params in enumerate(combos):
                errs = []
                for itr, ite in inner.split(Xtr, ytr, gtr):
                    m_inner = RandomForestRegressor(
                        n_estimators=N_ESTIMATORS,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        **params
                    )
                    m_inner.fit(Xtr.iloc[itr], ytr.iloc[itr])
                    errs.append(ytr.iloc[ite].values - m_inner.predict(Xtr.iloc[ite]))
                rmses[i] = np.sqrt(np.mean(np.concatenate(errs)**2))
            best = combos[min(rmses, key=rmses.get)]

        m = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **best
        )
        m.fit(Xtr, ytr)

        # Grouped Permutation Importance
        baseline_score = np.sqrt(mean_squared_error(yte, m.predict(Xte)))
        rng = np.random.default_rng(RANDOM_STATE)
        
        group_imp_fold = []
        for g_name in group_names:
            cols = FEATURE_GROUPS[g_name]
            valid_cols = [c for c in cols if c in X.columns]
            
            if not valid_cols:
                group_imp_fold.append(0.0)
                continue
                
            drops = []
            for _ in range(N_REPEATS_PERM):
                X_perm = Xte.copy()
                perm_idx = rng.permutation(len(X_perm))
                X_perm[valid_cols] = X_perm[valid_cols].values[perm_idx, :]
                score = np.sqrt(mean_squared_error(yte, m.predict(X_perm)))
                drops.append(score - baseline_score)
            group_imp_fold.append(np.mean(drops))
            
        fold_means.append(group_imp_fold)

    fold_means_matrix = np.array(fold_means)

    return pd.DataFrame({
        "feature": group_names,
        "importance_mean": np.mean(fold_means_matrix, axis=0),
        "importance_std": np.std(fold_means_matrix, axis=0)
    }).sort_values("importance_mean", ascending=False)


def compute_pls_cv_permutation_importance(X, y, groups, inner_groups_full, splitter):
    """Compute out-of-fold grouped permutation feature importance across external CV folds for PLS."""
    fold_means = []
    group_names = list(FEATURE_GROUPS.keys())

    for tr, te in splitter.split(X, y, groups):
        Xtr, ytr, gtr = X.iloc[tr], y.iloc[tr], inner_groups_full.iloc[tr]
        Xte, yte = X.iloc[te], y.iloc[te]

        max_c = min(MAX_PLS_COMP, Xtr.shape[1], len(Xtr) - 1)
        best_c = 1
        n_inner_groups = pd.Series(np.asarray(gtr)).nunique()

        if max_c > 1 and n_inner_groups >= 3:
            inner = GroupKFold(n_splits=min(5, n_inner_groups))
            rmses = {}
            for c in range(1, max_c + 1):
                errs = []
                for itr, ite in inner.split(Xtr, ytr, gtr):
                    m_inner = PLSRegression(n_components=c, scale=True)
                    m_inner.fit(Xtr.iloc[itr], ytr.iloc[itr])
                    errs.append(ytr.iloc[ite].values - m_inner.predict(Xtr.iloc[ite]).ravel())
                rmses[c] = np.sqrt(np.mean(np.concatenate(errs)**2))
            best_c = min(rmses, key=rmses.get)

        m = PLSRegression(n_components=best_c, scale=True)
        m.fit(Xtr, ytr)

        # Grouped Permutation Importance
        baseline_score = np.sqrt(mean_squared_error(yte, m.predict(Xte).ravel()))
        rng = np.random.default_rng(RANDOM_STATE)
        
        group_imp_fold = []
        for g_name in group_names:
            cols = FEATURE_GROUPS[g_name]
            valid_cols = [c for c in cols if c in X.columns]
            
            if not valid_cols:
                group_imp_fold.append(0.0)
                continue
                
            drops = []
            for _ in range(N_REPEATS_PERM):
                X_perm = Xte.copy()
                perm_idx = rng.permutation(len(X_perm))
                X_perm[valid_cols] = X_perm[valid_cols].values[perm_idx, :]
                score = np.sqrt(mean_squared_error(yte, m.predict(X_perm).ravel()))
                drops.append(score - baseline_score)
            group_imp_fold.append(np.mean(drops))
            
        fold_means.append(group_imp_fold)

    fold_means_matrix = np.array(fold_means)

    return pd.DataFrame({
        "feature": group_names,
        "importance_mean": np.mean(fold_means_matrix, axis=0),
        "importance_std": np.std(fold_means_matrix, axis=0)
    }).sort_values("importance_mean", ascending=False)


def resumen(nombre, cv_tree, n_feats):
    """Format model summary parameters into a structured DataFrame."""
    filas = [
        ("Modelo", nombre),
        ("R2_CV_arbol_OOF", cv_tree["R2"]),
        ("MAE_CV_arbol_OOF", cv_tree["MAE"]),
        ("RMSE_CV_arbol_OOF", cv_tree["RMSE"]),
        ("n_features", n_feats),
        ("rejilla_hiperparametros", str(RF_GRID)),
        ("seleccion_hiperparametros", "CV interna agrupada por árbol (RMSE mínimo)"),
        ("random_state", RANDOM_STATE)
    ]
    return pd.DataFrame(filas, columns=["Parametro", "Valor"])


# ============================
# 1) SLICE-LEVEL FEATURE EXTRACTION & MORPHOLOGY
# ============================

rows_olives = []
rows_pixels = []
procesados = 0
omitidos = 0

tiff_files = sorted(
    glob(os.path.join(IMAGES_DIR, "*.tif")) +
    glob(os.path.join(IMAGES_DIR, "*.tiff"))
)

for img_path in tiff_files:
    name = os.path.splitext(os.path.basename(img_path))[0]
    print("Procesando:", name)

    mask_flesh_path = os.path.join(MASKS_FLESH_DIR, name + ".png")
    mask_pit_path = os.path.join(MASKS_PIT_DIR, name + ".png")

    if not os.path.exists(mask_flesh_path) or not os.path.exists(mask_pit_path):
        print("    ⚠ Faltan máscaras, se omite.")
        omitidos += 1
        continue
    
    procesados += 1
    img = load_tiff32(img_path)

    if img.ndim == 3 and img.shape[2] == 3:
        img_gray = img[..., 0]
    else:
        img_gray = img

    H, W = img_gray.shape[:2]

    mask_flesh = load_and_resize_mask(mask_flesh_path, (H, W))
    mask_pit = load_and_resize_mask(mask_pit_path, (H, W))

    flesh_area = np.sum(mask_flesh)
    pit_area = np.sum(mask_pit)
    ratio = flesh_area / pit_area if pit_area > 0 else np.nan
    flesh_intensity = np.mean(img_gray[mask_flesh]) if flesh_area > 0 else np.nan

    arbol = extract_tree_name(name)

    rows_olives.append({
        "name": name,
        "arbol": arbol,
        "flesh_area": flesh_area,
        "pit_area": pit_area,
        "ratio_flesh_pit": ratio,
        "flesh_intensity": flesh_intensity
    })

    intensidades = img_gray[mask_flesh]

    corte = None
    if "corte_" in name:
        try:
            corte = int(name.split("corte_")[1].split("_")[0])
        except Exception:
            corte = None

    for i, val in enumerate(intensidades):
        rows_pixels.append({
            "arbol": arbol,
            "oliva": name,
            "corte": corte,
            "pixel_id": i,
            "intensidad": float(val)
        })


# ============================
# 2) SAVE BASIC METRICS & TREE AGGREGATION
# ============================

df_olives = pd.DataFrame(rows_olives)

print("\n--- RECUENTO DE DATOS (TIFF 32-bits) ---")
print(f"Cortes procesados: {procesados}")
print(f"Cortes omitidos (sin máscara): {omitidos}")
print(f"Cortes únicos: {df_olives['name'].nunique()}")
print(f"Unidades experimentales (árbol × etapa × campaña): {df_olives['arbol'].nunique()}")
print(f"Árboles físicos únicos (arbol_id): {add_groups(df_olives.copy())['arbol_id'].nunique()}")
print("----------------------------------------\n")

df_olives.to_csv(OUTPUT_OLIVES, index=False)

df_trees = df_olives.groupby("arbol").agg({
    "flesh_area": "mean",
    "pit_area": "mean",
    "ratio_flesh_pit": "mean",
    "flesh_intensity": "mean"
}).reset_index()
df_trees.to_csv(OUTPUT_TREES, index=False)

df_pixels = pd.DataFrame(rows_pixels)
df_pixels.to_csv(OUTPUT_PIXELS, index=False)

df_feat = df_pixels.groupby("arbol")["intensidad"].agg([
    "mean", "std", "median",
    lambda x: np.percentile(x, 5),
    lambda x: np.percentile(x, 25),
    lambda x: np.percentile(x, 75),
    lambda x: np.percentile(x, 95),
    "count"
]).reset_index()

df_feat.columns = [
    "arbol", "mean", "std", "median",
    "p05", "p25", "p75", "p95", "n_pixeles"
]
df_feat.to_csv(OUTPUT_FEATURES, index=False)


# ============================
# 3) MACHINE LEARNING MODELLING (FLESH INTENSITIES ONLY)
# ============================

if os.path.exists(GRASA_FILE):

    if GRASA_FILE.endswith(".xlsx"):
        df_grasa = pd.read_excel(GRASA_FILE)
    else:
        df_grasa = pd.read_csv(GRASA_FILE)

    df_grasa.columns = df_grasa.columns.str.strip().str.lower()
    col_arbol = [c for c in df_grasa.columns if "arbol" in c][0]
    col_grasa = [c for c in df_grasa.columns if "grasa" in c][0]

    df_grasa = df_grasa.rename(columns={
        col_arbol: "arbol",
        col_grasa: "grasa"
    })[["arbol", "grasa"]]

    # Merge intensity features with reference oil content
    print(f"Unidades experimentales antes del merge: {len(df_feat)}")
    df_ml_int = add_groups(df_feat.merge(df_grasa, on="arbol"))
    print(f"Unidades experimentales tras el merge: {len(df_ml_int)}\n")

    n_trees = df_ml_int["arbol_id"].nunique()
    gkf = GroupKFold(n_splits=min(5, n_trees))

    # ------------------------------------------------------------
    # MODEL: FLESH SIGNAL INTENSITIES (RF & PLS)
    # ------------------------------------------------------------
    print("\n=== MODELO: INTENSIDADES ===")

    cv_tree_int, folds_tree_int, oof_tree_int = run_cv(
        df_ml_int, FEATS, gkf, df_ml_int["arbol_id"], label="Int RF | GroupKFold árbol"
    )

    cv_tree_int_pls, folds_tree_int_pls, oof_tree_int_pls = run_cv(
        df_ml_int, FEATS, gkf, df_ml_int["arbol_id"],
        label="Int PLS | GroupKFold árbol", fit_predict=pls_fit_predict
    )

    # Feature correlation matrix & permutation importance
    df_corr_int = df_ml_int[FEATS].corr()

    df_importancia_int = compute_rf_cv_permutation_importance(
        df_ml_int[FEATS], df_ml_int["grasa"], df_ml_int["arbol_id"], df_ml_int["arbol_id"], gkf
    )

    df_importancia_int_pls = compute_pls_cv_permutation_importance(
        df_ml_int[FEATS], df_ml_int["grasa"], df_ml_int["arbol_id"], df_ml_int["arbol_id"], gkf
    )

    df_resultados_int = df_ml_int.copy()
    df_resultados_int["pred_oof_arbol_rf"] = oof_tree_int
    df_resultados_int["pred_oof_arbol_pls"] = oof_tree_int_pls

    # ------------------------------------------------------------
    # EXPORT RESULTS TO EXCEL
    # ------------------------------------------------------------
    with pd.ExcelWriter(EXCEL_OUTPUT, engine="openpyxl") as writer:

        # RF Intensities
        df_resultados_int.to_excel(writer, sheet_name="predicciones_intensidad", index=False)
        df_importancia_int.to_excel(writer, sheet_name="importancia_rf_intensidad", index=False)
        df_corr_int.to_excel(writer, sheet_name="correlacion_predictores", index=True)

        resumen("RF intensidades", cv_tree_int, len(FEATS)) \
            .to_excel(writer, sheet_name="modelo_rf_intensidad", index=False)
        folds_tree_int.to_excel(writer, sheet_name="folds_arbol_rf_int", index=False)

        # PLS Intensities
        df_importancia_int_pls.to_excel(writer, sheet_name="importancia_pls_intensidad", index=False)
        resumen("PLS intensidades", cv_tree_int_pls, len(FEATS)) \
            .to_excel(writer, sheet_name="modelo_pls_intensidad", index=False)
        folds_tree_int_pls.to_excel(writer, sheet_name="folds_arbol_pls_int", index=False)

        # Algorithm comparison summary table
        comp_rows = [
            {"Algoritmo": "RF", "Validacion": "GroupKFold árbol", **cv_tree_int},
            {"Algoritmo": "PLS", "Validacion": "GroupKFold árbol", **cv_tree_int_pls},
        ]

        pd.DataFrame(comp_rows).to_excel(writer, sheet_name="comparacion_RF_vs_PLS", index=False)

    print("\n==============================")
    print("   RESULTADOS RANDOM FOREST Y PLS (TIFF 32-bits)")
    print("==============================")
    print("Intensidades RF  | GroupKFold árbol -> R2: %.3f MAE: %.3f RMSE: %.3f" %
          (cv_tree_int["R2"], cv_tree_int["MAE"], cv_tree_int["RMSE"]))
    print("Intensidades PLS | GroupKFold árbol -> R2: %.3f MAE: %.3f RMSE: %.3f" %
          (cv_tree_int_pls["R2"], cv_tree_int_pls["MAE"], cv_tree_int_pls["RMSE"]))
    print("\n✔ Archivo Excel generado:", EXCEL_OUTPUT)

else:
    print("\n⚠ No se encontró el archivo de grasa:", GRASA_FILE)
    print("   → Se omite el modelo ML.")
