# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 11:53:10 2026

@author: victo
"""

import os
import re
import numpy as np
import pandas as pd
import imageio.v3 as iio
from glob import glob
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ============================================================
# CONFIGURACIÓN
# ============================================================

IMAGES_DIR = r"C:\Users\victo\OneDrive - Universidad Complutense de Madrid (UCM)\DOCTORADO\ARTÍCULOS PUBLICADOS POR MI\ARTÍCULO OLIVAS MRI BRAGANÇA\DATASETS USADOS\MODELOS MACHINE LEARNING\JPG\DATASET JPG MÁSCARAS Y CÁLCULOS"
MASKS_FLESH_DIR = r"C:\Users\victo\OneDrive - Universidad Complutense de Madrid (UCM)\DOCTORADO\ARTÍCULOS PUBLICADOS POR MI\ARTÍCULO OLIVAS MRI BRAGANÇA\DATASETS USADOS\MODELOS MACHINE LEARNING\JPG\RESULTADOS FINALES MÁSCARAS\masks_output (1)\flesh"
MASKS_PIT_DIR   = r"C:\Users\victo\OneDrive - Universidad Complutense de Madrid (UCM)\DOCTORADO\ARTÍCULOS PUBLICADOS POR MI\ARTÍCULO OLIVAS MRI BRAGANÇA\DATASETS USADOS\MODELOS MACHINE LEARNING\JPG\RESULTADOS FINALES MÁSCARAS\masks_output (1)\pit"

OUTPUT_OLIVES   = "resultados_jpg_por_oliva.csv"
OUTPUT_TREES    = "resultados_jpg_por_arbol.csv"
OUTPUT_PIXELS   = "intensidades_pixeles_pulpa_jpg.csv"
OUTPUT_FEATURES = "features_por_arbol_jpg.csv"

GRASA_FILE      = r"C:\Users\victo\OneDrive - Universidad Complutense de Madrid (UCM)\DOCTORADO\ARTÍCULOS PUBLICADOS POR MI\ARTÍCULO OLIVAS MRI BRAGANÇA\DATASETS USADOS\MODELOS MACHINE LEARNING\grasa_arbol.xlsx"

N_ESTIMATORS = 500
RANDOM_STATE = 42
MAX_PLS_COMP = 6   # nº máximo de componentes PLS a explorar (se elige por CV interna)
FEATS = ["mean", "std", "median", "p05", "p25", "p75", "p95", "n_pixeles"]


# ============================================================
# FUNCIONES
# ============================================================

def load_jpg(path):
    img = iio.imread(path)
    return img.astype(np.float32)

def load_and_resize_mask(mask_path, target_shape):
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((target_shape[1], target_shape[0]), Image.NEAREST)
    return np.array(mask) > 0

def extract_tree_name(name):
    parts = name.split("_")
    if "arbol" not in parts:
        return "desconocido"

    idx = parts.index("arbol")
    arbol_num = parts[idx + 1]
    mes = parts[idx + 2]

    if parts[idx + 3].isdigit() and len(parts[idx + 3]) == 1:
        sufijo = "_" + parts[idx + 3]
        año = parts[idx + 4]
    else:
        sufijo = ""
        año = parts[idx + 3]

    return f"arbol_{arbol_num}_{mes}{sufijo}_{año}"

def add_groups(df):
    """
    A partir del identificador 'arbol' (arbol_{num}_{mes}[_sufijo]_{año})
    crea:
      - arbol_id: número de árbol físico (grupo para GroupKFold)
      - cosecha:  año (grupo para Leave-One-Harvest-Out)
    """
    parts = df["arbol"].str.split("_")
    df = df.copy()
    df["arbol_id"] = parts.str[1]
    df["cosecha"] = parts.str[-1]
    return df

def cv_metrics(y, pred):
    return {
        "R2": r2_score(y, pred),
        "MAE": mean_absolute_error(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
    }

def rf_fit_predict(Xtr, ytr, gtr, Xte):
    m = RandomForestRegressor(n_estimators=N_ESTIMATORS,
                              max_depth=None,
                              random_state=RANDOM_STATE)
    m.fit(Xtr, ytr)
    return m.predict(Xte), ""

def pls_fit_predict(Xtr, ytr, gtr, Xte):
    """
    PLS-R lineal (variables autoescaladas). El nº de componentes se elige
    DENTRO del conjunto de entrenamiento con una CV interna agrupada por
    árbol físico (gtr), para que el conjunto de test no influya en la elección.
    """
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
            rmses[c] = np.sqrt(np.mean(np.concatenate(errs) ** 2))
        best_c = min(rmses, key=rmses.get)

    m = PLSRegression(n_components=best_c, scale=True)
    m.fit(Xtr, ytr)
    return m.predict(Xte).ravel(), f"n_comp={best_c}"

def resumen_pls(nombre, cv_tree, cv_loho, n_feats):
    filas = [("Modelo", nombre)]
    filas += [(f"{k}_CV_arbol_OOF", v) for k, v in cv_tree.items()]
    filas += [(f"{k}_LOHO_OOF", v) for k, v in cv_loho.items()]
    filas += [("n_features", n_feats),
              ("max_componentes_explorados", min(MAX_PLS_COMP, n_feats)),
              ("seleccion_componentes", "CV interna agrupada por árbol (RMSE mínimo)"),
              ("escalado", "autoescalado (scale=True)")]
    return pd.DataFrame(filas, columns=["Parametro", "Valor"])

def run_cv(df, feats, splitter, groups, label="", fit_predict=rf_fit_predict, inner_groups=None):
    """
    Validación cruzada agrupada.
    Devuelve: métricas sobre predicciones out-of-fold agrupadas,
              DataFrame con métricas por fold, y predicciones out-of-fold.
    inner_groups: grupos (árbol físico) para la CV interna de selección de
                  hiperparámetros (solo usada por PLS).
    """
    X, y = df[feats], df["grasa"]
    if inner_groups is None:
        inner_groups = df["arbol_id"]
    oof = np.full(len(df), np.nan)
    folds = []

    for k, (tr, te) in enumerate(splitter.split(X, y, groups)):
        pred, info = fit_predict(X.iloc[tr], y.iloc[tr], inner_groups.iloc[tr], X.iloc[te])
        oof[te] = pred

        met = cv_metrics(y.iloc[te], oof[te])
        met["info"] = info
        met["fold"] = k + 1
        met["grupos_test"] = ",".join(sorted(set(groups.iloc[te].astype(str))))
        met["n_train"] = len(tr)
        met["n_test"] = len(te)
        folds.append(met)
        print(f"[{label}] fold {k+1} | test: {met['grupos_test'][:60]} | "
              f"R2: {met['R2']:.3f} | MAE: {met['MAE']:.3f} | RMSE: {met['RMSE']:.3f}")

    overall = cv_metrics(y, oof)
    print(f"[{label}] OOF agrupado -> R2: {overall['R2']:.3f} | "
          f"MAE: {overall['MAE']:.3f} | RMSE: {overall['RMSE']:.3f}\n")
    return overall, pd.DataFrame(folds), oof

def resumen(nombre, cv_tree, cv_loho, n_feats):
    filas = [("Modelo", nombre)]
    filas += [(f"{k}_CV_arbol_OOF", v) for k, v in cv_tree.items()]
    filas += [(f"{k}_LOHO_OOF", v) for k, v in cv_loho.items()]
    filas += [("n_features", n_feats),
              ("n_estimators", N_ESTIMATORS),
              ("max_depth", "None"),
              ("random_state", RANDOM_STATE)]
    return pd.DataFrame(filas, columns=["Parametro", "Valor"])


# ============================================================
# 1) PROCESAMIENTO POR OLIVA + EXTRACCIÓN DE PIXELES
# ============================================================

rows_olives = []
rows_pixels = []

jpg_files = sorted(
    glob(os.path.join(IMAGES_DIR, "*.jpg")) +
    glob(os.path.join(IMAGES_DIR, "*.jpeg"))
)

for img_path in jpg_files:

    name = os.path.splitext(os.path.basename(img_path))[0]
    print("Procesando:", name)

    mask_flesh_path = os.path.join(MASKS_FLESH_DIR, name + ".png")
    mask_pit_path   = os.path.join(MASKS_PIT_DIR, name + ".png")

    if not os.path.exists(mask_flesh_path) or not os.path.exists(mask_pit_path):
        print("   ⚠ Faltan máscaras, se omite.")
        continue

    img = load_jpg(img_path)

    if img.ndim == 3 and img.shape[2] == 3:
        img_gray = img[..., 0]
    else:
        img_gray = img

    H, W = img_gray.shape[:2]

    mask_flesh = load_and_resize_mask(mask_flesh_path, (H, W))
    mask_pit   = load_and_resize_mask(mask_pit_path, (H, W))

    flesh_area = np.sum(mask_flesh)
    pit_area   = np.sum(mask_pit)
    ratio      = flesh_area / pit_area if pit_area > 0 else np.nan
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
        except:
            corte = None

    for i, val in enumerate(intensidades):
        rows_pixels.append({
            "arbol": arbol,
            "oliva": name,
            "corte": corte,
            "pixel_id": i,
            "intensidad": float(val)
        })


# ============================================================
# 2) GUARDAR RESULTADOS POR OLIVA
# ============================================================

df_olives = pd.DataFrame(rows_olives)
df_olives.to_csv(OUTPUT_OLIVES, index=False)


# ============================================================
# 3) PROMEDIOS POR ÁRBOL
# ============================================================

df_trees = df_olives.groupby("arbol").agg({
    "flesh_area": "mean",
    "pit_area": "mean",
    "ratio_flesh_pit": "mean",
    "flesh_intensity": "mean"
}).reset_index()

df_trees.to_csv(OUTPUT_TREES, index=False)


# ============================================================
# 4) GUARDAR INTENSIDADES POR PIXEL
# ============================================================

df_pixels = pd.DataFrame(rows_pixels)
df_pixels.to_csv(OUTPUT_PIXELS, index=False)


# ============================================================
# 5) GENERAR FEATURES POR ÁRBOL (INTENSIDADES)
# ============================================================

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


# ============================================================
# 6) RANDOM FOREST (2 MODELOS) + CV AGRUPADA + LOHO + EXCEL
# ============================================================

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

    # ------------------------------------------------------------
    # Datasets (grupos: arbol_id = árbol físico; cosecha = año)
    # ------------------------------------------------------------
    df_ml_int = add_groups(df_feat.merge(df_grasa, on="arbol"))
    df_ratio  = add_groups(df_trees[["arbol", "ratio_flesh_pit"]].merge(df_grasa, on="arbol"))

    print("\nComprobación de grupos (primeras filas):")
    print(df_ml_int[["arbol", "arbol_id", "cosecha"]].head(10))
    print("Árboles físicos:", df_ml_int["arbol_id"].nunique(),
          "| Cosechas:", sorted(df_ml_int["cosecha"].unique()),
          "| n muestras:", len(df_ml_int))

    n_trees = df_ml_int["arbol_id"].nunique()
    gkf = GroupKFold(n_splits=min(5, n_trees))
    loho = LeaveOneGroupOut()

    # ------------------------------------------------------------
    # MODELO 1: INTENSIDADES
    # ------------------------------------------------------------
    print("\n=== MODELO 1: INTENSIDADES ===")
    cv_tree_int, folds_tree_int, oof_tree_int = run_cv(
        df_ml_int, FEATS, gkf, df_ml_int["arbol_id"], label="Int | GroupKFold árbol")
    cv_loho_int, folds_loho_int, oof_loho_int = run_cv(
        df_ml_int, FEATS, loho, df_ml_int["cosecha"], label="Int | LOHO")

    # PLS-R lineal (mismas particiones que RF)
    cv_tree_int_pls, folds_tree_int_pls, oof_tree_int_pls = run_cv(
        df_ml_int, FEATS, gkf, df_ml_int["arbol_id"],
        label="Int PLS | GroupKFold árbol", fit_predict=pls_fit_predict)
    cv_loho_int_pls, folds_loho_int_pls, oof_loho_int_pls = run_cv(
        df_ml_int, FEATS, loho, df_ml_int["cosecha"],
        label="Int PLS | LOHO", fit_predict=pls_fit_predict)

    # Modelo final (solo para importancias y uso posterior; sus métricas
    # in-sample NO se reportan)
    model_int = RandomForestRegressor(n_estimators=N_ESTIMATORS,
                                      max_depth=None,
                                      random_state=RANDOM_STATE)
    model_int.fit(df_ml_int[FEATS], df_ml_int["grasa"])

    df_resultados_int = df_ml_int.copy()
    df_resultados_int["pred_oof_arbol"] = oof_tree_int
    df_resultados_int["pred_oof_loho"] = oof_loho_int
    df_resultados_int["error_oof_arbol"] = df_resultados_int["pred_oof_arbol"] - df_resultados_int["grasa"]
    df_resultados_int["error_oof_loho"] = df_resultados_int["pred_oof_loho"] - df_resultados_int["grasa"]
    df_resultados_int["pred_oof_arbol_pls"] = oof_tree_int_pls
    df_resultados_int["pred_oof_loho_pls"] = oof_loho_int_pls
    df_resultados_int["error_oof_arbol_pls"] = df_resultados_int["pred_oof_arbol_pls"] - df_resultados_int["grasa"]
    df_resultados_int["error_oof_loho_pls"] = df_resultados_int["pred_oof_loho_pls"] - df_resultados_int["grasa"]

    df_importancia_int = pd.DataFrame({
        "feature": FEATS,
        "importance": model_int.feature_importances_
    }).sort_values("importance", ascending=False)

    # ------------------------------------------------------------
    # MODELO 2: SOLO RATIO FLESH/PIT
    # ------------------------------------------------------------
    print("\n=== MODELO 2: RATIO FLESH/PIT ===")
    FEATS_RATIO = ["ratio_flesh_pit"]

    cv_tree_ratio, folds_tree_ratio, oof_tree_ratio = run_cv(
        df_ratio, FEATS_RATIO, gkf, df_ratio["arbol_id"], label="Ratio | GroupKFold árbol")
    cv_loho_ratio, folds_loho_ratio, oof_loho_ratio = run_cv(
        df_ratio, FEATS_RATIO, loho, df_ratio["cosecha"], label="Ratio | LOHO")

    # PLS-R con una sola variable = regresión lineal simple (1 componente)
    cv_tree_ratio_pls, folds_tree_ratio_pls, oof_tree_ratio_pls = run_cv(
        df_ratio, FEATS_RATIO, gkf, df_ratio["arbol_id"],
        label="Ratio PLS | GroupKFold árbol", fit_predict=pls_fit_predict)
    cv_loho_ratio_pls, folds_loho_ratio_pls, oof_loho_ratio_pls = run_cv(
        df_ratio, FEATS_RATIO, loho, df_ratio["cosecha"],
        label="Ratio PLS | LOHO", fit_predict=pls_fit_predict)

    model_ratio = RandomForestRegressor(n_estimators=N_ESTIMATORS,
                                        max_depth=None,
                                        random_state=RANDOM_STATE)
    model_ratio.fit(df_ratio[FEATS_RATIO], df_ratio["grasa"])

    df_resultados_ratio = df_ratio.copy()
    df_resultados_ratio["pred_oof_arbol"] = oof_tree_ratio
    df_resultados_ratio["pred_oof_loho"] = oof_loho_ratio
    df_resultados_ratio["error_oof_arbol"] = df_resultados_ratio["pred_oof_arbol"] - df_resultados_ratio["grasa"]
    df_resultados_ratio["error_oof_loho"] = df_resultados_ratio["pred_oof_loho"] - df_resultados_ratio["grasa"]
    df_resultados_ratio["pred_oof_arbol_pls"] = oof_tree_ratio_pls
    df_resultados_ratio["pred_oof_loho_pls"] = oof_loho_ratio_pls
    df_resultados_ratio["error_oof_arbol_pls"] = df_resultados_ratio["pred_oof_arbol_pls"] - df_resultados_ratio["grasa"]
    df_resultados_ratio["error_oof_loho_pls"] = df_resultados_ratio["pred_oof_loho_pls"] - df_resultados_ratio["grasa"]

    df_importancia_ratio = pd.DataFrame({
        "feature": FEATS_RATIO,
        "importance": model_ratio.feature_importances_
    })

    # ------------------------------------------------------------
    # EXPORTAR TODO EN UN SOLO EXCEL
    # ------------------------------------------------------------
    excel_output = "resultados_modelo_randomforest_jpg_dos_modelos.xlsx"

    with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:

        # Modelo intensidades
        df_resultados_int.to_excel(writer, sheet_name="predicciones_intensidad", index=False)
        df_importancia_int.to_excel(writer, sheet_name="importancia_intensidad", index=False)
        resumen("RandomForest (intensidades)", cv_tree_int, cv_loho_int, len(FEATS)) \
            .to_excel(writer, sheet_name="modelo_intensidad", index=False)
        folds_tree_int.to_excel(writer, sheet_name="folds_arbol_int", index=False)
        folds_loho_int.to_excel(writer, sheet_name="folds_loho_int", index=False)

        # Modelo ratio
        df_resultados_ratio.to_excel(writer, sheet_name="predicciones_ratio", index=False)
        df_importancia_ratio.to_excel(writer, sheet_name="importancia_ratio", index=False)
        resumen("RandomForest (ratio flesh/pit)", cv_tree_ratio, cv_loho_ratio, len(FEATS_RATIO)) \
            .to_excel(writer, sheet_name="modelo_ratio", index=False)
        folds_tree_ratio.to_excel(writer, sheet_name="folds_arbol_ratio", index=False)
        folds_loho_ratio.to_excel(writer, sheet_name="folds_loho_ratio", index=False)

        # PLS-R lineal
        resumen_pls("PLS-R (intensidades)", cv_tree_int_pls, cv_loho_int_pls, len(FEATS)) \
            .to_excel(writer, sheet_name="modelo_pls_intensidad", index=False)
        folds_tree_int_pls.to_excel(writer, sheet_name="folds_arbol_pls_int", index=False)
        folds_loho_int_pls.to_excel(writer, sheet_name="folds_loho_pls_int", index=False)
        resumen_pls("PLS-R (ratio flesh/pit)", cv_tree_ratio_pls, cv_loho_ratio_pls, len(FEATS_RATIO)) \
            .to_excel(writer, sheet_name="modelo_pls_ratio", index=False)
        folds_tree_ratio_pls.to_excel(writer, sheet_name="folds_arbol_pls_ratio", index=False)
        folds_loho_ratio_pls.to_excel(writer, sheet_name="folds_loho_pls_ratio", index=False)

        # Comparación RF vs PLS-R
        comp_rows = []
        for modelo, algoritmo, val, met in [
            ("Intensidades", "RandomForest", "GroupKFold árbol", cv_tree_int),
            ("Intensidades", "RandomForest", "LOHO", cv_loho_int),
            ("Intensidades", "PLS-R", "GroupKFold árbol", cv_tree_int_pls),
            ("Intensidades", "PLS-R", "LOHO", cv_loho_int_pls),
            ("Ratio", "RandomForest", "GroupKFold árbol", cv_tree_ratio),
            ("Ratio", "RandomForest", "LOHO", cv_loho_ratio),
            ("Ratio", "PLS-R", "GroupKFold árbol", cv_tree_ratio_pls),
            ("Ratio", "PLS-R", "LOHO", cv_loho_ratio_pls),
        ]:
            comp_rows.append({"Variables": modelo, "Algoritmo": algoritmo,
                              "Validacion": val, **met})
        df_comp = pd.DataFrame(comp_rows)
        df_comp.to_excel(writer, sheet_name="comparacion_RF_vs_PLS", index=False)

    print("\n==============================")
    print("   RESULTADOS RANDOM FOREST (JPG)")
    print("==============================")
    print("Intensidades | GroupKFold árbol -> R2: %.3f MAE: %.3f RMSE: %.3f" %
          (cv_tree_int["R2"], cv_tree_int["MAE"], cv_tree_int["RMSE"]))
    print("Intensidades | LOHO            -> R2: %.3f MAE: %.3f RMSE: %.3f" %
          (cv_loho_int["R2"], cv_loho_int["MAE"], cv_loho_int["RMSE"]))
    print("Ratio        | GroupKFold árbol -> R2: %.3f MAE: %.3f RMSE: %.3f" %
          (cv_tree_ratio["R2"], cv_tree_ratio["MAE"], cv_tree_ratio["RMSE"]))
    print("Ratio        | LOHO            -> R2: %.3f MAE: %.3f RMSE: %.3f" %
          (cv_loho_ratio["R2"], cv_loho_ratio["MAE"], cv_loho_ratio["RMSE"]))
    print("\nComparación RF vs PLS-R (out-of-fold agrupado):")
    print(df_comp.round(3).to_string(index=False))
    print("✔ Archivo Excel generado:", excel_output)

else:
    print("\n⚠ No se encontró el archivo de grasa:", GRASA_FILE)
    print("   → Se omite el modelo ML.")
