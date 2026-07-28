# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:54:32 2026

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
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ============================================================
# CONFIGURACIÓN
# ============================================================

IMAGES_DIR = r"C:\Users\victo\OneDrive - Universidad Complutense de Madrid (UCM)\DOCTORADO\ESTANCIA BRAGANZA 2026\PROCESADO\OLIVAS\COSAS BUENAS COMPLETAS\COSAS CON PNG\PNG USADOS PARA CÁLCULOS"
MASKS_FLESH_DIR = r"C:\Users\victo\OneDrive - Universidad Complutense de Madrid (UCM)\DOCTORADO\ESTANCIA BRAGANZA 2026\PROCESADO\OLIVAS\COSAS BUENAS COMPLETAS\COSAS CON PNG\masks_output (1)\flesh"
MASKS_PIT_DIR   = r"C:\Users\victo\OneDrive - Universidad Complutense de Madrid (UCM)\DOCTORADO\ESTANCIA BRAGANZA 2026\PROCESADO\OLIVAS\COSAS BUENAS COMPLETAS\COSAS CON PNG\masks_output (1)\pit"

OUTPUT_OLIVES   = "resultados_png_por_oliva.csv"
OUTPUT_TREES    = "resultados_png_por_arbol.csv"
OUTPUT_PIXELS   = "intensidades_pixeles_pulpa_png.csv"
OUTPUT_FEATURES = "features_por_arbol_png.csv"

GRASA_FILE      = r"C:\Users\victo\OneDrive - Universidad Complutense de Madrid (UCM)\DOCTORADO\ESTANCIA BRAGANZA 2026\PROCESADO\OLIVAS\COSAS BUENAS COMPLETAS\grasa_arbol.xlsx"


# ============================================================
# FUNCIONES
# ============================================================

def load_png(path):
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


# ============================================================
# 1) PROCESADO POR OLIVA + EXTRACCIÓN DE PIXELES
# ============================================================

rows_olives = []
rows_pixels = []

png_files = sorted(glob(os.path.join(IMAGES_DIR, "*.png")))

for img_path in png_files:

    name = os.path.basename(img_path).replace(".png", "")
    print("Procesando:", name)

    mask_flesh_path = os.path.join(MASKS_FLESH_DIR, name + ".png")
    mask_pit_path   = os.path.join(MASKS_PIT_DIR, name + ".png")

    if not os.path.exists(mask_flesh_path) or not os.path.exists(mask_pit_path):
        print("   ⚠ Faltan máscaras, se omite.")
        continue

    img = load_png(img_path)

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
# 6) RANDOM FOREST + CV + EXCEL
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
    })

    # ============================================================
    # MODELO 1: INTENSIDADES
    # ============================================================

    df_ml_int = df_feat.merge(df_grasa, on="arbol")

    X_int = df_ml_int.drop(columns=["arbol", "grasa"])
    y_int = df_ml_int["grasa"]

    model_cv_int = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        random_state=42
    )

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    r2_int = cross_val_score(model_cv_int, X_int, y_int, cv=kfold, scoring="r2")
    mae_int = -cross_val_score(model_cv_int, X_int, y_int, cv=kfold, scoring="neg_mean_absolute_error")
    rmse_int = np.sqrt(-cross_val_score(model_cv_int, X_int, y_int, cv=kfold, scoring="neg_mean_squared_error"))

    X_train_int, X_test_int, y_train_int, y_test_int = train_test_split(
        X_int, y_int, test_size=0.25, random_state=42
    )

    model_int = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        random_state=42
    )

    model_int.fit(X_train_int, y_train_int)
    pred_int = model_int.predict(X_test_int)

    R2_int = r2_score(y_test_int, pred_int)
    MAE_int = mean_absolute_error(y_test_int, pred_int)
    RMSE_int = np.sqrt(mean_squared_error(y_test_int, pred_int))

    df_resultados_int = df_ml_int.copy()
    df_resultados_int["prediccion"] = model_int.predict(X_int)
    df_resultados_int["error"] = df_resultados_int["prediccion"] - df_resultados_int["grasa"]

    df_importancia_int = pd.DataFrame({
        "feature": X_int.columns,
        "importance": model_int.feature_importances_
    }).sort_values("importance", ascending=False)

   

    # ============================================================
    # EXPORTAR TODO EN UN SOLO EXCEL
    # ============================================================

    excel_output = "resultados_modelo_randomforest_png_dos_modelos.xlsx"

    with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:

        # Modelo intensidades
        df_resultados_int.to_excel(writer, sheet_name="predicciones_intensidad", index=False)
        df_importancia_int.to_excel(writer, sheet_name="importancia_intensidad", index=False)

        resumen_int = pd.DataFrame({
            "Parametro": [
                "Modelo",
                "R2_test", "MAE_test", "RMSE_test",
                "R2_CV_mean", "R2_CV_std",
                "MAE_CV_mean", "MAE_CV_std",
                "RMSE_CV_mean", "RMSE_CV_std",
                "n_estimators", "max_depth"
            ],
            "Valor": [
                "RandomForest (intensidades)",
                R2_int, MAE_int, RMSE_int,
                r2_int.mean(), r2_int.std(),
                mae_int.mean(), mae_int.std(),
                rmse_int.mean(), rmse_int.std(),
                500, "None"
            ]
        })
        resumen_int.to_excel(writer, sheet_name="modelo_intensidad", index=False)

        

    print("\n==============================")
    print("   RESULTADOS RANDOM FOREST (PNG)")
    print("==============================")
    print("Modelo intensidades -> R2:", R2_int, "MAE:", MAE_int, "RMSE:", RMSE_int)
    print("✔ Archivo Excel generado:", excel_output)

else:
    print("\n⚠ No se encontró el archivo de grasa:", GRASA_FILE)
    print("   → Se omite el modelo ML.")

