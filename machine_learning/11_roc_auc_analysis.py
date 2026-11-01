import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score
)

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT = os.getcwd()

GENOMICS_FILE = os.path.join(ROOT, "Genomics_Subtype_Aligned.csv")
EPIGENOMICS_FILE = os.path.join(ROOT, "Epigenomics_Subtype_Aligned.csv")
LABELS_FILE = os.path.join(ROOT, "Clinical_Subtype_Labels.csv")

TABLE_DIR = os.path.join(ROOT, "results", "tables")
FIGURE_DIR = os.path.join(ROOT, "results", "figures")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# --------------------------------------------------
# 2. LOAD AND ALIGN DATA
# --------------------------------------------------

print("Loading datasets...")

genomics = pd.read_csv(GENOMICS_FILE)
epigenomics = pd.read_csv(EPIGENOMICS_FILE)
labels_df = pd.read_csv(LABELS_FILE)

for df in [genomics, epigenomics, labels_df]:
    df["Patient_ID"] = (
        df["Patient_ID"]
        .astype(str)
        .str.replace(".", "-", regex=False)
        .str[:12]
    )

common_ids = sorted(
    set(genomics["Patient_ID"])
    & set(epigenomics["Patient_ID"])
    & set(labels_df["Patient_ID"])
)

genomics = (
    genomics.drop_duplicates("Patient_ID")
    .set_index("Patient_ID")
    .loc[common_ids]
)

epigenomics = (
    epigenomics.drop_duplicates("Patient_ID")
    .set_index("Patient_ID")
    .loc[common_ids]
)

labels_df = (
    labels_df.drop_duplicates("Patient_ID")
    .set_index("Patient_ID")
    .loc[common_ids]
)

genomics_features = genomics.drop(columns=["Sample ID"], errors="ignore")
epigenomics_features = epigenomics.drop(columns=["Sample ID"], errors="ignore")

X = pd.concat(
    [
        genomics_features.add_prefix("GEN_"),
        epigenomics_features.add_prefix("EPI_")
    ],
    axis=1
)

y = labels_df["mRNA Subtype Clusters"]

print("Dataset shape:", X.shape)
print("Samples:", len(y))
print("\nSubtype distribution:")
print(y.value_counts())

# --------------------------------------------------
# 3. DEFINE INTEGRATED MODEL
# --------------------------------------------------

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("variance_filter", VarianceThreshold()),
    ("feature_selection", SelectKBest(score_func=f_classif, k=100)),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=42
    ))
])

# --------------------------------------------------
# 4. GENERATE OUT-OF-FOLD PROBABILITIES
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nGenerating out-of-fold probabilities...")

y_proba = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
    method="predict_proba",
    n_jobs=1
)

# --------------------------------------------------
# 5. CALCULATE MULTICLASS ROC-AUC
# --------------------------------------------------

class_names = np.sort(y.unique())

y_binary = label_binarize(
    y,
    classes=class_names
)

auc_scores = {}

print("\nSUBTYPE-WISE ROC-AUC")
print("--------------------")

for i, class_name in enumerate(class_names):
    score = roc_auc_score(
        y_binary[:, i],
        y_proba[:, i]
    )

    auc_scores[class_name] = score

    print(f"{class_name}: {score:.4f}")

macro_auc = roc_auc_score(
    y,
    y_proba,
    multi_class="ovr",
    average="macro",
    labels=class_names
)

micro_auc = roc_auc_score(
    y_binary,
    y_proba,
    average="micro"
)

print("\nOVERALL ROC-AUC")
print("---------------")
print(f"Macro-average AUC: {macro_auc:.4f}")
print(f"Micro-average AUC: {micro_auc:.4f}")

# --------------------------------------------------
# 6. GENERATE ROC CURVES
# --------------------------------------------------

plt.figure(figsize=(10, 8))

for i, class_name in enumerate(class_names):
    fpr, tpr, _ = roc_curve(
        y_binary[:, i],
        y_proba[:, i]
    )

    class_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{class_name} (AUC = {class_auc:.3f})"
    )

# Micro-average ROC
micro_fpr, micro_tpr, _ = roc_curve(
    y_binary.ravel(),
    y_proba.ravel()
)

plt.plot(
    micro_fpr,
    micro_tpr,
    linestyle="--",
    linewidth=2,
    label=f"Micro-average (AUC = {micro_auc:.3f})"
)

# No-skill reference line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle=":",
    linewidth=1.5,
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "Multiclass ROC Curves - Integrated Logistic Regression",
    fontsize=13
)

plt.legend(loc="lower right")
plt.grid(alpha=0.25)
plt.tight_layout()

figure_path = os.path.join(
    FIGURE_DIR,
    "Integrated_LR_ROC_Curve.png"
)

plt.savefig(figure_path, dpi=300)
plt.close()

# --------------------------------------------------
# 7. SAVE AUC RESULTS
# --------------------------------------------------

auc_results = pd.DataFrame({
    "Subtype": list(auc_scores.keys()),
    "ROC_AUC": list(auc_scores.values())
})

summary = pd.DataFrame({
    "Subtype": ["Macro-average", "Micro-average"],
    "ROC_AUC": [macro_auc, micro_auc]
})

auc_results = pd.concat(
    [auc_results, summary],
    ignore_index=True
)

table_path = os.path.join(
    TABLE_DIR,
    "Integrated_LR_ROC_AUC_Results.csv"
)

auc_results.to_csv(table_path, index=False)

# --------------------------------------------------
# 8. COMPLETION
# --------------------------------------------------

print("\nROC-AUC analysis completed.")

print("\nSaved files:")
print(figure_path)
print(table_path)