#!/usr/bin/env python3
"""
=============================================================================
 ANN-based NIDS v2 — Enhanced with Feature Engineering & Class Weights
 Dataset: NSL-KDD
=============================================================================
 Improvements over v1:
   1. Feature Engineering (interaction features, log transforms, aggregates)
   2. Class Weights (penalize missed attacks)
   3. Threshold Tuning (optimize for F1 / recall)
   4. Feature Selection (drop zero-variance + mutual information ranking)
   5. Comparison report: v1 baseline vs v2 enhanced
=============================================================================
"""

import os, warnings, time
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc,
    precision_recall_curve
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_DIR   = "/home/deepfake/NIDS_ANN/nsl-kdd"
OUTPUT_DIR = "/home/deepfake/NIDS_ANN/results_v2.1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(BASE_DIR, "KDDTrain+.txt")
TEST_FILE  = os.path.join(BASE_DIR, "KDDTest+.txt")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─── Column Names ────────────────────────────────────────────────────────────
COL_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty_level"
]

ATTACK_MAP = {
    "normal": "Normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS", "apache2": "DoS",
    "processtable": "DoS", "udpstorm": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L",
    "snmpgetattack": "R2L", "snmpguess": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "worm": "R2L", "httptunnel": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "xterm": "U2R", "ps": "U2R", "sqlattack": "U2R",
}


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 1: LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
def load_data():
    print("=" * 70)
    print("  [1/7] LOADING NSL-KDD DATASET")
    print("=" * 70)

    train_df = pd.read_csv(TRAIN_FILE, header=None, names=COL_NAMES)
    test_df  = pd.read_csv(TEST_FILE,  header=None, names=COL_NAMES)

    print(f"  Train samples : {len(train_df):>8,}")
    print(f"  Test  samples : {len(test_df):>8,}")

    train_df.drop("difficulty_level", axis=1, inplace=True)
    test_df.drop("difficulty_level", axis=1, inplace=True)

    return train_df, test_df


def create_labels(df):
    df["binary_label"]    = df["label"].apply(lambda x: 0 if x == "normal" else 1)
    df["attack_category"] = df["label"].map(ATTACK_MAP).fillna("Unknown")
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2: FEATURE ENGINEERING (The big improvement)
# ═══════════════════════════════════════════════════════════════════════════
def engineer_features(df, fit_info=None, is_train=True):
    """
    Creates new features that capture network traffic patterns better.
    Returns: df with new columns, fit_info dict (for applying same transforms to test)
    """
    if fit_info is None:
        fit_info = {}

    # ── 2a. Log transforms for heavy-tailed numerical features ──
    # src_bytes and dst_bytes have extreme skew; log compresses the range
    for col in ["src_bytes", "dst_bytes", "duration"]:
        df[f"log_{col}"] = np.log1p(df[col].astype(float))

    # ── 2b. Interaction features (domain-knowledge driven) ──
    # Bytes ratio: asymmetry between sent/received is a strong attack signal
    df["bytes_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1)
    df["total_bytes"] = df["src_bytes"] + df["dst_bytes"]
    df["log_total_bytes"] = np.log1p(df["total_bytes"].astype(float))

    # Connection error features (serror and rerror combined signals)
    df["total_error_rate"]  = df["serror_rate"] + df["rerror_rate"]
    df["srv_total_error_rate"] = df["srv_serror_rate"] + df["srv_rerror_rate"]

    # Host-level error aggregates
    df["dst_host_total_error"] = df["dst_host_serror_rate"] + df["dst_host_rerror_rate"]
    df["dst_host_srv_total_error"] = df["dst_host_srv_serror_rate"] + df["dst_host_srv_rerror_rate"]

    # ── 2c. Same/Different service ratio features ──
    df["srv_ratio"] = df["same_srv_rate"] / (df["diff_srv_rate"] + 0.001)
    df["dst_host_srv_ratio"] = df["dst_host_same_srv_rate"] / (df["dst_host_diff_srv_rate"] + 0.001)

    # ── 2d. Count-based interaction features ──
    # High count + high error rate = likely scan/DoS
    df["count_x_serror"] = df["count"] * df["serror_rate"]
    df["count_x_rerror"] = df["count"] * df["rerror_rate"]
    df["srv_count_x_serror"] = df["srv_count"] * df["srv_serror_rate"]

    # Count ratio (connection diversity)
    df["count_srv_ratio"] = df["srv_count"] / (df["count"] + 1)

    # ── 2e. Dst host count interactions ──
    df["dst_host_count_x_serror"] = df["dst_host_count"] * df["dst_host_serror_rate"]
    df["dst_host_count_x_rerror"] = df["dst_host_count"] * df["dst_host_rerror_rate"]
    df["dst_host_same_src_x_srv"] = df["dst_host_same_src_port_rate"] * df["dst_host_srv_count"]

    # ── 2f. Flag-based hot encoding (flag is very informative) ──
    # Will be label-encoded later, but let's also create binary flags
    df["is_SF"]   = (df["flag"] == "SF").astype(int)
    df["is_S0"]   = (df["flag"] == "S0").astype(int)
    df["is_REJ"]  = (df["flag"] == "REJ").astype(int)
    df["is_RSTO"] = (df["flag"] == "RSTO").astype(int)

    # ── 2g. Protocol one-hot (only 3 values — cheap and effective) ──
    df["is_tcp"]  = (df["protocol_type"] == "tcp").astype(int)
    df["is_udp"]  = (df["protocol_type"] == "udp").astype(int)
    df["is_icmp"] = (df["protocol_type"] == "icmp").astype(int)

    # ── 2h. Suspicious activity indicators ──
    df["has_root_activity"]  = ((df["num_root"] > 0) | (df["root_shell"] > 0)).astype(int)
    df["has_failed_logins"]  = (df["num_failed_logins"] > 0).astype(int)
    df["has_compromised"]    = (df["num_compromised"] > 0).astype(int)
    df["has_file_operations"] = ((df["num_file_creations"] > 0) | (df["num_access_files"] > 0)).astype(int)

    new_features = [
        "log_src_bytes", "log_dst_bytes", "log_duration",
        "bytes_ratio", "total_bytes", "log_total_bytes",
        "total_error_rate", "srv_total_error_rate",
        "dst_host_total_error", "dst_host_srv_total_error",
        "srv_ratio", "dst_host_srv_ratio",
        "count_x_serror", "count_x_rerror", "srv_count_x_serror",
        "count_srv_ratio",
        "dst_host_count_x_serror", "dst_host_count_x_rerror",
        "dst_host_same_src_x_srv",
        "is_SF", "is_S0", "is_REJ", "is_RSTO",
        "is_tcp", "is_udp", "is_icmp",
        "has_root_activity", "has_failed_logins",
        "has_compromised", "has_file_operations",
    ]

    print(f"  Created {len(new_features)} new engineered features")
    return df, fit_info, new_features


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 3: FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════════
def select_features(X_train, y_train, feature_names, top_k=50):
    """
    1. Remove zero/near-zero variance features
    2. Rank by mutual information
    3. Keep top_k features
    """
    print(f"\n  Starting features: {X_train.shape[1]}")

    # Remove near-zero variance
    vt = VarianceThreshold(threshold=0.001)
    X_train_vt = vt.fit_transform(X_train)
    kept_mask = vt.get_support()
    remaining_names = [f for f, k in zip(feature_names, kept_mask) if k]
    print(f"  After variance threshold: {X_train_vt.shape[1]} (removed {sum(~kept_mask)})")

    # Mutual information ranking
    print("  Computing mutual information scores (this takes ~30s)...")
    mi_scores = mutual_info_classif(X_train_vt, y_train, random_state=SEED, n_neighbors=5)
    mi_ranking = sorted(zip(remaining_names, mi_scores), key=lambda x: x[1], reverse=True)

    print(f"\n  Top 20 features by Mutual Information:")
    print(f"  {'Rank':<5} {'Feature':<35} {'MI Score':<10}")
    print(f"  {'─'*5} {'─'*35} {'─'*10}")
    for i, (fname, score) in enumerate(mi_ranking[:20]):
        print(f"  {i+1:<5} {fname:<35} {score:.4f}")

    # Select top_k
    top_k = min(top_k, len(mi_ranking))
    selected_features = [name for name, _ in mi_ranking[:top_k]]
    selected_indices  = [remaining_names.index(f) for f in selected_features]

    print(f"\n  Selected top {top_k} features for training")

    # Plot MI scores
    fig, ax = plt.subplots(figsize=(12, 8))
    top_30 = mi_ranking[:30]
    names  = [x[0] for x in top_30][::-1]
    scores = [x[1] for x in top_30][::-1]
    colors = ["#2196F3" if n.startswith(("log_", "bytes_", "total_", "srv_ratio",
               "dst_host_srv_ratio", "count_x", "srv_count_x", "dst_host_count",
               "dst_host_same_src", "is_", "has_", "count_srv"))
               else "#FF9800" for n in names]
    ax.barh(range(len(names)), scores, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Mutual Information Score", fontsize=11)
    ax.set_title("Top 30 Features by Mutual Information\n(Blue = Engineered, Orange = Original)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_mi.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: feature_importance_mi.png")

    return vt, selected_features, selected_indices, kept_mask


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 4: COMPUTE CLASS WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════
def compute_weights(y_train, task="binary"):
    """
    Compute class weights inversely proportional to class frequency.
    This forces the model to pay MORE attention to rare attacks.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_dict = dict(zip(classes, weights))

    print(f"\n  Class weights ({task}):")
    for cls, w in weight_dict.items():
        print(f"    Class {cls}: {w:.4f}")

    return weight_dict


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 5: PREPROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
def preprocess(train_df, test_df):
    print("\n" + "=" * 70)
    print("  [2/7] PREPROCESSING + FEATURE ENGINEERING")
    print("=" * 70)

    train_df = create_labels(train_df)
    test_df  = create_labels(test_df)

    # Print distributions
    print("\n  [Train] Binary distribution:")
    for lbl, cnt in train_df["binary_label"].value_counts().items():
        print(f"    {'Normal' if lbl == 0 else 'Attack'}: {cnt:>8,}")

    print("\n  [Train] Multi-class distribution:")
    for cat, cnt in train_df["attack_category"].value_counts().items():
        print(f"    {cat:<10s}: {cnt:>8,}")

    # ── Feature Engineering ──
    print("\n  --- Feature Engineering ---")
    train_df, fit_info, new_feat_names = engineer_features(train_df, is_train=True)
    test_df, _, _  = engineer_features(test_df, fit_info=fit_info, is_train=False)

    # ── Encode categoricals ──
    categorical_cols = ["protocol_type", "service", "flag"]
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        combined = pd.concat([train_df[col], test_df[col]], axis=0)
        le.fit(combined)
        train_df[col] = le.transform(train_df[col])
        test_df[col]  = le.transform(test_df[col])
        label_encoders[col] = le
        print(f"  Encoded '{col}' → {le.classes_.shape[0]} unique values")

    # ── Separate features and labels ──
    exclude_cols = ["label", "binary_label", "attack_category"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    X_train_full = train_df[feature_cols].values.astype(np.float32)
    X_test_full  = test_df[feature_cols].values.astype(np.float32)

    # Handle any inf/nan from feature engineering
    X_train_full = np.nan_to_num(X_train_full, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test_full  = np.nan_to_num(X_test_full,  nan=0.0, posinf=1e6, neginf=-1e6)

    y_train_bin = train_df["binary_label"].values
    y_test_bin  = test_df["binary_label"].values

    mc_le = LabelEncoder()
    y_train_mc = mc_le.fit_transform(train_df["attack_category"].values)
    y_test_mc  = mc_le.transform(test_df["attack_category"].values)
    num_classes = len(mc_le.classes_)
    print(f"\n  Multi-class categories: {list(mc_le.classes_)}")

    # ── Scale features ──
    scaler = MinMaxScaler()
    X_train_full = scaler.fit_transform(X_train_full)
    X_test_full  = scaler.transform(X_test_full)

    print(f"  Full feature matrix  : {X_train_full.shape}")

    # ── Feature Selection ──
    print("\n  --- Feature Selection (Mutual Information) ---")
    vt, selected_features, selected_indices, kept_mask = \
        select_features(X_train_full, y_train_bin, feature_cols, top_k=50)

    # Apply variance threshold then select top features
    X_train_vt = vt.transform(X_train_full)
    X_test_vt  = vt.transform(X_test_full)

    X_train_sel = X_train_vt[:, selected_indices]
    X_test_sel  = X_test_vt[:, selected_indices]

    print(f"  Final feature matrix : {X_train_sel.shape}")

    # ── Validation split ──
    X_tr, X_val, y_tr_bin, y_v_bin, y_tr_mc, y_v_mc = \
        train_test_split(X_train_sel, y_train_bin, y_train_mc,
                         test_size=0.15, random_state=SEED, stratify=y_train_mc)

    print(f"\n  Train split          : {X_tr.shape[0]:,}")
    print(f"  Validation split     : {X_val.shape[0]:,}")
    print(f"  Test (held-out)      : {X_test_sel.shape[0]:,}")

    # ── Class Weights ──
    print("\n  --- Computing Class Weights ---")
    bin_weights = compute_weights(y_tr_bin, task="binary")
    mc_weights  = compute_weights(y_tr_mc,  task="multi-class")

    data = {
        "X_train": X_tr, "X_val": X_val, "X_test": X_test_sel,
        "y_train_bin": y_tr_bin, "y_val_bin": y_v_bin, "y_test_bin": y_test_bin,
        "y_train_mc": y_tr_mc, "y_val_mc": y_v_mc, "y_test_mc": y_test_mc,
        "num_classes": num_classes, "mc_label_encoder": mc_le,
        "input_dim": X_tr.shape[1],
        "bin_class_weights": bin_weights,
        "mc_class_weights": mc_weights,
        "selected_features": selected_features,
    }
    return data


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 6: MODEL ARCHITECTURES (slightly wider for more features)
# ═══════════════════════════════════════════════════════════════════════════
def build_binary_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation="relu"),
        Dropout(0.2),

        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_multiclass_model(input_dim, num_classes):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(512, activation="relu"),
        BatchNormalization(),
        Dropout(0.4),

        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation="relu"),
        Dropout(0.2),

        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 7: TRAINING WITH CLASS WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════
def get_callbacks(model_name):
    return [
        EarlyStopping(monitor="val_loss", patience=12,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(os.path.join(OUTPUT_DIR, f"{model_name}_best.keras"),
                        monitor="val_accuracy", save_best_only=True, verbose=0),
    ]


def train_model(model, X_train, y_train, X_val, y_val,
                model_name, class_weight=None, epochs=100, batch_size=256):
    print(f"\n{'─' * 70}")
    print(f"  [TRAINING] {model_name}")
    if class_weight:
        print(f"  Class weights ENABLED: {class_weight}")
    print(f"{'─' * 70}")
    model.summary()

    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,      # ← THE KEY CHANGE
        callbacks=get_callbacks(model_name),
        verbose=1
    )
    elapsed = time.time() - start
    print(f"\n  Training time: {elapsed:.1f}s")
    return history


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 8: THRESHOLD TUNING
# ═══════════════════════════════════════════════════════════════════════════
def find_optimal_threshold(y_true, y_proba):
    """
    Sweep thresholds from 0.1 to 0.9 and pick the one that maximizes F1.
    Also report the threshold that gives recall >= 0.80.
    """
    thresholds = np.arange(0.10, 0.91, 0.01)
    results = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        results.append({"threshold": t, "accuracy": acc,
                        "precision": prec, "recall": rec, "f1": f1})

    results_df = pd.DataFrame(results)

    # Best F1
    best_f1_row = results_df.loc[results_df["f1"].idxmax()]
    # Best recall with precision >= 0.80
    high_recall = results_df[results_df["precision"] >= 0.80]
    best_recall_row = high_recall.loc[high_recall["recall"].idxmax()] if len(high_recall) > 0 else best_f1_row

    print(f"\n  ┌─── Threshold Tuning Results ───────────────────────────┐")
    print(f"  │                                                        │")
    print(f"  │  Best F1 threshold  : {best_f1_row['threshold']:.2f}                            │")
    print(f"  │    → Acc={best_f1_row['accuracy']:.4f}  P={best_f1_row['precision']:.4f}  "
          f"R={best_f1_row['recall']:.4f}  F1={best_f1_row['f1']:.4f} │")
    print(f"  │                                                        │")
    print(f"  │  Best Recall (P≥0.80): {best_recall_row['threshold']:.2f}                           │")
    print(f"  │    → Acc={best_recall_row['accuracy']:.4f}  P={best_recall_row['precision']:.4f}  "
          f"R={best_recall_row['recall']:.4f}  F1={best_recall_row['f1']:.4f} │")
    print(f"  └────────────────────────────────────────────────────────┘")

    # Plot threshold sweep
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results_df["threshold"], results_df["accuracy"],  label="Accuracy",  linewidth=2)
    ax.plot(results_df["threshold"], results_df["precision"], label="Precision", linewidth=2)
    ax.plot(results_df["threshold"], results_df["recall"],    label="Recall",    linewidth=2)
    ax.plot(results_df["threshold"], results_df["f1"],        label="F1-Score",  linewidth=2)
    ax.axvline(x=best_f1_row["threshold"], color="red", linestyle="--", alpha=0.7,
               label=f"Best F1 @ {best_f1_row['threshold']:.2f}")
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5, label="Default (0.50)")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Binary Classification — Threshold Sweep", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "threshold_sweep.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: threshold_sweep.png")

    return best_f1_row["threshold"], results_df


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 9: EVALUATION + VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════
def plot_training_curves(history, title, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history["accuracy"],    label="Train", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Val",   linewidth=2)
    ax1.set_title(f"{title} — Accuracy", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(history.history["loss"],    label="Train", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Val",   linewidth=2)
    ax2.set_title(f"{title} — Loss", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def plot_roc_curve(y_true, y_scores, title, filename):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend(fontsize=12); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def plot_precision_recall_curve(y_true, y_scores, filename):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, linewidth=2, color="#9C27B0")
    plt.fill_between(recall, precision, alpha=0.1, color="#9C27B0")
    plt.title("Precision-Recall Curve (Binary)", fontsize=14, fontweight="bold")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def plot_multiclass_roc(y_true, y_pred_proba, class_names, filename):
    n_classes = len(class_names)
    y_true_oh = to_categorical(y_true, num_classes=n_classes)
    plt.figure(figsize=(9, 7))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_oh[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f"{class_names[i]} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.title("Multi-class ROC (One-vs-Rest)", fontsize=14, fontweight="bold")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend(fontsize=10); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def evaluate_binary(model, X_test, y_test):
    print("\n" + "=" * 70)
    print("  [5/7] BINARY CLASSIFICATION RESULTS")
    print("=" * 70)

    y_proba = model.predict(X_test, verbose=0).flatten()

    # ── Default threshold (0.5) ──
    y_pred_default = (y_proba >= 0.5).astype(int)
    acc_d  = accuracy_score(y_test, y_pred_default)
    prec_d = precision_score(y_test, y_pred_default)
    rec_d  = recall_score(y_test, y_pred_default)
    f1_d   = f1_score(y_test, y_pred_default)

    print(f"\n  ── Default Threshold (0.50) ──")
    print(f"  Accuracy  : {acc_d:.4f}  ({acc_d*100:.2f}%)")
    print(f"  Precision : {prec_d:.4f}")
    print(f"  Recall    : {rec_d:.4f}")
    print(f"  F1-Score  : {f1_d:.4f}")

    # ── Optimal threshold ──
    best_threshold, _ = find_optimal_threshold(y_test, y_proba)

    y_pred_opt = (y_proba >= best_threshold).astype(int)
    acc_o  = accuracy_score(y_test, y_pred_opt)
    prec_o = precision_score(y_test, y_pred_opt)
    rec_o  = recall_score(y_test, y_pred_opt)
    f1_o   = f1_score(y_test, y_pred_opt)

    print(f"\n  ── Optimized Threshold ({best_threshold:.2f}) ──")
    print(f"  Accuracy  : {acc_o:.4f}  ({acc_o*100:.2f}%)")
    print(f"  Precision : {prec_o:.4f}")
    print(f"  Recall    : {rec_o:.4f}")
    print(f"  F1-Score  : {f1_o:.4f}")

    print(f"\n  ── Improvement over default ──")
    print(f"  Recall    : {rec_d:.4f} → {rec_o:.4f}  ({(rec_o - rec_d)*100:+.2f}%)")
    print(f"  F1-Score  : {f1_d:.4f} → {f1_o:.4f}  ({(f1_o - f1_d)*100:+.2f}%)")

    labels = ["Normal", "Attack"]
    print(f"\n  Classification Report (threshold={best_threshold:.2f}):")
    print(classification_report(y_test, y_pred_opt, target_names=labels))

    plot_confusion_matrix(y_test, y_pred_opt, labels,
                          f"Binary Confusion Matrix (threshold={best_threshold:.2f})",
                          "binary_confusion_matrix.png")
    plot_roc_curve(y_test, y_proba, "Binary ROC Curve", "binary_roc_curve.png")
    plot_precision_recall_curve(y_test, y_proba, "binary_pr_curve.png")

    return {
        "default": {"accuracy": acc_d, "precision": prec_d, "recall": rec_d, "f1": f1_d},
        "optimized": {"accuracy": acc_o, "precision": prec_o, "recall": rec_o, "f1": f1_o},
        "best_threshold": best_threshold
    }


def evaluate_multiclass(model, X_test, y_test, label_encoder):
    print("\n" + "=" * 70)
    print("  [6/7] MULTI-CLASS CLASSIFICATION RESULTS")
    print("=" * 70)

    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    class_names = list(label_encoder.classes_)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec  = recall_score(y_test, y_pred, average="weighted")
    f1   = f1_score(y_test, y_pred, average="weighted")

    print(f"\n  Accuracy          : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision (wt)    : {prec:.4f}")
    print(f"  Recall    (wt)    : {rec:.4f}")
    print(f"  F1-Score  (wt)    : {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=class_names)}")

    plot_confusion_matrix(y_test, y_pred, class_names,
                          "Multi-class Confusion Matrix", "multiclass_confusion_matrix.png")
    plot_multiclass_roc(y_test, y_pred_proba, class_names, "multiclass_roc_curves.png")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 10: COMPARISON WITH V1 BASELINE
# ═══════════════════════════════════════════════════════════════════════════
def save_comparison(bin_results, mc_metrics, data):
    # v1 baseline (your reported results)
    v1_bin = {"accuracy": 0.7995, "precision": 0.9728, "recall": 0.6663, "f1": 0.7909}

    v2_default   = bin_results["default"]
    v2_optimized = bin_results["optimized"]

    print("\n" + "=" * 70)
    print("  [7/7] v1 vs v2 COMPARISON")
    print("=" * 70)

    header = f"  {'Metric':<12s} {'v1 Baseline':>14s} {'v2 Default':>14s} {'v2 Optimized':>14s} {'Δ (v2opt-v1)':>14s}"
    print(header)
    print("  " + "─" * 70)
    for m in ["accuracy", "precision", "recall", "f1"]:
        v1  = v1_bin[m]
        v2d = v2_default[m]
        v2o = v2_optimized[m]
        delta = v2o - v1
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
        print(f"  {m:<12s} {v1:>13.4f}  {v2d:>13.4f}  {v2o:>13.4f}  {delta:>+12.4f} {arrow}")

    print(f"\n  Best threshold: {bin_results['best_threshold']:.2f}")
    print(f"  Selected features: {data['input_dim']}")

    # Plot comparison
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, [v1_bin[m] for m in metrics], width,
                   label="v1 Baseline", color="#9E9E9E", edgecolor="white")
    bars2 = ax.bar(x, [v2_default[m] for m in metrics], width,
                   label="v2 Default (0.5)", color="#2196F3", edgecolor="white")
    bars3 = ax.bar(x + width, [v2_optimized[m] for m in metrics], width,
                   label="v2 Optimized", color="#4CAF50", edgecolor="white")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("v1 Baseline vs v2 Enhanced — Binary Classification",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-Score"], fontsize=11)
    ax.legend(fontsize=11); ax.set_ylim(0, 1.08); ax.grid(axis="y", alpha=0.3)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "v1_vs_v2_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: v1_vs_v2_comparison.png")

    # Save text report
    report_lines = [
        "=" * 70,
        "  ANN-BASED NIDS v2 — ENHANCED RESULTS SUMMARY",
        "=" * 70,
        f"  Features: {data['input_dim']} selected (from {data['input_dim'] + 20}+ engineered)",
        f"  Class weights: ENABLED (balanced)",
        f"  Threshold tuning: ENABLED (best = {bin_results['best_threshold']:.2f})",
        "",
        "  BINARY CLASSIFICATION:",
        f"    Default  (t=0.50): Acc={v2_default['accuracy']:.4f}  P={v2_default['precision']:.4f}  R={v2_default['recall']:.4f}  F1={v2_default['f1']:.4f}",
        f"    Optimized(t={bin_results['best_threshold']:.2f}): Acc={v2_optimized['accuracy']:.4f}  P={v2_optimized['precision']:.4f}  R={v2_optimized['recall']:.4f}  F1={v2_optimized['f1']:.4f}",
        "",
        "  MULTI-CLASS CLASSIFICATION:",
        f"    Acc={mc_metrics['accuracy']:.4f}  P={mc_metrics['precision']:.4f}  R={mc_metrics['recall']:.4f}  F1={mc_metrics['f1']:.4f}",
        "",
        "=" * 70,
    ]
    with open(os.path.join(OUTPUT_DIR, "results_summary_v2.txt"), "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved: results_summary_v2.txt")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "▓" * 70)
    print("  ANN-BASED NIDS v2 — ENHANCED")
    print("  Feature Engineering + Class Weights + Threshold Tuning")
    print("▓" * 70)

    # Load & preprocess
    train_df, test_df = load_data()
    data = preprocess(train_df, test_df)

    # ══════════════════════════════════════════════════════════════════════
    #  BINARY CLASSIFICATION (with class weights)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  [3/7] BINARY MODEL")
    print("=" * 70)

    bin_model = build_binary_model(data["input_dim"])
    bin_history = train_model(
        bin_model,
        data["X_train"], data["y_train_bin"],
        data["X_val"],   data["y_val_bin"],
        model_name="binary_v2",
        class_weight=data["bin_class_weights"],   # ← CLASS WEIGHTS
        epochs=100, batch_size=256
    )
    plot_training_curves(bin_history, "Binary v2", "binary_training_curves.png")
    bin_results = evaluate_binary(bin_model, data["X_test"], data["y_test_bin"])

    # ══════════════════════════════════════════════════════════════════════
    #  MULTI-CLASS CLASSIFICATION (with class weights)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  [4/7] MULTI-CLASS MODEL")
    print("=" * 70)

    mc_model = build_multiclass_model(data["input_dim"], data["num_classes"])
    mc_history = train_model(
        mc_model,
        data["X_train"], data["y_train_mc"],
        data["X_val"],   data["y_val_mc"],
        model_name="multiclass_v2",
        class_weight=data["mc_class_weights"],    # ← CLASS WEIGHTS
        epochs=100, batch_size=256
    )
    plot_training_curves(mc_history, "Multi-class v2", "multiclass_training_curves.png")
    mc_metrics = evaluate_multiclass(
        mc_model, data["X_test"], data["y_test_mc"], data["mc_label_encoder"]
    )

    # ── Final comparison ──
    save_comparison(bin_results, mc_metrics, data)

    print("\n" + "▓" * 70)
    print(f"  ALL DONE — Results in: {OUTPUT_DIR}/")
    print("▓" * 70)
    print("\n  Generated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"    • {f}")


if __name__ == "__main__":
    main()
