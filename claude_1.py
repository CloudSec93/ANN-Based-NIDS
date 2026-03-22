#!/usr/bin/env python3
"""
=============================================================================
 ANN-based Network Intrusion Detection System (NIDS) — NSL-KDD Dataset
 Author : Senior ML Engineer Build
 Dataset: https://www.kaggle.com/datasets/hassan06/nslkdd
=============================================================================
 Performs BOTH:
   1. Binary   classification  (Normal vs Attack)
   2. Multi-class classification (Normal, DoS, Probe, R2L, U2R)
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

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc,
    precision_recall_curve
)
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_DIR   = "/home/harsh/NIDS_ANN/nsl-kdd"
OUTPUT_DIR = "/home/harsh/NIDS_ANN/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(BASE_DIR, "KDDTrain+.txt")
TEST_FILE  = os.path.join(BASE_DIR, "KDDTest+.txt")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─── 1. NSL-KDD Column Names (41 features + label + difficulty) ─────────────
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

# Attack-type → category mapping (NSL-KDD standard)
ATTACK_MAP = {
    "normal": "Normal",
    # DoS attacks
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS", "apache2": "DoS",
    "processtable": "DoS", "udpstorm": "DoS",
    # Probe attacks
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L attacks
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L",
    "snmpgetattack": "R2L", "snmpguess": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "worm": "R2L", "httptunnel": "R2L",
    # U2R attacks
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "xterm": "U2R", "ps": "U2R",
    "sqlattack": "U2R", "httptunnel": "R2L",
}


# ─── 2. Load & Preprocess Data ──────────────────────────────────────────────
def load_data():
    print("=" * 70)
    print("  LOADING NSL-KDD DATASET")
    print("=" * 70)

    train_df = pd.read_csv(TRAIN_FILE, header=None, names=COL_NAMES)
    test_df  = pd.read_csv(TEST_FILE,  header=None, names=COL_NAMES)

    print(f"  Train samples : {len(train_df):>8,}")
    print(f"  Test  samples : {len(test_df):>8,}")

    # Drop difficulty_level (last column — not a feature)
    train_df.drop("difficulty_level", axis=1, inplace=True)
    test_df.drop("difficulty_level", axis=1, inplace=True)

    return train_df, test_df


def create_labels(df):
    """Create both binary and multi-class labels."""
    # Binary: Normal=0, Attack=1
    df["binary_label"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)

    # Multi-class: 5 categories
    df["attack_category"] = df["label"].map(ATTACK_MAP)
    # Handle any unmapped attacks → assign to nearest category
    df["attack_category"].fillna("Unknown", inplace=True)

    return df


def preprocess(train_df, test_df):
    print("\n" + "=" * 70)
    print("  PREPROCESSING")
    print("=" * 70)

    train_df = create_labels(train_df)
    test_df  = create_labels(test_df)

    # ── Print class distributions ──
    print("\n  [Train] Binary distribution:")
    for lbl, cnt in train_df["binary_label"].value_counts().items():
        print(f"    {'Normal' if lbl == 0 else 'Attack'}: {cnt:>8,}")

    print("\n  [Train] Multi-class distribution:")
    for cat, cnt in train_df["attack_category"].value_counts().items():
        print(f"    {cat:<10s}: {cnt:>8,}")

    # ── Encode categorical features ──
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
    feature_cols = [c for c in train_df.columns
                    if c not in ["label", "binary_label", "attack_category"]]

    X_train = train_df[feature_cols].values.astype(np.float32)
    X_test  = test_df[feature_cols].values.astype(np.float32)

    y_train_bin = train_df["binary_label"].values
    y_test_bin  = test_df["binary_label"].values

    # Encode multi-class labels
    mc_le = LabelEncoder()
    y_train_mc = mc_le.fit_transform(train_df["attack_category"].values)
    y_test_mc  = mc_le.transform(test_df["attack_category"].values)
    num_classes = len(mc_le.classes_)
    print(f"\n  Multi-class categories: {list(mc_le.classes_)}")
    print(f"  Number of classes     : {num_classes}")

    # ── Scale features (MinMax to [0, 1]) ──
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"\n  Feature matrix shape  : {X_train.shape}")
    print(f"  Scaling               : MinMaxScaler [0, 1]")

    # ── Validation split from training data ──
    X_train_s, X_val, y_train_bin_s, y_val_bin, y_train_mc_s, y_val_mc = \
        train_test_split(X_train, y_train_bin, y_train_mc,
                         test_size=0.15, random_state=SEED, stratify=y_train_mc)

    print(f"  Train split           : {X_train_s.shape[0]:,}")
    print(f"  Validation split      : {X_val.shape[0]:,}")
    print(f"  Test (held-out)       : {X_test.shape[0]:,}")

    data = {
        "X_train": X_train_s, "X_val": X_val, "X_test": X_test,
        "y_train_bin": y_train_bin_s, "y_val_bin": y_val_bin, "y_test_bin": y_test_bin,
        "y_train_mc": y_train_mc_s, "y_val_mc": y_val_mc, "y_test_mc": y_test_mc,
        "num_classes": num_classes, "mc_label_encoder": mc_le,
        "input_dim": X_train_s.shape[1],
    }
    return data


# ─── 3. Build ANN Models ────────────────────────────────────────────────────
def build_binary_model(input_dim):
    """Deep ANN for Binary Classification (Normal vs Attack)."""
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

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_multiclass_model(input_dim, num_classes):
    """Deep ANN for Multi-class Classification (5 attack categories)."""
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

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ─── 4. Training Utilities ──────────────────────────────────────────────────
def get_callbacks(model_name):
    return [
        EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=5, min_lr=1e-6, verbose=1
        ),
        ModelCheckpoint(
            os.path.join(OUTPUT_DIR, f"{model_name}_best.keras"),
            monitor="val_accuracy", save_best_only=True, verbose=0
        ),
    ]


def train_model(model, X_train, y_train, X_val, y_val, model_name, epochs=100, batch_size=256):
    print(f"\n{'─' * 70}")
    print(f"  TRAINING: {model_name}")
    print(f"{'─' * 70}")
    model.summary()

    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(model_name),
        verbose=1
    )
    elapsed = time.time() - start
    print(f"\n  Training time: {elapsed:.1f}s")
    return history


# ─── 5. Evaluation & Visualization ──────────────────────────────────────────
def plot_training_curves(history, title, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(history.history["accuracy"],     label="Train Accuracy", linewidth=2)
    ax1.plot(history.history["val_accuracy"],  label="Val Accuracy",   linewidth=2)
    ax1.set_title(f"{title} — Accuracy", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(history.history["loss"],     label="Train Loss", linewidth=2)
    ax2.plot(history.history["val_loss"],  label="Val Loss",   linewidth=2)
    ax2.set_title(f"{title} — Loss", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(True, alpha=0.3)

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
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(fontsize=12); plt.grid(True, alpha=0.3)
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
    plt.title("Multi-class ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(fontsize=10); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def evaluate_binary(model, X_test, y_test):
    print("\n" + "=" * 70)
    print("  BINARY CLASSIFICATION RESULTS (Normal vs Attack)")
    print("=" * 70)

    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    print(f"\n  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")

    labels = ["Normal", "Attack"]
    print(f"\n{classification_report(y_test, y_pred, target_names=labels)}")

    plot_confusion_matrix(y_test, y_pred, labels,
                          "Binary Classification — Confusion Matrix",
                          "binary_confusion_matrix.png")
    plot_roc_curve(y_test, y_pred_prob,
                   "Binary Classification — ROC Curve",
                   "binary_roc_curve.png")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def evaluate_multiclass(model, X_test, y_test, label_encoder):
    print("\n" + "=" * 70)
    print("  MULTI-CLASS CLASSIFICATION RESULTS (5 Attack Categories)")
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
                          "Multi-class — Confusion Matrix",
                          "multiclass_confusion_matrix.png")
    plot_multiclass_roc(y_test, y_pred_proba, class_names,
                        "multiclass_roc_curves.png")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def plot_comparison_bar(bin_metrics, mc_metrics, filename):
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, [bin_metrics[m] for m in metrics], width,
                   label="Binary", color="#2196F3", edgecolor="white")
    bars2 = ax.bar(x + width/2, [mc_metrics[m] for m in metrics], width,
                   label="Multi-class", color="#FF9800", edgecolor="white")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Binary vs Multi-class Performance Comparison",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-Score"], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def plot_class_distribution(train_df, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Binary
    counts_bin = train_df["binary_label"].value_counts()
    ax1.bar(["Normal", "Attack"], counts_bin.values, color=["#4CAF50", "#F44336"])
    ax1.set_title("Binary Label Distribution", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Count")
    for i, v in enumerate(counts_bin.values):
        ax1.text(i, v + 500, f"{v:,}", ha="center", fontweight="bold")

    # Multi-class
    counts_mc = train_df["attack_category"].value_counts()
    colors = ["#4CAF50", "#F44336", "#2196F3", "#FF9800", "#9C27B0"]
    ax2.bar(counts_mc.index, counts_mc.values, color=colors[:len(counts_mc)])
    ax2.set_title("Attack Category Distribution", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Count"); ax2.tick_params(axis="x", rotation=15)
    for i, v in enumerate(counts_mc.values):
        ax2.text(i, v + 500, f"{v:,}", ha="center", fontweight="bold", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


# ─── 6. Summary Report ──────────────────────────────────────────────────────
def save_summary(bin_metrics, mc_metrics, data):
    report = []
    report.append("=" * 70)
    report.append("  ANN-BASED NIDS — FINAL RESULTS SUMMARY")
    report.append("=" * 70)
    report.append(f"\n  Dataset        : NSL-KDD")
    report.append(f"  Input features : {data['input_dim']}")
    report.append(f"  Train samples  : {data['X_train'].shape[0]:,}")
    report.append(f"  Val samples    : {data['X_val'].shape[0]:,}")
    report.append(f"  Test samples   : {data['X_test'].shape[0]:,}")
    report.append(f"\n{'─' * 70}")
    report.append(f"  BINARY CLASSIFICATION (Normal vs Attack)")
    report.append(f"{'─' * 70}")
    for k, v in bin_metrics.items():
        report.append(f"    {k:<12s}: {v:.4f}  ({v*100:.2f}%)")
    report.append(f"\n{'─' * 70}")
    report.append(f"  MULTI-CLASS CLASSIFICATION (5 Categories)")
    report.append(f"{'─' * 70}")
    for k, v in mc_metrics.items():
        report.append(f"    {k:<12s}: {v:.4f}  ({v*100:.2f}%)")
    report.append("\n" + "=" * 70)

    text = "\n".join(report)
    print(text)

    with open(os.path.join(OUTPUT_DIR, "results_summary.txt"), "w") as f:
        f.write(text)
    print(f"\n  Summary saved to: {OUTPUT_DIR}/results_summary.txt")


# ─── 7. Main Pipeline ───────────────────────────────────────────────────────
def main():
    print("\n" + "▓" * 70)
    print("  ANN-BASED NETWORK INTRUSION DETECTION SYSTEM")
    print("  Dataset: NSL-KDD  |  Framework: TensorFlow/Keras")
    print("▓" * 70)

    # ── Load data ──
    train_df, test_df = load_data()

    # ── Preprocess ──
    data = preprocess(train_df, test_df)

    # ── Plot class distribution ──
    train_df = create_labels(train_df)
    plot_class_distribution(train_df, "class_distribution.png")

    # ══════════════════════════════════════════════════════════════════════
    #  BINARY CLASSIFICATION
    # ══════════════════════════════════════════════════════════════════════
    bin_model = build_binary_model(data["input_dim"])
    bin_history = train_model(
        bin_model,
        data["X_train"], data["y_train_bin"],
        data["X_val"],   data["y_val_bin"],
        model_name="binary_nids",
        epochs=100, batch_size=256
    )
    plot_training_curves(bin_history, "Binary Classification",
                         "binary_training_curves.png")
    bin_metrics = evaluate_binary(bin_model, data["X_test"], data["y_test_bin"])

    # ══════════════════════════════════════════════════════════════════════
    #  MULTI-CLASS CLASSIFICATION
    # ══════════════════════════════════════════════════════════════════════
    mc_model = build_multiclass_model(data["input_dim"], data["num_classes"])
    mc_history = train_model(
        mc_model,
        data["X_train"], data["y_train_mc"],
        data["X_val"],   data["y_val_mc"],
        model_name="multiclass_nids",
        epochs=100, batch_size=256
    )
    plot_training_curves(mc_history, "Multi-class Classification",
                         "multiclass_training_curves.png")
    mc_metrics = evaluate_multiclass(
        mc_model, data["X_test"], data["y_test_mc"], data["mc_label_encoder"]
    )

    # ── Comparison plot ──
    plot_comparison_bar(bin_metrics, mc_metrics, "performance_comparison.png")

    # ── Final summary ──
    save_summary(bin_metrics, mc_metrics, data)

    print("\n" + "▓" * 70)
    print(f"  ALL DONE — Results saved to: {OUTPUT_DIR}/")
    print("▓" * 70)
    print("\n  Generated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"    • {f}")


if __name__ == "__main__":
    main()