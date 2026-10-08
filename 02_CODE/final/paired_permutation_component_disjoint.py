from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "data" / "raw"

SPLIT_DIR = (
    ROOT
    / "results"
    / "temporal_overlap_audit"
)

OUT_DIR = (
    ROOT
    / "results"
    / "component_disjoint_experiment"
)

RAW_CKPT = OUT_DIR / "raw_ce_best.pt"
SIL_CKPT = OUT_DIR / "sil_v2_best.pt"

NUM_CLASSES = 7
REPR_DIM = 128

N_PERM = 20000
SEED = 20261005


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 78)
print("PAIRED PERMUTATION TEST â€” COMPONENT-DISJOINT TEST")
print("=" * 78)

print()
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD ORIGINAL DATA
# EXACTLY AS TRAINING SCRIPT
# ============================================================

X_train_orig = np.load(
    DATA_DIR / "data" / "X_train.npy"
)

y_train_orig = np.load(
    DATA_DIR / "label" / "y_train.npy"
).astype(np.int64)

X_val_orig = np.load(
    DATA_DIR / "data" / "X_val.npy"
)

y_val_orig = np.load(
    DATA_DIR / "label" / "y_val.npy"
).astype(np.int64)

X_test_orig = np.load(
    DATA_DIR / "data" / "X_test.npy"
)

y_test_orig = np.load(
    DATA_DIR / "label" / "y_test.npy"
).astype(np.int64)


# ============================================================
# ORIGINAL GLOBAL-ID OFFSETS
# EXACTLY AS TRAINING SCRIPT
# ============================================================

OFFSETS = {
    "train": 0,
    "val": len(X_train_orig),
    "test": (
        len(X_train_orig)
        + len(X_val_orig)
    ),
}

ORIGINAL_X = {
    "train": X_train_orig,
    "val": X_val_orig,
    "test": X_test_orig,
}

ORIGINAL_Y = {
    "train": y_train_orig,
    "val": y_val_orig,
    "test": y_test_orig,
}


def decode_global_id(gid):

    gid = int(gid)

    if gid < OFFSETS["val"]:
        return "train", gid

    if gid < OFFSETS["test"]:
        return (
            "val",
            gid - OFFSETS["val"]
        )

    return (
        "test",
        gid - OFFSETS["test"]
    )


# ============================================================
# LOAD COMPONENT-DISJOINT TEST IDS
# ============================================================

test_global_ids = np.load(
    SPLIT_DIR
    / "component_disjoint_test_global_ids.npy"
).astype(np.int64)

print()
print("Component-disjoint test IDs:", len(test_global_ids))


# ============================================================
# MATERIALIZE TEST SET
# EXACTLY AS TRAINING SCRIPT
# ============================================================

X_list = []
y_list = []

source_counts = {
    "train": 0,
    "val": 0,
    "test": 0
}

for gid in test_global_ids:

    source_split, local_idx = decode_global_id(gid)

    X_list.append(
        ORIGINAL_X[source_split][local_idx]
    )

    y_list.append(
        ORIGINAL_Y[source_split][local_idx]
    )

    source_counts[source_split] += 1


X_test = np.asarray(
    X_list,
    dtype=np.float32
)

y_test = np.asarray(
    y_list,
    dtype=np.int64
)

print("Materialized test shape:", X_test.shape)
print("Materialized labels:", y_test.shape)

print()
print("Test-source composition:")
print("  original train:", source_counts["train"])
print("  original val  :", source_counts["val"])
print("  original test :", source_counts["test"])


# ============================================================
# RECONSTRUCT NEW TRAIN SET
# NEEDED FOR TRAIN-FITTED NORMALIZATION
# ============================================================

train_global_ids = np.load(
    SPLIT_DIR
    / "component_disjoint_train_global_ids.npy"
).astype(np.int64)

X_train_list = []

for gid in train_global_ids:

    source_split, local_idx = decode_global_id(gid)

    X_train_list.append(
        ORIGINAL_X[source_split][local_idx]
    )

X_train = np.asarray(
    X_train_list,
    dtype=np.float32
)

print()
print("Component-disjoint train shape:", X_train.shape)


# ============================================================
# TRAIN-FITTED PER-SUBCARRIER NORMALIZATION
# EXACTLY AS TRAINING SCRIPT
# ============================================================

train_mean = X_train.mean(
    axis=(0, 1),
    keepdims=True
)

train_std = X_train.std(
    axis=(0, 1),
    keepdims=True
)

train_std = np.maximum(
    train_std,
    1e-6
)


def normalize(X):

    return (
        (X - train_mean)
        / train_std
    ).astype(np.float32)


X_test = normalize(X_test)

print()
print(
    "Normalized train mean:",
    float(
        normalize(X_train).mean()
    )
)

print(
    "Normalized train std:",
    float(
        normalize(X_train).std()
    )
)


# ============================================================
# MODEL
# EXACTLY AS TRAINING SCRIPT
# ============================================================

class SimpleCSIEncoder(nn.Module):

    def __init__(
        self,
        repr_dim=128,
        num_classes=7
    ):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=(7, 5),
                padding="same"
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=(5, 3),
                padding="same"
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding="same"
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.projection = nn.Linear(
            128,
            repr_dim
        )

        self.classifier = nn.Linear(
            repr_dim,
            num_classes
        )


    def encode(self, x):

        h = self.features(x)

        h = h.flatten(1)

        z = self.projection(h)

        return z


    def forward(self, x):

        z = self.encode(x)

        logits = self.classifier(z)

        return logits, z


# ============================================================
# LOAD CHECKPOINTS
# ============================================================

raw_ckpt = torch.load(
    RAW_CKPT,
    map_location=DEVICE,
    weights_only=False
)

sil_ckpt = torch.load(
    SIL_CKPT,
    map_location=DEVICE,
    weights_only=False
)

raw_model = SimpleCSIEncoder(
    repr_dim=REPR_DIM,
    num_classes=NUM_CLASSES
).to(DEVICE)

sil_model = SimpleCSIEncoder(
    repr_dim=REPR_DIM,
    num_classes=NUM_CLASSES
).to(DEVICE)

raw_model.load_state_dict(
    raw_ckpt["model"]
)

sil_model.load_state_dict(
    sil_ckpt["model"]
)

raw_model.eval()
sil_model.eval()


# ============================================================
# PREDICTIONS
# ============================================================

X_tensor = torch.from_numpy(
    X_test
).float().unsqueeze(1).to(DEVICE)

y_tensor = torch.from_numpy(
    y_test
).long().to(DEVICE)


with torch.no_grad():

    raw_logits, _ = raw_model(
        X_tensor
    )

    sil_logits, _ = sil_model(
        X_tensor
    )


raw_pred = (
    raw_logits
    .argmax(dim=1)
    .cpu()
    .numpy()
)

sil_pred = (
    sil_logits
    .argmax(dim=1)
    .cpu()
    .numpy()
)


# ============================================================
# ACCURACY
# ============================================================

raw_correct = (
    raw_pred == y_test
)

sil_correct = (
    sil_pred == y_test
)

raw_acc = raw_correct.mean()

sil_acc = sil_correct.mean()

observed = (
    sil_acc - raw_acc
)


print()
print("=" * 78)
print("OBSERVED RESULT")
print("=" * 78)

print(
    f"Raw CE accuracy : "
    f"{raw_acc * 100:.2f}%"
)

print(
    f"SIL-v2 accuracy : "
    f"{sil_acc * 100:.2f}%"
)

print(
    f"Observed SIL - Raw: "
    f"{observed * 100:+.2f} pp"
)


# ============================================================
# PAIRED DISAGREEMENTS
# ============================================================

sil_only = np.sum(
    sil_correct & ~raw_correct
)

raw_only = np.sum(
    raw_correct & ~sil_correct
)

both_correct = np.sum(
    sil_correct & raw_correct
)

both_wrong = np.sum(
    ~sil_correct & ~raw_correct
)

n_disagree = (
    sil_only + raw_only
)

print()
print(
    "SIL-only correct :",
    int(sil_only)
)

print(
    "Raw-only correct :",
    int(raw_only)
)

print(
    "Both correct     :",
    int(both_correct)
)

print(
    "Both wrong       :",
    int(both_wrong)
)

print(
    "Disagreement cases:",
    int(n_disagree)
)


# ============================================================
# EXACT PAIRED PERMUTATION TEST
# ============================================================

rng = np.random.default_rng(
    SEED
)

extreme = 0

for _ in range(N_PERM):

    sil_wins = rng.binomial(
        n_disagree,
        0.5
    )

    raw_wins = (
        n_disagree - sil_wins
    )

    perm_difference = (
        sil_wins - raw_wins
    ) / len(y_test)

    if (
        abs(perm_difference)
        >= abs(observed)
    ):

        extreme += 1


p_value = (
    extreme + 1
) / (
    N_PERM + 1
)


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 78)
print("PAIRED PERMUTATION RESULT")
print("=" * 78)

print(
    f"Observed difference : "
    f"{observed * 100:+.2f} pp"
)

print(
    f"Two-sided p-value   : "
    f"{p_value:.6f}"
)

print(
    f"Permutations        : "
    f"{N_PERM}"
)

if p_value < 0.05:

    print(
        "DECISION: SUPPORTED"
    )

else:

    print(
        "DECISION: NOT SIGNIFICANT "
        "AT alpha=0.05"
    )


# ============================================================
# SAVE
# ============================================================

out = {

    "protocol":
        "component_disjoint",

    "test_samples":
        int(len(y_test)),

    "source_counts":
        source_counts,

    "raw_accuracy":
        float(raw_acc),

    "sil_accuracy":
        float(sil_acc),

    "observed_difference_pp":
        float(observed * 100),

    "sil_only_correct":
        int(sil_only),

    "raw_only_correct":
        int(raw_only),

    "both_correct":
        int(both_correct),

    "both_wrong":
        int(both_wrong),

    "disagreement_cases":
        int(n_disagree),

    "n_permutations":
        N_PERM,

    "seed":
        SEED,

    "paired_permutation_p":
        float(p_value)
}


out_path = (
    OUT_DIR
    / "paired_permutation_component_disjoint.json"
)

with open(
    out_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        out,
        f,
        indent=2
    )


print()
print("Saved:")
print(out_path)

print()
print("=" * 78)
print("DONE")
print("=" * 78)
