from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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

EXP_DIR = (
    ROOT
    / "results"
    / "component_disjoint_experiment"
)

SEED = 20261005
N_BOOT = 20000

NUM_CLASSES = 7
REPR_DIM = 128

BATCH_SIZE = 64

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# RECREATE EXACT MODEL
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

        return self.projection(h)

    def forward(self, x):

        z = self.encode(x)

        logits = self.classifier(z)

        return logits, z


# ============================================================
# LOAD ORIGINAL DATA
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
# GLOBAL-ID DECODING
# ============================================================

offset_val = len(X_train_orig)

offset_test = (
    len(X_train_orig)
    + len(X_val_orig)
)


def decode_global_id(gid):

    gid = int(gid)

    if gid < offset_val:

        return (
            X_train_orig,
            y_train_orig,
            gid
        )

    if gid < offset_test:

        return (
            X_val_orig,
            y_val_orig,
            gid - offset_val
        )

    return (
        X_test_orig,
        y_test_orig,
        gid - offset_test
    )


# ============================================================
# LOAD COMPONENT-DISJOINT TEST SET
# ============================================================

test_global_ids = np.load(
    SPLIT_DIR
    / "component_disjoint_test_global_ids.npy"
).astype(np.int64)


X_test = []
y_test = []

for gid in test_global_ids:

    X_source, y_source, idx = (
        decode_global_id(gid)
    )

    X_test.append(
        X_source[idx]
    )

    y_test.append(
        y_source[idx]
    )


X_test = np.asarray(
    X_test,
    dtype=np.float32
)

y_test = np.asarray(
    y_test,
    dtype=np.int64
)


# ============================================================
# EXACT TRAIN-FITTED NORMALIZATION
# ============================================================

train_global_ids = np.load(
    SPLIT_DIR
    / "component_disjoint_train_global_ids.npy"
).astype(np.int64)


X_train_for_norm = []

for gid in train_global_ids:

    X_source, _, idx = (
        decode_global_id(gid)
    )

    X_train_for_norm.append(
        X_source[idx]
    )

X_train_for_norm = np.asarray(
    X_train_for_norm,
    dtype=np.float32
)


train_mean = X_train_for_norm.mean(
    axis=(0, 1),
    keepdims=True
)

train_std = X_train_for_norm.std(
    axis=(0, 1),
    keepdims=True
)

train_std = np.maximum(
    train_std,
    1e-6
)

X_test = (
    (X_test - train_mean)
    / train_std
).astype(np.float32)


print("=" * 78)
print("PAIRED BOOTSTRAP â€” COMPONENT-DISJOINT TEST")
print("=" * 78)

print("Test samples:", len(X_test))
print("Device:", DEVICE)


# ============================================================
# LOAD CHECKPOINTS
# ============================================================

raw_model = SimpleCSIEncoder(
    repr_dim=REPR_DIM,
    num_classes=NUM_CLASSES
).to(DEVICE)

sil_model = SimpleCSIEncoder(
    repr_dim=REPR_DIM,
    num_classes=NUM_CLASSES
).to(DEVICE)


raw_ckpt = torch.load(
    EXP_DIR / "raw_ce_best.pt",
    map_location=DEVICE
)

sil_ckpt = torch.load(
    EXP_DIR / "sil_v2_best.pt",
    map_location=DEVICE
)


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
).unsqueeze(1)

loader = DataLoader(
    TensorDataset(X_tensor),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


raw_pred = []
sil_pred = []

with torch.no_grad():

    for (x,) in loader:

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        raw_logits, _ = raw_model(x)

        sil_logits, _ = sil_model(x)

        raw_pred.extend(
            raw_logits.argmax(
                dim=1
            ).cpu().numpy()
        )

        sil_pred.extend(
            sil_logits.argmax(
                dim=1
            ).cpu().numpy()
        )


raw_pred = np.asarray(
    raw_pred
)

sil_pred = np.asarray(
    sil_pred
)


# ============================================================
# OBSERVED ACCURACIES
# ============================================================

raw_correct = (
    raw_pred == y_test
)

sil_correct = (
    sil_pred == y_test
)

raw_acc = raw_correct.mean()
sil_acc = sil_correct.mean()

observed_diff = (
    sil_acc - raw_acc
) * 100.0


print()
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
    f"{observed_diff:+.2f} pp"
)


# ============================================================
# PAIRED BOOTSTRAP
# ============================================================

rng = np.random.default_rng(
    SEED
)

n = len(y_test)

bootstrap_diffs = np.empty(
    N_BOOT,
    dtype=np.float64
)

for b in range(N_BOOT):

    indices = rng.integers(
        0,
        n,
        size=n
    )

    raw_boot = (
        raw_correct[indices]
        .mean()
    )

    sil_boot = (
        sil_correct[indices]
        .mean()
    )

    bootstrap_diffs[b] = (
        sil_boot - raw_boot
    ) * 100.0


# ============================================================
# CONFIDENCE INTERVAL
# ============================================================

ci_low, ci_high = np.percentile(
    bootstrap_diffs,
    [2.5, 97.5]
)


# ============================================================
# TWO-SIDED BOOTSTRAP P-VALUE
# ============================================================

# For testing whether the paired difference
# is zero, calculate the fraction of bootstrap
# differences on the opposite side of zero.

if observed_diff > 0:

    p_value = 2.0 * min(
        np.mean(
            bootstrap_diffs <= 0
        ),
        np.mean(
            bootstrap_diffs >=
            2.0 * observed_diff
        )
    )

else:

    p_value = 2.0 * min(
        np.mean(
            bootstrap_diffs >= 0
        ),
        np.mean(
            bootstrap_diffs <=
            2.0 * observed_diff
        )
    )

p_value = min(
    1.0,
    float(p_value)
)


# ============================================================
# SAVE RESULTS
# ============================================================

result = {

    "protocol":
        "component_disjoint_test",

    "n_test":
        int(n),

    "seed":
        SEED,

    "n_bootstrap":
        N_BOOT,

    "raw_accuracy":
        float(raw_acc),

    "sil_v2_accuracy":
        float(sil_acc),

    "observed_difference_pp":
        float(observed_diff),

    "bootstrap_ci_95_pp": [
        float(ci_low),
        float(ci_high)
    ],

    "two_sided_bootstrap_p":
        p_value,

    "interpretation":
        (
            "SIL-v2 advantage is supported by "
            "the paired bootstrap if the 95% CI "
            "excludes zero."
        )
}


with open(
    EXP_DIR
    / "paired_bootstrap_component_disjoint.json",
    "w"
) as f:

    json.dump(
        result,
        f,
        indent=2
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 78)
print("BOOTSTRAP RESULT")
print("=" * 78)

print(
    f"Observed difference: "
    f"{observed_diff:+.2f} pp"
)

print(
    f"95% CI: "
    f"[{ci_low:+.2f}, "
    f"{ci_high:+.2f}] pp"
)

print(
    f"Two-sided bootstrap p: "
    f"{p_value:.6f}"
)

if ci_low > 0:

    print()
    print(
        "DECISION: SUPPORTED"
    )

    print(
        "The SIL-v2 advantage over Raw CE "
        "is supported by the paired "
        "bootstrap 95% CI."
    )

elif ci_high < 0:

    print()
    print(
        "DECISION: RAW CE SUPPORTED"
    )

else:

    print()
    print(
        "DECISION: INCONCLUSIVE"
    )

    print(
        "The 95% CI includes zero."
    )

print()
print(
    "Saved:"
)

print(
    EXP_DIR
    / "paired_bootstrap_component_disjoint.json"
)

print()
print("=" * 78)
