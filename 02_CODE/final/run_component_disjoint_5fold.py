from pathlib import Path
import json
import random
import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "data" / "raw"

OUT_DIR = (
    ROOT
    / "results"
    / "component_disjoint_5fold"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20261005

N_FOLDS = 5

BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 5e-4

MAX_EPOCHS = 150
PATIENCE = 20

NUM_CLASSES = 7
REPR_DIM = 128

STATIC_SIGMA = 0.4
ALPHA_PERTURBED_CE = 1.0
LAMBDA_INV = 0.15

LABEL_SMOOTHING = 0.10

# Exact overlap component definition used in the earlier audit.
BLOCK_LEN = 150

# The original 4973 windows are [250, 90].
WINDOW_LEN = 250


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 78)
print("5-FOLD COMPONENT-DISJOINT RAW CE vs SIL-v2")
print("=" * 78)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


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
# BUILD ONE GLOBAL 4973-WINDOW DATASET
# ============================================================

X_all = np.concatenate(
    [
        X_train_orig,
        X_val_orig,
        X_test_orig
    ],
    axis=0
)

y_all = np.concatenate(
    [
        y_train_orig,
        y_val_orig,
        y_test_orig
    ],
    axis=0
)

N = len(X_all)

assert N == 4973
assert X_all.shape[1:] == (250, 90)

print()
print("Global dataset:", X_all.shape)
print("Global labels :", y_all.shape)


# ============================================================
# EXACT TEMPORAL-OVERLAP COMPONENT CONSTRUCTION
#
# Two windows are connected if they contain an identical
# contiguous 150-row CSI block.
#
# We use union-find over exact block hashes.
# ============================================================

print()
print("=" * 78)
print("BUILDING EXACT 150-ROW OVERLAP COMPONENTS")
print("=" * 78)

parent = np.arange(N, dtype=np.int32)
rank = np.zeros(N, dtype=np.int8)


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a, b):

    ra = find(a)
    rb = find(b)

    if ra == rb:
        return

    if rank[ra] < rank[rb]:
        parent[ra] = rb

    elif rank[ra] > rank[rb]:
        parent[rb] = ra

    else:
        parent[rb] = ra
        rank[ra] += 1


# ------------------------------------------------------------
# Hash every exact 150-row contiguous block.
#
# A 250-row window contains:
# 250 - 150 + 1 = 101 blocks.
#
# Total:
# 4973 * 101 = 502273 block signatures.
# ------------------------------------------------------------

hash_owner = {}

total_blocks = N * (WINDOW_LEN - BLOCK_LEN + 1)

processed = 0

for i in range(N):

    x = X_all[i]

    for start in range(
        WINDOW_LEN - BLOCK_LEN + 1
    ):

        block = x[
            start:start + BLOCK_LEN
        ]

        # SHA1 is used only as an exact-content lookup key.
        # The actual equality is additionally checked before union.
        h = hashlib.sha1(
            block.tobytes()
        ).digest()

        key = (h, block.shape, block.dtype.str)

        if key in hash_owner:

            owner_i, owner_start = hash_owner[key]

            # Verify exact equality to eliminate theoretical hash collision.
            if np.array_equal(
                X_all[owner_i][
                    owner_start:
                    owner_start + BLOCK_LEN
                ],
                block
            ):

                union(i, owner_i)

        else:

            hash_owner[key] = (
                i,
                start
            )

        processed += 1

    if (
        (i + 1) % 250 == 0
        or i == N - 1
    ):

        print(
            f"Processed windows: "
            f"{i + 1}/{N}"
        )


# ============================================================
# COMPONENT IDS
# ============================================================

roots = np.array(
    [find(i) for i in range(N)],
    dtype=np.int32
)

unique_roots, component_ids = np.unique(
    roots,
    return_inverse=True
)

n_components = len(unique_roots)

print()
print("Number of components:", n_components)

assert n_components == 556, (
    f"Expected 556 components from the "
    f"previous audit, got {n_components}."
)


# ============================================================
# VERIFY LABEL PURITY
# ============================================================

component_labels = {}

for c in range(n_components):

    idx = np.where(
        component_ids == c
    )[0]

    labels = np.unique(
        y_all[idx]
    )

    component_labels[c] = int(
        labels[0]
    )

    if len(labels) != 1:

        raise RuntimeError(
            f"Component {c} contains "
            f"multiple labels: {labels}"
        )


print(
    "All components are label-pure."
)


# ============================================================
# SAVE COMPONENT ASSIGNMENT
# ============================================================

np.save(
    OUT_DIR / "global_component_ids.npy",
    component_ids
)

with open(
    OUT_DIR / "component_metadata.json",
    "w"
) as f:

    json.dump(
        {
            "n_components": int(n_components),
            "block_length": BLOCK_LEN,
            "window_length": WINDOW_LEN,
            "total_windows": int(N),
        },
        f,
        indent=2
    )


# ============================================================
# DATASET
# ============================================================

class CSIDataset(Dataset):

    def __init__(
        self,
        X,
        y
    ):

        self.X = torch.from_numpy(
            X.astype(np.float32)
        )

        self.y = torch.from_numpy(
            y.astype(np.int64)
        )


    def __len__(self):

        return len(self.y)


    def __getitem__(self, idx):

        # [T,K] -> [1,T,K]
        x = self.X[idx].unsqueeze(0)

        y = self.y[idx]

        return x, y


# ============================================================
# MODEL
# EXACT COMPONENT-DISJOINT MODEL
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
# STATIC DECOMPOSITION
# ============================================================

def decompose_static_dynamic(x):

    # x = [B,1,T,K]

    static = x.mean(
        dim=2,
        keepdim=True
    )

    dynamic = x - static

    return static, dynamic


# ============================================================
# STATIC PERTURBATION
# ============================================================

def perturb_static_component(x):

    static, dynamic = (
        decompose_static_dynamic(x)
    )

    B = x.shape[0]
    K = x.shape[-1]

    g = torch.randn(
        B,
        1,
        1,
        K,
        device=x.device
    ) * STATIC_SIGMA

    perturbed_static = (
        (1.0 + g)
        * static
    )

    x_tilde = (
        perturbed_static
        + dynamic
    )

    return x_tilde


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=LABEL_SMOOTHING
)


def sil_v2_loss(
    model,
    x,
    y
):

    logits, z = model(x)

    x_tilde = perturb_static_component(x)

    logits_tilde, z_tilde = model(
        x_tilde
    )

    clean_ce = criterion(
        logits,
        y
    )

    perturbed_ce = criterion(
        logits_tilde,
        y
    )

    z_norm = F.normalize(
        z,
        dim=1
    )

    z_tilde_norm = F.normalize(
        z_tilde,
        dim=1
    )

    cosine_similarity = (
        z_norm
        * z_tilde_norm
    ).sum(dim=1)

    invariance_loss = (
        1.0
        - cosine_similarity
    ).mean()

    loss = (
        clean_ce
        + ALPHA_PERTURBED_CE
        * perturbed_ce
        + LAMBDA_INV
        * invariance_loss
    )

    return loss


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader
):

    model.eval()

    correct = 0
    total = 0

    all_pred = []
    all_true = []

    for x, y in loader:

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        logits, _ = model(x)

        pred = logits.argmax(
            dim=1
        )

        correct += (
            pred == y
        ).sum().item()

        total += len(y)

        all_pred.extend(
            pred.cpu().numpy()
        )

        all_true.extend(
            y.cpu().numpy()
        )

    accuracy = (
        correct / total
    )

    all_pred = np.asarray(
        all_pred
    )

    all_true = np.asarray(
        all_true
    )

    f1s = []

    for c in range(
        NUM_CLASSES
    ):

        tp = np.sum(
            (all_pred == c)
            &
            (all_true == c)
        )

        fp = np.sum(
            (all_pred == c)
            &
            (all_true != c)
        )

        fn = np.sum(
            (all_pred != c)
            &
            (all_true == c)
        )

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        if (
            precision + recall
            > 0
        ):

            f1 = (
                2
                * precision
                * recall
                / (
                    precision
                    + recall
                )
            )

        else:

            f1 = 0.0

        f1s.append(f1)

    macro_f1 = float(
        np.mean(f1s)
    )

    return (
        float(accuracy),
        macro_f1
    )


# ============================================================
# MATERIALIZE DATA
# ============================================================

def get_data(indices):

    return (
        X_all[indices].copy(),
        y_all[indices].copy()
    )


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_one_model(
    Xtr,
    ytr,
    Xva,
    yva,
    Xte,
    yte,
    mode,
    fold
):

    # --------------------------------------------------------
    # FIT NORMALIZATION ON TRAINING FOLD ONLY
    # --------------------------------------------------------

    train_mean = Xtr.mean(
        axis=(0, 1),
        keepdims=True
    )

    train_std = Xtr.std(
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


    Xtr = normalize(Xtr)
    Xva = normalize(Xva)
    Xte = normalize(Xte)


    # --------------------------------------------------------
    # LOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        CSIDataset(Xtr, ytr),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        CSIDataset(Xva, yva),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        CSIDataset(Xte, yte),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = SimpleCSIEncoder(
        repr_dim=REPR_DIM,
        num_classes=NUM_CLASSES
    ).to(DEVICE)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS
    )


    best_val = -1.0
    best_epoch = -1
    patience_counter = 0

    history = []

    checkpoint_path = (
        OUT_DIR
        / f"fold_{fold}"
        / f"{mode}_best.pt"
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0
        n_seen = 0

        for x, y in train_loader:

            x = x.to(
                DEVICE,
                non_blocking=True
            )

            y = y.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            if mode == "raw_ce":

                logits, _ = model(x)

                loss = criterion(
                    logits,
                    y
                )

            elif mode == "sil_v2":

                loss = sil_v2_loss(
                    model,
                    x,
                    y
                )

            else:

                raise ValueError(
                    mode
                )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * len(y)
            )

            n_seen += len(y)

        scheduler.step()

        train_loss = (
            running_loss
            / n_seen
        )

        val_acc, val_f1 = evaluate(
            model,
            val_loader
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "lr":
                    optimizer.param_groups[
                        0
                    ]["lr"]
            }
        )

        print(
            f"Fold {fold} | "
            f"{mode:7s} | "
            f"epoch {epoch:03d} | "
            f"loss {train_loss:.4f} | "
            f"val {val_acc*100:.2f}% | "
            f"F1 {val_f1:.4f}"
        )

        if val_acc > best_val:

            best_val = val_acc
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model":
                        model.state_dict(),
                    "best_val_acc":
                        val_acc,
                    "best_val_f1":
                        val_f1,
                    "mode": mode,
                    "fold": fold,
                    "seed": SEED,
                },
                checkpoint_path
            )

        else:

            patience_counter += 1

        if (
            patience_counter
            >= PATIENCE
        ):

            print(
                f"Fold {fold} | "
                f"{mode}: early stop "
                f"at epoch {epoch}"
            )

            break


    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model"]
    )


    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    test_acc, test_f1 = evaluate(
        model,
        test_loader
    )


    return {
        "mode": mode,
        "fold": fold,
        "best_epoch":
            int(best_epoch),
        "best_val_acc":
            float(best_val),
        "best_val_f1":
            float(
                checkpoint[
                    "best_val_f1"
                ]
            ),
        "test_acc":
            float(test_acc),
        "test_f1":
            float(test_f1),
        "history": history,
    }


# ============================================================
# BUILD STRATIFIED GROUP FOLDS
# ============================================================

print()
print("=" * 78)
print("CREATING 5 STRATIFIED COMPONENT FOLDS")
print("=" * 78)

# Each component is a group.
# Every component is label-pure, so its class is well-defined.
#
# StratifiedGroupKFold attempts to preserve class proportions
# while never splitting a group between folds.

sgkf = StratifiedGroupKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED
)

fold_indices = []

for fold, (
    train_idx,
    test_idx
) in enumerate(
    sgkf.split(
        np.zeros(N),
        y_all,
        groups=component_ids
    ),
    start=1
):

    fold_indices.append(
        (
            train_idx,
            test_idx
        )
    )

    print(
        f"Fold {fold}: "
        f"trainval={len(train_idx)}, "
        f"test={len(test_idx)}"
    )

    print(
        "  test classes:",
        np.bincount(
            y_all[test_idx],
            minlength=NUM_CLASSES
        ).tolist()
    )


# ============================================================
# CHECK EVERY SAMPLE APPEARS IN TEST EXACTLY ONCE
# ============================================================

test_membership = np.zeros(
    N,
    dtype=np.int32
)

for _, test_idx in fold_indices:

    test_membership[test_idx] += 1

if not np.all(
    test_membership == 1
):

    raise RuntimeError(
        "5-fold test partition is invalid: "
        "some samples are missing or "
        "appear in multiple test folds."
    )


# ============================================================
# RUN CV
#
# For each outer fold:
#
#   test = outer fold
#
#   remaining 4 folds:
#       one fold = validation
#       three folds = training
#
# Validation is also component-disjoint.
#
# We rotate the validation fold across the four
# non-test folds.
# ============================================================

all_results = []

for outer_fold in range(
    N_FOLDS
):

    print()
    print("=" * 78)
    print(
        f"OUTER FOLD {outer_fold + 1}/"
        f"{N_FOLDS}"
    )
    print("=" * 78)


    outer_trainval = fold_indices[
        outer_fold
    ][0]

    outer_test = fold_indices[
        outer_fold
    ][1]


    # --------------------------------------------------------
    # The remaining 4 folds become train/validation.
    #
    # Rotate validation deterministically.
    # --------------------------------------------------------

    remaining_folds = [
        f
        for f in range(N_FOLDS)
        if f != outer_fold
    ]

    validation_fold = remaining_folds[
        outer_fold % len(
            remaining_folds
        )
    ]

    val_idx = fold_indices[
        validation_fold
    ][1]


    train_mask = np.ones(
        N,
        dtype=bool
    )

    train_mask[val_idx] = False
    train_mask[outer_test] = False

    train_idx = np.where(
        train_mask
    )[0]


    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    train_components = set(
        component_ids[
            train_idx
        ].tolist()
    )

    val_components = set(
        component_ids[
            val_idx
        ].tolist()
    )

    test_components = set(
        component_ids[
            outer_test
        ].tolist()
    )


    if (
        train_components
        & val_components
    ):

        raise RuntimeError(
            "Component overlap between "
            "train and validation."
        )


    if (
        train_components
        & test_components
    ):

        raise RuntimeError(
            "Component overlap between "
            "train and test."
        )


    if (
        val_components
        & test_components
    ):

        raise RuntimeError(
            "Component overlap between "
            "validation and test."
        )


    print()
    print(
        "Train:",
        len(train_idx),
        "samples"
    )

    print(
        "Validation:",
        len(val_idx),
        "samples"
    )

    print(
        "Test:",
        len(outer_test),
        "samples"
    )

    print(
        "Train classes:",
        np.bincount(
            y_all[train_idx],
            minlength=NUM_CLASSES
        ).tolist()
    )

    print(
        "Validation classes:",
        np.bincount(
            y_all[val_idx],
            minlength=NUM_CLASSES
        ).tolist()
    )

    print(
        "Test classes:",
        np.bincount(
            y_all[outer_test],
            minlength=NUM_CLASSES
        ).tolist()
    )


    # --------------------------------------------------------
    # Materialize
    # --------------------------------------------------------

    Xtr, ytr = get_data(
        train_idx
    )

    Xva, yva = get_data(
        val_idx
    )

    Xte, yte = get_data(
        outer_test
    )


    # --------------------------------------------------------
    # RAW CE
    # --------------------------------------------------------

    print()
    print(
        f"FOLD {outer_fold + 1}: "
        "RAW CE"
    )

    # Reset RNG identically for both models.
    torch.manual_seed(
        SEED + outer_fold
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            SEED + outer_fold
        )

    raw_result = train_one_model(
        Xtr,
        ytr,
        Xva,
        yva,
        Xte,
        yte,
        "raw_ce",
        outer_fold + 1
    )


    # --------------------------------------------------------
    # SIL-v2
    # --------------------------------------------------------

    print()
    print(
        f"FOLD {outer_fold + 1}: "
        "SIL-v2"
    )

    torch.manual_seed(
        SEED + outer_fold
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            SEED + outer_fold
        )

    sil_result = train_one_model(
        Xtr,
        ytr,
        Xva,
        yva,
        Xte,
        yte,
        "sil_v2",
        outer_fold + 1
    )


    # --------------------------------------------------------
    # Fold summary
    # --------------------------------------------------------

    fold_summary = {

        "fold":
            outer_fold + 1,

        "train_samples":
            int(len(train_idx)),

        "val_samples":
            int(len(val_idx)),

        "test_samples":
            int(len(outer_test)),

        "raw_ce":
            raw_result,

        "sil_v2":
            sil_result,

        "accuracy_difference_pp":
            (
                sil_result["test_acc"]
                -
                raw_result["test_acc"]
            ) * 100.0,

        "f1_difference_pp":
            (
                sil_result["test_f1"]
                -
                raw_result["test_f1"]
            ) * 100.0,
    }

    all_results.append(
        fold_summary
    )

    with open(
        OUT_DIR
        / f"fold_{outer_fold + 1}"
        / "fold_summary.json",
        "w"
    ) as f:

        json.dump(
            fold_summary,
            f,
            indent=2
        )


    print()
    print(
        f"FOLD {outer_fold + 1} RESULT"
    )

    print(
        f"Raw CE : "
        f"{raw_result['test_acc']*100:.2f}%"
    )

    print(
        f"SIL-v2 : "
        f"{sil_result['test_acc']*100:.2f}%"
    )

    print(
        f"Delta  : "
        f"{fold_summary['accuracy_difference_pp']:+.2f} pp"
    )


# ============================================================
# AGGREGATE
# ============================================================

raw_acc = np.array(
    [
        r["raw_ce"]["test_acc"]
        for r in all_results
    ]
)

sil_acc = np.array(
    [
        r["sil_v2"]["test_acc"]
        for r in all_results
    ]
)

raw_f1 = np.array(
    [
        r["raw_ce"]["test_f1"]
        for r in all_results
    ]
)

sil_f1 = np.array(
    [
        r["sil_v2"]["test_f1"]
        for r in all_results
    ]
)

acc_delta = (
    sil_acc - raw_acc
)

f1_delta = (
    sil_f1 - raw_f1
)


summary = {

    "protocol":
        "5-fold component-disjoint "
        "stratified group CV",

    "n_folds":
        N_FOLDS,

    "component_definition":
        "connected components formed "
        "by exact shared contiguous "
        "150-row CSI blocks",

    "normalization":
        "train-fold-only per-subcarrier "
        "mean/std",

    "raw_ce": {

        "fold_accuracy":
            raw_acc.tolist(),

        "mean_accuracy":
            float(raw_acc.mean()),

        "std_accuracy":
            float(raw_acc.std(
                ddof=1
            )),

        "mean_f1":
            float(raw_f1.mean()),

        "std_f1":
            float(raw_f1.std(
                ddof=1
            )),
    },

    "sil_v2": {

        "fold_accuracy":
            sil_acc.tolist(),

        "mean_accuracy":
            float(sil_acc.mean()),

        "std_accuracy":
            float(sil_acc.std(
                ddof=1
            )),

        "mean_f1":
            float(sil_f1.mean()),

        "std_f1":
            float(sil_f1.std(
                ddof=1
            )),
    },

    "sil_minus_raw": {

        "fold_accuracy_difference_pp":
            (
                acc_delta * 100.0
            ).tolist(),

        "mean_difference_pp":
            float(
                acc_delta.mean()
                * 100.0
            ),

        "std_difference_pp":
            float(
                acc_delta.std(
                    ddof=1
                )
                * 100.0
            ),

        "fold_f1_difference_pp":
            (
                f1_delta * 100.0
            ).tolist(),

        "mean_f1_difference_pp":
            float(
                f1_delta.mean()
                * 100.0
            ),

        "std_f1_difference_pp":
            float(
                f1_delta.std(
                    ddof=1
                )
                * 100.0
            ),
    },

    "fold_results":
        all_results
}


with open(
    OUT_DIR
    / "5fold_summary.json",
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# FINAL PRINT
# ============================================================

print()
print("=" * 78)
print("5-FOLD COMPONENT-DISJOINT CV â€” FINAL")
print("=" * 78)

print()

print(
    "RAW CE"
)

print(
    f"Accuracy: "
    f"{raw_acc.mean()*100:.2f} "
    f"+/- "
    f"{raw_acc.std(ddof=1)*100:.2f}%"
)

print(
    f"Macro-F1: "
    f"{raw_f1.mean():.4f} "
    f"+/- "
    f"{raw_f1.std(ddof=1):.4f}"
)

print()

print(
    "SIL-v2"
)

print(
    f"Accuracy: "
    f"{sil_acc.mean()*100:.2f} "
    f"+/- "
    f"{sil_acc.std(ddof=1)*100:.2f}%"
)

print(
    f"Macro-F1: "
    f"{sil_f1.mean():.4f} "
    f"+/- "
    f"{sil_f1.std(ddof=1):.4f}"
)

print()

print(
    "SIL-v2 - Raw CE"
)

print(
    f"Accuracy difference: "
    f"{acc_delta.mean()*100:+.2f} "
    f"+/- "
    f"{acc_delta.std(ddof=1)*100:.2f} pp"
)

print(
    f"F1 difference: "
    f"{f1_delta.mean()*100:+.2f} "
    f"+/- "
    f"{f1_delta.std(ddof=1)*100:.2f} pp"
)

print()

print(
    "Saved:"
)

print(
    OUT_DIR / "5fold_summary.json"
)

print()

print("=" * 78)
print("DONE")
print("=" * 78)
