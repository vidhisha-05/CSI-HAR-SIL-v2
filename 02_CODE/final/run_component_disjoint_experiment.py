from pathlib import Path
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


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

OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

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
print("COMPONENT-DISJOINT RAW CE vs SIL-v2")
print("=" * 78)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


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
# ORIGINAL GLOBAL-ID OFFSETS
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
# LOAD COMPONENT-DISJOINT SPLIT
# ============================================================

new_global_ids = {}

for split in ["train", "val", "test"]:

    path = (
        SPLIT_DIR
        / f"component_disjoint_{split}_global_ids.npy"
    )

    new_global_ids[split] = np.load(
        path
    ).astype(np.int64)

    print(
        f"{split}: "
        f"{len(new_global_ids[split])} windows"
    )


# ============================================================
# MATERIALIZE NEW SPLIT
# ============================================================

def materialize_split(global_ids):

    X_list = []
    y_list = []

    for gid in global_ids:

        source_split, local_idx = (
            decode_global_id(gid)
        )

        X_list.append(
            ORIGINAL_X[source_split][local_idx]
        )

        y_list.append(
            ORIGINAL_Y[source_split][local_idx]
        )

    X = np.asarray(
        X_list,
        dtype=np.float32
    )

    y = np.asarray(
        y_list,
        dtype=np.int64
    )

    return X, y


X_train, y_train = materialize_split(
    new_global_ids["train"]
)

X_val, y_val = materialize_split(
    new_global_ids["val"]
)

X_test, y_test = materialize_split(
    new_global_ids["test"]
)


print()
print("New split shapes:")
print("X_train:", X_train.shape)
print("X_val:  ", X_val.shape)
print("X_test: ", X_test.shape)


# ============================================================
# CLASS COUNTS
# ============================================================

def print_class_counts(name, y):

    counts = np.bincount(
        y,
        minlength=NUM_CLASSES
    )

    print(
        f"{name} class counts:",
        counts.tolist()
    )


print()
print_class_counts("Train", y_train)
print_class_counts("Val", y_val)
print_class_counts("Test", y_test)


# ============================================================
# TRAIN-FITTED PER-SUBCARRIER NORMALIZATION
# ============================================================

# Same normalization used by clean SIL-v2:
# mean/std over train samples and time axis,
# separately for each subcarrier.

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


X_train = normalize(X_train)
X_val = normalize(X_val)
X_test = normalize(X_test)


print()
print(
    "Normalized train mean:",
    float(X_train.mean())
)

print(
    "Normalized train std:",
    float(X_train.std())
)


# ============================================================
# DATASET
# ============================================================

class CSIDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):

        return len(self.y)

    def __getitem__(self, idx):

        # [T,K] -> [1,T,K]

        x = self.X[idx].unsqueeze(0)

        y = self.y[idx]

        return x, y


train_ds = CSIDataset(
    X_train,
    y_train
)

val_ds = CSIDataset(
    X_val,
    y_val
)

test_ds = CSIDataset(
    X_test,
    y_test
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# MODEL
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

    # x: [B,1,T,K]

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

    # Perturb the stable temporal component.
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

    return (
        loss,
        clean_ce.detach(),
        perturbed_ce.detach(),
        invariance_loss.detach()
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, loader):

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

    for c in range(NUM_CLASSES):

        tp = np.sum(
            (all_pred == c)
            & (all_true == c)
        )

        fp = np.sum(
            (all_pred == c)
            & (all_true != c)
        )

        fn = np.sum(
            (all_pred != c)
            & (all_true == c)
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

        if precision + recall > 0:

            f1 = (
                2
                * precision
                * recall
                / (precision + recall)
            )

        else:

            f1 = 0.0

        f1s.append(f1)

    macro_f1 = float(
        np.mean(f1s)
    )

    return (
        accuracy,
        macro_f1
    )


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(model, mode):

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
        / f"{mode}_best.pt"
    )

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

                (
                    loss,
                    clean_ce,
                    perturbed_ce,
                    inv_loss
                ) = sil_v2_loss(
                    model,
                    x,
                    y
                )

            else:

                raise ValueError(mode)

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
                    optimizer.param_groups[0]["lr"]
            }
        )

        print(
            f"{mode:8s} | "
            f"epoch {epoch:03d} | "
            f"loss {train_loss:.4f} | "
            f"val acc {val_acc*100:.2f}% | "
            f"val F1 {val_f1:.4f}"
        )

        if val_acc > best_val:

            best_val = val_acc
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_val_acc": val_acc,
                    "best_val_f1": val_f1,
                    "mode": mode,
                    "seed": SEED,
                    "config": {
                        "batch_size": BATCH_SIZE,
                        "lr": LR,
                        "weight_decay": WEIGHT_DECAY,
                        "max_epochs": MAX_EPOCHS,
                        "patience": PATIENCE,
                        "static_sigma": STATIC_SIGMA,
                        "alpha_perturbed_ce":
                            ALPHA_PERTURBED_CE,
                        "lambda_inv":
                            LAMBDA_INV,
                        "label_smoothing":
                            LABEL_SMOOTHING,
                    }
                },
                checkpoint_path
            )

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                f"{mode}: early stopping "
                f"at epoch {epoch}"
            )

            break

    # Load best checkpoint.

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    test_acc, test_f1 = evaluate(
        model,
        test_loader
    )

    result = {
        "mode": mode,
        "best_epoch": int(best_epoch),
        "best_val_acc":
            float(checkpoint["best_val_acc"]),
        "best_val_f1":
            float(checkpoint["best_val_f1"]),
        "test_acc":
            float(test_acc),
        "test_f1":
            float(test_f1),
        "history": history
    }

    with open(
        OUT_DIR
        / f"{mode}_result.json",
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    return result


# ============================================================
# MODEL 1: RAW CSI + CE
# ============================================================

print()
print("=" * 78)
print("MODEL 1: RAW CSI + CE")
print("=" * 78)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

raw_model = SimpleCSIEncoder(
    repr_dim=REPR_DIM,
    num_classes=NUM_CLASSES
).to(DEVICE)

raw_result = train_model(
    raw_model,
    "raw_ce"
)


# ============================================================
# MODEL 2: RAW CSI + SIL-v2
# ============================================================

print()
print("=" * 78)
print("MODEL 2: RAW CSI + SIL-v2")
print("=" * 78)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

sil_model = SimpleCSIEncoder(
    repr_dim=REPR_DIM,
    num_classes=NUM_CLASSES
).to(DEVICE)

sil_result = train_model(
    sil_model,
    "sil_v2"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = {
    "protocol":
        "component_disjoint_exact_150_row_overlap",

    "split_sizes": {
        "train": len(X_train),
        "val": len(X_val),
        "test": len(X_test)
    },

    "seed": SEED,

    "raw_ce": raw_result,

    "sil_v2": sil_result,

    "difference_test_accuracy_pp":
        (
            sil_result["test_acc"]
            - raw_result["test_acc"]
        ) * 100.0,

    "difference_test_f1_pp":
        (
            sil_result["test_f1"]
            - raw_result["test_f1"]
        ) * 100.0
}

with open(
    OUT_DIR / "experiment_summary.json",
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )

print()
print("=" * 78)
print("FINAL SUMMARY")
print("=" * 78)

print(
    f"Raw CE  : "
    f"{raw_result['test_acc']*100:.2f}% "
    f"test accuracy, "
    f"{raw_result['test_f1']:.4f} F1"
)

print(
    f"SIL-v2  : "
    f"{sil_result['test_acc']*100:.2f}% "
    f"test accuracy, "
    f"{sil_result['test_f1']:.4f} F1"
)

print(
    f"SIL - Raw CE: "
    f"{summary['difference_test_accuracy_pp']:+.2f} pp"
)

print()
print("Results saved to:")
print(OUT_DIR)

print()
print("=" * 78)
print("DONE")
print("=" * 78)
