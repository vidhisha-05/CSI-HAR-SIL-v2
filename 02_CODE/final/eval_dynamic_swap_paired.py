from pathlib import Path
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# PAIRED DYNAMIC-SWAP AUDIT
# RAW CE vs GENERIC CONSISTENCY vs SIL-v2
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "data" / "raw"

TRAIN_X = DATA_DIR / "data" / "X_train.npy"
TEST_X = DATA_DIR / "data" / "X_test.npy"
TEST_Y = DATA_DIR / "label" / "y_test.npy"

CE_CKPT = (
    ROOT / "results" / "clean_sil_v2" / "raw_ce_best.pt"
)

GENERIC_CKPT = (
    ROOT / "results" / "generic_consistency_control"
    / "raw_generic_consistency_best.pt"
)

SIL_CKPT = (
    ROOT / "results" / "clean_sil_v2"
    / "raw_sil_v2_best.pt"
)

OUT_JSON = (
    ROOT / "results" / "generic_consistency_control"
    / "dynamic_swap_paired_audit.json"
)

SEED = 42
BATCH_SIZE = 64
NUM_CLASSES = 7
REPR_DIM = 128
N_BOOT = 20000


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# MODEL
# ============================================================

class SimpleCSIEncoder(nn.Module):

    def __init__(self, repr_dim=128):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1, 32,
                kernel_size=(7, 5),
                padding=(3, 2),
                bias=False
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64,
                kernel_size=(5, 3),
                padding=(2, 1),
                bias=False
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, repr_dim)
        )

    def forward(self, x):

        x = x.unsqueeze(1)

        h = self.features(x)

        return self.projection(h)


class Classifier(nn.Module):

    def __init__(
        self,
        repr_dim=128,
        num_classes=7
    ):

        super().__init__()

        self.fc = nn.Linear(
            repr_dim,
            num_classes
        )

    def forward(self, z):

        return self.fc(z)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(checkpoint_path, device):

    encoder = SimpleCSIEncoder(
        REPR_DIM
    ).to(device)

    classifier = Classifier(
        REPR_DIM,
        NUM_CLASSES
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    encoder.load_state_dict(
        checkpoint["encoder"]
    )

    classifier.load_state_dict(
        checkpoint["classifier"]
    )

    encoder.eval()
    classifier.eval()

    return encoder, classifier


# ============================================================
# DECOMPOSITION
# ============================================================

def decompose_numpy(x):

    static = x.mean(
        axis=1,
        keepdims=True
    )

    dynamic = x - static

    return static, dynamic


# ============================================================
# DYNAMIC SWAP
#
# X = S + D
#
# New sample:
# X' = S_original + D_donor
#
# Donor is always from a different activity class.
# ============================================================

def dynamic_swap_different_class(
    x,
    y,
    rng
):

    static, dynamic = decompose_numpy(x)

    n = len(y)

    donor_indices = np.empty(
        n,
        dtype=np.int64
    )

    for i in range(n):

        candidates = np.flatnonzero(
            y != y[i]
        )

        donor_indices[i] = rng.choice(
            candidates
        )

    donor_dynamic = dynamic[
        donor_indices
    ]

    return (
        static + donor_dynamic
    ).astype(np.float32)


# ============================================================
# PREDICTIONS
# ============================================================

@torch.no_grad()
def predictions(
    encoder,
    classifier,
    X,
    device
):

    dataset = TensorDataset(
        torch.from_numpy(X)
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    preds = []

    for (x_batch,) in loader:

        x_batch = x_batch.to(device)

        z = encoder(x_batch)

        logits = classifier(z)

        preds.append(
            logits.argmax(dim=1)
            .cpu()
            .numpy()
        )

    return np.concatenate(preds)


# ============================================================
# PAIRED BOOTSTRAP
# ============================================================

def paired_bootstrap(
    y_true,
    pred_a,
    pred_b,
    rng,
    n_boot=20000
):

    correct_a = (
        pred_a == y_true
    ).astype(np.float64)

    correct_b = (
        pred_b == y_true
    ).astype(np.float64)

    observed_diff = (
        correct_a.mean()
        -
        correct_b.mean()
    )

    n = len(y_true)

    diffs = np.empty(
        n_boot,
        dtype=np.float64
    )

    chunk = 1000

    start = 0

    while start < n_boot:

        m = min(
            chunk,
            n_boot - start
        )

        idx = rng.integers(
            0,
            n,
            size=(m, n)
        )

        boot_a = (
            correct_a[idx]
            .mean(axis=1)
        )

        boot_b = (
            correct_b[idx]
            .mean(axis=1)
        )

        diffs[start:start + m] = (
            boot_a - boot_b
        )

        start += m

    ci_low, ci_high = np.percentile(
        diffs,
        [2.5, 97.5]
    )

    p_value = 2.0 * min(
        np.mean(diffs <= 0),
        np.mean(diffs >= 0)
    )

    p_value = min(
        1.0,
        float(p_value)
    )

    return {
        "accuracy_a":
            float(correct_a.mean() * 100),

        "accuracy_b":
            float(correct_b.mean() * 100),

        "difference_pp":
            float(observed_diff * 100),

        "ci95_low_pp":
            float(ci_low * 100),

        "ci95_high_pp":
            float(ci_high * 100),

        "bootstrap_p_value_two_sided":
            p_value,

        "n_bootstrap":
            n_boot
    }


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 78)
    print("PAIRED DYNAMIC-SWAP AUDIT")
    print("=" * 78)

    print("Device:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    X_train = np.load(
        TRAIN_X
    ).astype(np.float32)

    X_test = np.load(
        TEST_X
    ).astype(np.float32)

    y_test = np.load(
        TEST_Y
    ).astype(np.int64)

    print(
        "Test samples:",
        len(y_test)
    )

    # --------------------------------------------------------
    # SAME TRAIN-FITTED NORMALIZATION
    # --------------------------------------------------------

    mean = X_train.mean(
        axis=(0, 1),
        keepdims=True
    )

    std = X_train.std(
        axis=(0, 1),
        keepdims=True
    )

    std = np.maximum(
        std,
        1e-6
    )

    X_test = (
        (X_test - mean) / std
    ).astype(np.float32)

    # --------------------------------------------------------
    # CREATE DYNAMIC SWAP
    # --------------------------------------------------------

    X_swap = dynamic_swap_different_class(
        X_test,
        y_test,
        np.random.default_rng(107)
    )

    # --------------------------------------------------------
    # VERIFY STATIC PRESERVATION
    # --------------------------------------------------------

    original_static, _ = (
        decompose_numpy(X_test)
    )

    swapped_static, _ = (
        decompose_numpy(X_swap)
    )

    static_error = np.max(
        np.abs(
            swapped_static
            -
            original_static
        )
    )

    print(
        "Dynamic-swap static preservation "
        "max error:",
        static_error
    )

    assert static_error < 1e-4

    # --------------------------------------------------------
    # LOAD THREE MODELS
    # --------------------------------------------------------

    ce_encoder, ce_classifier = load_model(
        CE_CKPT,
        device
    )

    generic_encoder, generic_classifier = load_model(
        GENERIC_CKPT,
        device
    )

    sil_encoder, sil_classifier = load_model(
        SIL_CKPT,
        device
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    print()
    print("Generating paired predictions...")

    pred_ce = predictions(
        ce_encoder,
        ce_classifier,
        X_swap,
        device
    )

    pred_generic = predictions(
        generic_encoder,
        generic_classifier,
        X_swap,
        device
    )

    pred_sil = predictions(
        sil_encoder,
        sil_classifier,
        X_swap,
        device
    )

    acc_ce = (
        np.mean(pred_ce == y_test)
        * 100
    )

    acc_generic = (
        np.mean(pred_generic == y_test)
        * 100
    )

    acc_sil = (
        np.mean(pred_sil == y_test)
        * 100
    )

    print(
        f"RAW CE              : {acc_ce:.2f}%"
    )

    print(
        f"GENERIC CONSISTENCY : {acc_generic:.2f}%"
    )

    print(
        f"SIL-v2              : {acc_sil:.2f}%"
    )

    # --------------------------------------------------------
    # BOOTSTRAP
    # --------------------------------------------------------

    rng = np.random.default_rng(
        20261005
    )

    comparisons = {}

    pairs = {

        "SIL_v2_minus_GENERIC":
            (pred_sil, pred_generic),

        "SIL_v2_minus_RAW_CE":
            (pred_sil, pred_ce),

        "GENERIC_minus_RAW_CE":
            (pred_generic, pred_ce)
    }

    print()
    print("=" * 78)
    print("PAIRED BOOTSTRAP RESULTS")
    print("=" * 78)

    for name, (pred_a, pred_b) in pairs.items():

        result = paired_bootstrap(
            y_test,
            pred_a,
            pred_b,
            rng,
            N_BOOT
        )

        comparisons[name] = result

        print()
        print(name)

        print(
            f"Observed difference: "
            f"{result['difference_pp']:+.2f} pp"
        )

        print(
            f"95% CI: "
            f"[{result['ci95_low_pp']:+.2f}, "
            f"{result['ci95_high_pp']:+.2f}] pp"
        )

        print(
            f"Bootstrap p-value: "
            f"{result['bootstrap_p_value_two_sided']:.6f}"
        )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    primary = comparisons[
        "SIL_v2_minus_GENERIC"
    ]

    if primary["ci95_low_pp"] > 0:

        decision = (
            "SIL-v2 has a statistically supported "
            "advantage over generic consistency "
            "on dynamic-swap."
        )

    elif primary["ci95_high_pp"] < 0:

        decision = (
            "Generic consistency has a statistically "
            "supported advantage over SIL-v2 "
            "on dynamic-swap."
        )

    else:

        decision = (
            "The SIL-v2 versus generic difference "
            "on dynamic-swap is inconclusive."
        )

    print()
    print("=" * 78)
    print("PRIMARY DECISION")
    print("=" * 78)
    print(decision)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output = {

        "seed": SEED,

        "bootstrap_seed":
            20261005,

        "bootstrap_replicates":
            N_BOOT,

        "test_samples":
            int(len(y_test)),

        "condition":
            "dynamic_swap_different_class",

        "normalization":
            "train_fitted_per_subcarrier_mean_std",

        "dynamic_swap_static_max_error":
            float(static_error),

        "accuracies": {

            "raw_ce":
                float(acc_ce),

            "generic_consistency":
                float(acc_generic),

            "sil_v2":
                float(acc_sil)
        },

        "comparisons":
            comparisons,

        "primary_decision":
            decision
    }

    OUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUT_JSON,
        "w"
    ) as f:

        json.dump(
            output,
            f,
            indent=2
        )

    print()
    print("Saved:")
    print(OUT_JSON)


if __name__ == "__main__":
    main()
