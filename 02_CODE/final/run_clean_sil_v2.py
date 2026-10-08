# ============================================================
# CLEAN SIL v2
# TASK-PRESERVING STATIC-COMPONENT INVARIANCE
# ============================================================
#
# Method:
#
# X = S + D
#
# S = temporal mean of normalized CSI
# D = X - S
#
# S_tilde = (1 + g) * S
# X_tilde = S_tilde + D
#
# Objective:
#
# L = CE(clean)
#   + alpha * CE(perturbed)
#   + lambda_inv * cosine_invariance
#
# The architecture is deliberately simple.
#
# IMPORTANT:
# This is a NEW experiment.
# It does NOT overwrite the previous clean SIL results.
# ============================================================

from pathlib import Path
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "data" / "raw"

TRAIN_X = DATA_DIR / "data" / "X_train.npy"
VAL_X   = DATA_DIR / "data" / "X_val.npy"
TEST_X  = DATA_DIR / "data" / "X_test.npy"

TRAIN_Y = DATA_DIR / "label" / "y_train.npy"
VAL_Y   = DATA_DIR / "label" / "y_val.npy"
TEST_Y  = DATA_DIR / "label" / "y_test.npy"

OUT_DIR = ROOT / "results" / "clean_sil_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


SEED = 42

BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 5e-4

MAX_EPOCHS = 150
PATIENCE = 20

NUM_CLASSES = 7
REPR_DIM = 128

# ------------------------------------------------------------
# METHOD HYPERPARAMETERS
# ------------------------------------------------------------

STATIC_GAIN_SIGMA = 0.4

# Weight of perturbed-view classification.
ALPHA_PERT_CE = 1.0

# Weight of representation invariance.
LAMBDA_INV = 0.15

LABEL_SMOOTHING = 0.10


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("LOADING RAW UT-HAR")
    print("=" * 70)

    X_train = np.load(TRAIN_X).astype(np.float32)
    X_val   = np.load(VAL_X).astype(np.float32)
    X_test  = np.load(TEST_X).astype(np.float32)

    y_train = np.load(TRAIN_Y).astype(np.int64)
    y_val   = np.load(VAL_Y).astype(np.int64)
    y_test  = np.load(TEST_Y).astype(np.int64)

    print("X_train:", X_train.shape)
    print("X_val:  ", X_val.shape)
    print("X_test: ", X_test.shape)

    print("y_train:", y_train.shape)
    print("y_val:  ", y_val.shape)
    print("y_test: ", y_test.shape)

    assert X_train.shape == (3977, 250, 90)
    assert X_val.shape == (496, 250, 90)
    assert X_test.shape == (500, 250, 90)

    return (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    )


# ============================================================
# TRAIN-FITTED NORMALIZATION
# ============================================================

def fit_global_normalization(X_train):

    mean = X_train.mean(
        axis=(0, 1),
        keepdims=True
    )

    std = X_train.std(
        axis=(0, 1),
        keepdims=True
    )

    std = np.maximum(std, 1e-6)

    return (
        mean.astype(np.float32),
        std.astype(np.float32)
    )


def normalize(X, mean, std):

    return ((X - mean) / std).astype(np.float32)


# ============================================================
# SIMPLE ENCODER
# ============================================================

class SimpleCSIEncoder(nn.Module):

    def __init__(self, repr_dim=128):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=(7, 5),
                padding=(3, 2),
                bias=False
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=(2, 2)
            ),

            nn.Conv2d(
                32,
                64,
                kernel_size=(5, 3),
                padding=(2, 1),
                bias=False
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=(2, 2)
            ),

            nn.Conv2d(
                64,
                128,
                kernel_size=(3, 3),
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.projection = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128,
                repr_dim
            )
        )


    def forward(self, x):

        # x: (B,T,K)

        x = x.unsqueeze(1)

        h = self.features(x)

        z = self.projection(h)

        return z


# ============================================================
# CLASSIFIER
# ============================================================

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
# STATIC / TEMPORAL-RESIDUAL DECOMPOSITION
# ============================================================

def static_dynamic_decomposition(x):

    """
    x:
        (B,T,K)

    Stable temporal component:
        S(k) = mean_t X(t,k)

    Temporal residual:
        D(t,k) = X(t,k) - S(k)
    """

    static = x.mean(
        dim=1,
        keepdim=True
    )

    dynamic = x - static

    return static, dynamic


# ============================================================
# STATIC-COMPONENT PERTURBATION
# ============================================================

def static_perturbation(
    x,
    sigma=0.4
):

    """
    Modify only the temporally stable component.

    X = S + D

    S_tilde = (1 + g) S

    X_tilde = S_tilde + D

    g ~ N(0, sigma^2)
    """

    static, dynamic = (
        static_dynamic_decomposition(x)
    )

    B, _, K = static.shape

    g = torch.randn(
        B,
        1,
        K,
        device=x.device,
        dtype=x.dtype
    ) * sigma

    perturbed_static = (
        (1.0 + g) * static
    )

    x_tilde = (
        perturbed_static + dynamic
    )

    return x_tilde


# ============================================================
# INVARIANCE LOSS
# ============================================================

def cosine_invariance_loss(
    z,
    z_tilde
):

    z = F.normalize(
        z,
        dim=1
    )

    z_tilde = F.normalize(
        z_tilde,
        dim=1
    )

    similarity = (
        z * z_tilde
    ).sum(dim=1)

    return (
        1.0 - similarity
    ).mean()


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    encoder,
    classifier,
    loader,
    device
):

    encoder.eval()
    classifier.eval()

    correct = 0
    total = 0

    loss_sum = 0.0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            z = encoder(x)

            logits = classifier(z)

            loss = F.cross_entropy(
                logits,
                y
            )

            pred = logits.argmax(
                dim=1
            )

            correct += (
                pred == y
            ).sum().item()

            total += y.numel()

            loss_sum += (
                loss.item() * y.size(0)
            )

    return {
        "loss": loss_sum / total,
        "acc": correct / total
    }


# ============================================================
# TRAINING
# ============================================================

def train_experiment(
    name,
    train_loader,
    val_loader,
    test_loader,
    device,
    use_sil
):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    encoder = SimpleCSIEncoder(
        repr_dim=REPR_DIM
    ).to(device)

    classifier = Classifier(
        repr_dim=REPR_DIM,
        num_classes=NUM_CLASSES
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters())
        +
        list(classifier.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=7,
            min_lr=1e-6
        )
    )

    best_val = -1.0
    best_epoch = -1
    patience_counter = 0

    history = []

    save_path = (
        OUT_DIR
        /
        f"{name.lower().replace(' ', '_')}_best.pt"
    )

    for epoch in range(MAX_EPOCHS):

        encoder.train()
        classifier.train()

        total_loss_sum = 0.0

        clean_cls_sum = 0.0
        pert_cls_sum = 0.0
        inv_loss_sum = 0.0

        correct = 0
        total = 0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------
            # CLEAN VIEW
            # ------------------------------------------------

            z = encoder(x)

            logits = classifier(z)

            clean_cls_loss = F.cross_entropy(
                logits,
                y,
                label_smoothing=LABEL_SMOOTHING
            )

            # ------------------------------------------------
            # PERTURBED VIEW
            # ------------------------------------------------

            if use_sil:

                x_tilde = static_perturbation(
                    x,
                    sigma=STATIC_GAIN_SIGMA
                )

                z_tilde = encoder(
                    x_tilde
                )

                logits_tilde = classifier(
                    z_tilde
                )

                pert_cls_loss = F.cross_entropy(
                    logits_tilde,
                    y,
                    label_smoothing=LABEL_SMOOTHING
                )

                inv_loss = (
                    cosine_invariance_loss(
                        z,
                        z_tilde
                    )
                )

                loss = (
                    clean_cls_loss
                    +
                    ALPHA_PERT_CE
                    * pert_cls_loss
                    +
                    LAMBDA_INV
                    * inv_loss
                )

            else:

                pert_cls_loss = torch.zeros(
                    (),
                    device=device
                )

                inv_loss = torch.zeros(
                    (),
                    device=device
                )

                loss = clean_cls_loss

            # ------------------------------------------------
            # BACKPROP
            # ------------------------------------------------

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters())
                +
                list(classifier.parameters()),
                max_norm=1.0
            )

            optimizer.step()

            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            total_loss_sum += (
                loss.item()
                * x.size(0)
            )

            clean_cls_sum += (
                clean_cls_loss.item()
                * x.size(0)
            )

            pert_cls_sum += (
                pert_cls_loss.item()
                * x.size(0)
            )

            inv_loss_sum += (
                inv_loss.item()
                * x.size(0)
            )

            pred = logits.argmax(
                dim=1
            )

            correct += (
                pred == y
            ).sum().item()

            total += y.numel()

        train_loss = (
            total_loss_sum / total
        )

        train_clean_cls = (
            clean_cls_sum / total
        )

        train_pert_cls = (
            pert_cls_sum / total
        )

        train_inv = (
            inv_loss_sum / total
        )

        train_acc = (
            correct / total
        )

        val_result = evaluate(
            encoder,
            classifier,
            val_loader,
            device
        )

        scheduler.step(
            val_result["acc"]
        )

        history.append({

            "epoch": epoch + 1,

            "train_loss": train_loss,

            "train_clean_cls":
                train_clean_cls,

            "train_pert_cls":
                train_pert_cls,

            "train_inv":
                train_inv,

            "train_acc":
                train_acc,

            "val_loss":
                val_result["loss"],

            "val_acc":
                val_result["acc"],

            "lr":
                optimizer.param_groups[0]["lr"]
        })

        print(
            f"Epoch {epoch+1:03d} | "
            f"train {train_acc*100:6.2f}% | "
            f"val {val_result['acc']*100:6.2f}% | "
            f"cleanCE {train_clean_cls:.4f} | "
            f"pertCE {train_pert_cls:.4f} | "
            f"inv {train_inv:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.2e}"
        )

        # ------------------------------------------------
        # BEST CHECKPOINT
        # ------------------------------------------------

        if val_result["acc"] > best_val:

            best_val = (
                val_result["acc"]
            )

            best_epoch = epoch + 1

            patience_counter = 0

            torch.save(
                {
                    "epoch":
                        epoch + 1,

                    "encoder":
                        encoder.state_dict(),

                    "classifier":
                        classifier.state_dict(),

                    "val_acc":
                        best_val,

                    "use_sil":
                        use_sil,

                    "static_gain_sigma":
                        STATIC_GAIN_SIGMA,

                    "alpha_pert_ce":
                        ALPHA_PERT_CE,

                    "lambda_inv":
                        LAMBDA_INV,

                    "seed":
                        SEED,

                    "architecture":
                        "SimpleCSIEncoder",

                    "input":
                        "raw_CSI_250x90",

                    "preprocessing":
                        "train_fitted_per_subcarrier_global_normalization_only",

                    "objective":
                        "CE_clean + alpha_CE*CE_perturbed + lambda_inv*cosine"
                },
                save_path
            )

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                f"Early stopping at epoch {epoch+1}"
            )

            break

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------

    checkpoint = torch.load(
        save_path,
        map_location=device,
        weights_only=True
    )

    encoder.load_state_dict(
        checkpoint["encoder"]
    )

    classifier.load_state_dict(
        checkpoint["classifier"]
    )

    test_result = evaluate(
        encoder,
        classifier,
        test_loader,
        device
    )

    print()
    print("BEST RESULT")
    print("-" * 50)
    print(
        f"Best epoch : {best_epoch}"
    )
    print(
        f"Best val   : {best_val*100:.2f}%"
    )
    print(
        f"Test acc   : {test_result['acc']*100:.2f}%"
    )

    return {

        "name":
            name,

        "use_sil":
            use_sil,

        "best_epoch":
            best_epoch,

        "best_val_acc":
            best_val,

        "test_acc":
            test_result["acc"],

        "checkpoint":
            str(save_path),

        "history":
            history
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

    print()
    print("=" * 70)
    print("CLEAN SIL v2")
    print("TASK-PRESERVING STATIC-COMPONENT INVARIANCE")
    print("=" * 70)

    print("Device:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = load_data()

    # --------------------------------------------------------
    # TRAIN-FITTED NORMALIZATION
    # --------------------------------------------------------

    mean, std = (
        fit_global_normalization(
            X_train
        )
    )

    X_train = normalize(
        X_train,
        mean,
        std
    )

    X_val = normalize(
        X_val,
        mean,
        std
    )

    X_test = normalize(
        X_test,
        mean,
        std
    )

    print()
    print("After global normalization:")

    print(
        "train mean/std:",
        float(X_train.mean()),
        float(X_train.std())
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )

    val_ds = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val)
    )

    test_ds = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test)
    )

    # --------------------------------------------------------
    # RAW CE
    # --------------------------------------------------------

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
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

    raw_ce = train_experiment(
        name="RAW_CE",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        use_sil=False
    )

    # --------------------------------------------------------
    # RESET SEED
    # --------------------------------------------------------

    set_seed(SEED)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # RAW SIL v2
    # --------------------------------------------------------

    raw_sil = train_experiment(
        name="RAW_SIL_V2",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        use_sil=True
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results = {

        "seed":
            SEED,

        "input_shape":
            [250, 90],

        "preprocessing": {

            "hampel":
                False,

            "detrend":
                False,

            "bandpass":
                False,

            "velocity":
                False,

            "acceleration":
                False,

            "stft":
                False,

            "global_train_fitted_normalization":
                True
        },

        "method": {

            "static_definition":
                "temporal_mean_per_subcarrier",

            "residual_definition":
                "x_minus_static",

            "perturbation":
                "static_multiplicative_gain",

            "gain_distribution":
                "Normal(0, sigma^2)",

            "sigma":
                STATIC_GAIN_SIGMA,

            "alpha_perturbed_ce":
                ALPHA_PERT_CE,

            "lambda_inv":
                LAMBDA_INV,

            "objective":
                "CE(clean) + alpha*CE(perturbed) + lambda*cosine"
        },

        "raw_ce": {

            "best_val_acc":
                raw_ce["best_val_acc"],

            "test_acc":
                raw_ce["test_acc"],

            "best_epoch":
                raw_ce["best_epoch"],

            "checkpoint":
                raw_ce["checkpoint"]
        },

        "raw_sil_v2": {

            "best_val_acc":
                raw_sil["best_val_acc"],

            "test_acc":
                raw_sil["test_acc"],

            "best_epoch":
                raw_sil["best_epoch"],

            "checkpoint":
                raw_sil["checkpoint"]
        }
    }

    with open(
        OUT_DIR / "clean_sil_v2_results.json",
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(
        f"RAW + CE     : "
        f"{raw_ce['test_acc']*100:.2f}%"
    )

    print(
        f"RAW + SIL v2 : "
        f"{raw_sil['test_acc']*100:.2f}%"
    )

    print(
        f"Difference    : "
        f"{(raw_sil['test_acc'] - raw_ce['test_acc'])*100:+.2f} pp"
    )

    print()
    print(
        "Results saved to:",
        OUT_DIR
    )


if __name__ == "__main__":
    main()
