from pathlib import Path
from collections import defaultdict
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "data" / "raw"
SPLIT_DIR = ROOT / "results" / "temporal_overlap_audit"

OVERLAP_LEN = 150

FILES = {
    "train": (
        DATA / "data" / "X_train.npy",
        SPLIT_DIR / "component_disjoint_train_global_ids.npy",
    ),
    "val": (
        DATA / "data" / "X_val.npy",
        SPLIT_DIR / "component_disjoint_val_global_ids.npy",
    ),
    "test": (
        DATA / "data" / "X_test.npy",
        SPLIT_DIR / "component_disjoint_test_global_ids.npy",
    ),
}

# Original global-ID offsets
X_train = np.load(DATA / "data" / "X_train.npy", mmap_mode="r")
X_val   = np.load(DATA / "data" / "X_val.npy", mmap_mode="r")
X_test  = np.load(DATA / "data" / "X_test.npy", mmap_mode="r")

offsets = {
    "train": 0,
    "val": len(X_train),
    "test": len(X_train) + len(X_val),
}

arrays = {
    "train": X_train,
    "val": X_val,
    "test": X_test,
}

# Build mapping from global ID -> original split/index
global_to_source = {}

for split, X in arrays.items():
    offset = offsets[split]

    for i in range(len(X)):
        global_to_source[offset + i] = (split, i)


def block_hash(block):
    return hashlib.blake2b(
        block.tobytes(order="C"),
        digest_size=16
    ).digest()


print("=" * 78)
print("VERIFYING EXACT TEMPORAL OVERLAP")
print("=" * 78)

# ------------------------------------------------------------
# Build hashes for each NEW split
# ------------------------------------------------------------

hash_sets = {}

for new_split in ["train", "val", "test"]:

    global_ids = np.load(
        FILES[new_split][1]
    )

    hashes = set()

    print()
    print(
        f"{new_split.upper()}: "
        f"{len(global_ids)} windows"
    )

    for n, gid in enumerate(global_ids):

        source_split, source_idx = global_to_source[
            int(gid)
        ]

        X = arrays[source_split]

        for start in range(
            X.shape[1] - OVERLAP_LEN + 1
        ):

            block = X[
                source_idx,
                start:start + OVERLAP_LEN
            ]

            hashes.add(
                block_hash(block)
            )

        if n % 500 == 0 or n == len(global_ids) - 1:
            print(
                f"  processed {n + 1}/{len(global_ids)}"
            )

    hash_sets[new_split] = hashes

    print(
        f"  unique {OVERLAP_LEN}-row blocks: "
        f"{len(hashes)}"
    )


# ------------------------------------------------------------
# Check pairwise intersections
# ------------------------------------------------------------

print()
print("=" * 78)
print("PAIRWISE EXACT-OVERLAP CHECK")
print("=" * 78)

pairs = [
    ("train", "val"),
    ("train", "test"),
    ("val", "test"),
]

results = {}

for a, b in pairs:

    intersection = (
        hash_sets[a] &
        hash_sets[b]
    )

    n = len(intersection)

    results[f"{a}_vs_{b}"] = n

    print(
        f"{a.upper()} vs {b.upper()}: "
        f"{n} shared exact {OVERLAP_LEN}-row blocks"
    )


# ------------------------------------------------------------
# Final decision
# ------------------------------------------------------------

print()
print("=" * 78)
print("FINAL DECISION")
print("=" * 78)

if all(v == 0 for v in results.values()):

    print()
    print("PASS")
    print(
        "No exact 150-row CSI blocks are shared "
        "between any new partitions."
    )
    print()
    print(
        "The component-disjoint split passes "
        "the exact-overlap verification."
    )

else:

    print()
    print("FAIL")
    print(
        "Exact temporal overlap still exists "
        "between at least one pair of partitions."
    )

print()
print("=" * 78)
