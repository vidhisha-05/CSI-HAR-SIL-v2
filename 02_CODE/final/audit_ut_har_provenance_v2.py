from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json
import numpy as np


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "data" / "raw"

OUT_DIR = ROOT / "results" / "ut_har_provenance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]

X = {
    "train": np.load(DATA / "data" / "X_train.npy"),
    "val":   np.load(DATA / "data" / "X_val.npy"),
    "test":  np.load(DATA / "data" / "X_test.npy"),
}

Y = {
    "train": np.load(DATA / "label" / "y_train.npy"),
    "val":   np.load(DATA / "label" / "y_val.npy"),
    "test":  np.load(DATA / "label" / "y_test.npy"),
}

THRESHOLDS = [1, 10, 50, 100, 150, 200, 225, 240, 249]

# We only perform expensive exact contiguous analysis
# on pairs with at least this many shared row hashes.
STRONG_PAIR_THRESHOLD = 100


# ============================================================
# HASHING
# ============================================================

def row_hash(row):
    """
    Exact hash of one CSI timestep (90-dimensional row).
    """
    return hashlib.blake2b(
        np.ascontiguousarray(row).tobytes(),
        digest_size=16
    ).digest()


def hash_split(arr):
    """
    Return H[window, timestep] containing exact row hashes.
    """
    n, t, k = arr.shape

    H = np.empty((n, t), dtype=object)

    for i in range(n):
        for j in range(t):
            H[i, j] = row_hash(arr[i, j])

        if (i + 1) % 500 == 0 or i == n - 1:
            print(f"  hashed {i+1}/{n} windows")

    return H


# ============================================================
# BUILD HASH INDEX
# ============================================================

def build_index(H):
    """
    hash -> list of (window_index, timestep_index)
    """
    index = defaultdict(list)

    n, t = H.shape

    for i in range(n):
        for j in range(t):
            index[H[i, j]].append((i, j))

    return index


# ============================================================
# CROSS-SPLIT PAIR COUNTS
# ============================================================

def candidate_pairs(H_a, H_b, index_a, index_b):
    """
    Find cross-split window pairs sharing exact row hashes.

    Returns:
        pair_counts[(i,j)] = number of matching row occurrences
    """
    pair_counts = Counter()

    # Iterate over hashes that exist in both splits.
    common_hashes = set(index_a.keys()).intersection(index_b.keys())

    print(f"  common row hashes: {len(common_hashes):,}")

    for h in common_hashes:
        positions_a = index_a[h]
        positions_b = index_b[h]

        for ia, _ in positions_a:
            for ib, _ in positions_b:
                pair_counts[(ia, ib)] += 1

    return pair_counts


# ============================================================
# UNIQUE AFFECTED WINDOW COUNTS
# ============================================================

def affected_window_summary(pair_counts, n_a, n_b):
    """
    Count UNIQUE windows affected above each threshold.
    """
    result = {}

    for threshold in THRESHOLDS:
        strong = [
            (a, b, c)
            for (a, b), c in pair_counts.items()
            if c >= threshold
        ]

        affected_a = sorted(set(a for a, _, _ in strong))
        affected_b = sorted(set(b for _, b, _ in strong))

        result[str(threshold)] = {
            "pair_count": len(strong),
            "unique_split_A_windows": len(affected_a),
            "unique_split_B_windows": len(affected_b),
            "fraction_split_A": len(affected_a) / n_a,
            "fraction_split_B": len(affected_b) / n_b,
        }

    return result


# ============================================================
# EXACT CONTIGUOUS MATCH ANALYSIS
# ============================================================

def longest_contiguous_exact_match(A, B):
    """
    Find the longest contiguous sequence of exactly equal CSI rows
    between two windows A and B.

    Returns:
        {
            "length": longest exact contiguous row match,
            "a_start": start timestep in A,
            "b_start": start timestep in B,
            "shift": b_start - a_start
        }

    Important:
    This operates on actual float rows, not merely hashes.
    """

    TA = A.shape[0]
    TB = B.shape[0]

    # Hash rows first.
    ha = [row_hash(A[i]) for i in range(TA)]
    hb = [row_hash(B[j]) for j in range(TB)]

    # Map (relative diagonal shift) -> matching positions.
    diagonals = defaultdict(list)

    pos_b = defaultdict(list)

    for j, h in enumerate(hb):
        pos_b[h].append(j)

    for i, h in enumerate(ha):
        for j in pos_b.get(h, []):
            diagonals[j - i].append((i, j))

    best = {
        "length": 0,
        "a_start": None,
        "b_start": None,
        "shift": None,
    }

    for shift, matches in diagonals.items():

        # Sort by timestep in A.
        matches.sort()

        run_start = None
        previous_i = None
        previous_j = None

        for i, j in matches:

            if run_start is None:
                run_start = (i, j)

            elif i == previous_i + 1 and j == previous_j + 1:
                pass

            else:
                run_len = previous_i - run_start[0] + 1

                if run_len > best["length"]:
                    best = {
                        "length": run_len,
                        "a_start": run_start[0],
                        "b_start": run_start[1],
                        "shift": shift,
                    }

                run_start = (i, j)

            previous_i = i
            previous_j = j

        if run_start is not None:
            run_len = previous_i - run_start[0] + 1

            if run_len > best["length"]:
                best = {
                    "length": run_len,
                    "a_start": run_start[0],
                    "b_start": run_start[1],
                    "shift": shift,
                }

    # Final safety verification of the claimed contiguous run.
    if best["length"] > 0:
        a0 = best["a_start"]
        b0 = best["b_start"]
        L = best["length"]

        if not np.array_equal(
            A[a0:a0+L],
            B[b0:b0+L]
        ):
            raise RuntimeError(
                "Internal verification failed for contiguous match."
            )

    return best


# ============================================================
# STRONG PAIR REPORT
# ============================================================

def analyze_strong_pairs(split_a, split_b, pair_counts, max_pairs=100):

    strong = [
        (ia, ib, count)
        for (ia, ib), count in pair_counts.items()
        if count >= STRONG_PAIR_THRESHOLD
    ]

    strong.sort(key=lambda x: x[2], reverse=True)

    report = []

    print()
    print("=" * 80)
    print(f"STRONG PAIRS: {split_a} vs {split_b}")
    print("=" * 80)

    for rank, (ia, ib, count) in enumerate(
        strong[:max_pairs], start=1
    ):

        A = X[split_a][ia]
        B = X[split_b][ib]

        exact = longest_contiguous_exact_match(A, B)

        item = {
            "rank": rank,
            "window_a": int(ia),
            "window_b": int(ib),
            "hash_shared_count": int(count),
            "label_a": int(Y[split_a][ia]),
            "label_b": int(Y[split_b][ib]),
            "contiguous_exact_length": int(exact["length"]),
            "a_start": exact["a_start"],
            "b_start": exact["b_start"],
            "shift": exact["shift"],
        }

        report.append(item)

        if rank <= 30:
            print(
                f"{split_a}[{ia}] <-> {split_b}[{ib}] "
                f"hash_shared={count} "
                f"contiguous={exact['length']} "
                f"shift={exact['shift']} "
                f"labels=({Y[split_a][ia]}, {Y[split_b][ib]})"
            )

    return report


# ============================================================
# CHAIN-LIKE SUMMARY
# ============================================================

def summarize_stride_pattern(report):

    shifts = [
        r["shift"]
        for r in report
        if r["contiguous_exact_length"] >= 100
        and r["shift"] is not None
    ]

    lengths = [
        r["contiguous_exact_length"]
        for r in report
        if r["contiguous_exact_length"] >= 100
    ]

    print()
    print("=" * 80)
    print("CONTIGUOUS OVERLAP SUMMARY")
    print("=" * 80)

    if lengths:
        print(f"Pairs with contiguous exact overlap >=100: {len(lengths)}")
        print(
            f"Contiguous length: "
            f"min={min(lengths)}, "
            f"median={np.median(lengths):.1f}, "
            f"max={max(lengths)}"
        )

    if shifts:
        counts = Counter(shifts)

        print("Most common shifts:")

        for shift, count in counts.most_common(20):
            print(f"  shift={shift:+d}: {count} pairs")

    else:
        print("No >=100-row contiguous exact matches found.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("UT-HAR PROVENANCE AUDIT V2")
    print("=" * 80)

    for split in SPLITS:
        print(
            f"{split:>5}: "
            f"X={X[split].shape}, "
            f"y={Y[split].shape}"
        )

    # --------------------------------------------------------
    # Hash all splits
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("HASHING")
    print("=" * 80)

    H = {}

    for split in SPLITS:
        print(f"\nHashing {split}...")
        H[split] = hash_split(X[split])

    # --------------------------------------------------------
    # Build indexes
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("BUILDING HASH INDEXES")
    print("=" * 80)

    INDEX = {}

    for split in SPLITS:
        print(f"Indexing {split}...")
        INDEX[split] = build_index(H[split])
        print(
            f"  unique hashes: "
            f"{len(INDEX[split]):,}"
        )

    # --------------------------------------------------------
    # Cross-split analysis
    # --------------------------------------------------------

    all_results = {}

    pairs_to_check = [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]

    for split_a, split_b in pairs_to_check:

        print()
        print("=" * 80)
        print(f"ANALYZING {split_a} vs {split_b}")
        print("=" * 80)

        pair_counts = candidate_pairs(
            H[split_a],
            H[split_b],
            INDEX[split_a],
            INDEX[split_b],
        )

        print(
            f"Candidate window pairs: "
            f"{len(pair_counts):,}"
        )

        summary = affected_window_summary(
            pair_counts,
            len(X[split_a]),
            len(X[split_b]),
        )

        print()
        print("UNIQUE AFFECTED WINDOW SUMMARY")

        for threshold in THRESHOLDS:
            s = summary[str(threshold)]

            print(
                f"  >= {threshold:3d} rows: "
                f"pairs={s['pair_count']:5d} | "
                f"{split_a} affected="
                f"{s['unique_split_A_windows']:4d}/"
                f"{len(X[split_a])} "
                f"({100*s['fraction_split_A']:.2f}%) | "
                f"{split_b} affected="
                f"{s['unique_split_B_windows']:4d}/"
                f"{len(X[split_b])} "
                f"({100*s['fraction_split_B']:.2f}%)"
            )

        # Strong-pair exact contiguous analysis.
        strong_report = analyze_strong_pairs(
            split_a,
            split_b,
            pair_counts,
            max_pairs=100,
        )

        summarize_stride_pattern(strong_report)

        all_results[f"{split_a}_vs_{split_b}"] = {
            "unique_affected_summary": summary,
            "strong_pair_contiguous_analysis": strong_report,
        }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = {
        "dataset": "UT-HAR",
        "shapes": {
            split: list(X[split].shape)
            for split in SPLITS
        },
        "thresholds": THRESHOLDS,
        "strong_pair_threshold": STRONG_PAIR_THRESHOLD,
        "results": all_results,
    }

    output_path = OUT_DIR / "provenance_v2_summary.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)
    print(output_path)
    print()
    print("AUDIT V2 COMPLETE.")


if __name__ == "__main__":
    main()
