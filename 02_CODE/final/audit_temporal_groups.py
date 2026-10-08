from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json
import numpy as np


# ============================================================
# UT-HAR TEMPORAL OVERLAP / GROUP AUDIT
#
# We look for EXACT contiguous sequences of 150 CSI rows
# shared between windows.
#
# If two windows share an identical 150 x 90 block,
# they are connected in the temporal-overlap graph.
#
# IMPORTANT:
# This does NOT claim that a connected component is an
# original recording. It only establishes exact temporal
# overlap according to the defined criterion.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = ROOT / "data" / "data" / "raw"

X_TRAIN = DATA_ROOT / "data" / "X_train.npy"
X_VAL   = DATA_ROOT / "data" / "X_val.npy"
X_TEST  = DATA_ROOT / "data" / "X_test.npy"

OUT = (
    ROOT
    / "results"
    / "temporal_overlap_audit"
    / "temporal_group_audit.json"
)

# Exact contiguous overlap threshold.
OVERLAP_LEN = 150

# ------------------------------------------------------------
# Union-Find
# ------------------------------------------------------------

class DSU:

    def __init__(self, n):

        self.parent = np.arange(
            n,
            dtype=np.int32
        )

        self.size = np.ones(
            n,
            dtype=np.int32
        )

    def find(self, x):

        while self.parent[x] != x:

            self.parent[x] = (
                self.parent[
                    self.parent[x]
                ]
            )

            x = self.parent[x]

        return x

    def union(self, a, b):

        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return False

        if self.size[ra] < self.size[rb]:

            ra, rb = rb, ra

        self.parent[rb] = ra

        self.size[ra] += self.size[rb]

        return True


# ------------------------------------------------------------
# Hash exact contiguous block
# ------------------------------------------------------------

def block_hash(block):

    return hashlib.blake2b(
        block.tobytes(order="C"),
        digest_size=16
    ).digest()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 78)
    print("UT-HAR TEMPORAL OVERLAP / GROUP AUDIT")
    print("=" * 78)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    X_train = np.load(
        X_TRAIN,
        mmap_mode="r"
    )

    X_val = np.load(
        X_VAL,
        mmap_mode="r"
    )

    X_test = np.load(
        X_TEST,
        mmap_mode="r"
    )

    arrays = [
        ("train", X_train),
        ("val", X_val),
        ("test", X_test)
    ]

    print()
    print("Dataset shapes:")

    for name, X in arrays:

        print(
            f"  {name:5s}: "
            f"{X.shape}, dtype={X.dtype}"
        )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    for name, X in arrays:

        if X.ndim != 3:
            raise ValueError(
                f"{name} is not 3-D: {X.shape}"
            )

        if X.shape[1] < OVERLAP_LEN:
            raise ValueError(
                f"{name} has fewer than "
                f"{OVERLAP_LEN} temporal rows."
            )

    # --------------------------------------------------------
    # Concatenate logical indexing
    # --------------------------------------------------------

    offsets = {}

    total = 0

    for name, X in arrays:

        offsets[name] = total

        total += len(X)

    print()
    print("Total windows:", total)
    print(
        "Overlap criterion:",
        f"exact contiguous {OVERLAP_LEN} x {X_train.shape[2]} block"
    )

    # --------------------------------------------------------
    # Build hash index.
    #
    # Every 150-row contiguous segment inside every window
    # becomes a candidate temporal-overlap signature.
    #
    # There are:
    # 250 - 150 + 1 = 101
    # signatures per window.
    # --------------------------------------------------------

    index = defaultdict(list)

    print()
    print("Building exact contiguous-block index...")

    total_signatures = 0

    for split_name, X in arrays:

        global_offset = offsets[split_name]

        n_windows = len(X)

        n_positions = (
            X.shape[1]
            - OVERLAP_LEN
            + 1
        )

        for i in range(n_windows):

            global_id = (
                global_offset + i
            )

            window = X[i]

            for start in range(n_positions):

                block = window[
                    start:start + OVERLAP_LEN
                ]

                h = block_hash(block)

                index[h].append(
                    (
                        global_id,
                        split_name,
                        i,
                        start
                    )
                )

                total_signatures += 1

            if (
                i % 500 == 0
                or i == n_windows - 1
            ):

                print(
                    f"  {split_name}: "
                    f"{i + 1}/{n_windows}"
                )

    print()
    print(
        "Total block signatures:",
        total_signatures
    )

    # --------------------------------------------------------
    # Resolve hash collisions by exact verification.
    # --------------------------------------------------------

    dsu = DSU(total)

    exact_matches = []

    duplicate_hashes = 0

    for h, entries in index.items():

        if len(entries) < 2:
            continue

        duplicate_hashes += 1

        # A hash collision is possible in principle,
        # so verify blocks exactly before unioning.
        groups = []

        for entry in entries:

            gid, split, local_idx, start = entry

            X = dict(arrays)[split]

            block = X[
                local_idx,
                start:start + OVERLAP_LEN
            ]

            placed = False

            for g in groups:

                ref = g[0]

                rgid, rsplit, ridx, rstart = ref

                RX = dict(arrays)[rsplit]

                ref_block = RX[
                    ridx,
                    rstart:rstart + OVERLAP_LEN
                ]

                if np.array_equal(
                    block,
                    ref_block
                ):

                    g.append(entry)
                    placed = True
                    break

            if not placed:

                groups.append(
                    [entry]
                )

        for group in groups:

            if len(group) < 2:
                continue

            # Union all windows in this exact-overlap group.
            first_gid = group[0][0]

            for entry in group[1:]:

                second_gid = entry[0]

                if dsu.union(
                    first_gid,
                    second_gid
                ):

                    exact_matches.append(
                        {
                            "a": group[0],
                            "b": entry
                        }
                    )

    print()
    print(
        "Hash buckets containing multiple signatures:",
        duplicate_hashes
    )

    print(
        "Exact overlap unions:",
        len(exact_matches)
    )

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    components = defaultdict(list)

    for gid in range(total):

        root = dsu.find(gid)

        components[root].append(gid)

    component_sizes = np.array(
        [
            len(v)
            for v in components.values()
        ],
        dtype=np.int32
    )

    component_sizes.sort()

    # --------------------------------------------------------
    # Map global ID back to split/local ID
    # --------------------------------------------------------

    boundaries = []

    for name, X in arrays:

        start = offsets[name]

        end = start + len(X)

        boundaries.append(
            (
                start,
                end,
                name
            )
        )

    def decode_gid(gid):

        for start, end, name in boundaries:

            if start <= gid < end:

                return (
                    name,
                    gid - start
                )

        raise RuntimeError(
            "Could not decode global ID."
        )

    # --------------------------------------------------------
    # Component split composition
    # --------------------------------------------------------

    mixed_components = 0

    component_split_examples = []

    for root, members in components.items():

        splits = Counter()

        for gid in members:

            split, local_idx = decode_gid(gid)

            splits[split] += 1

        if len(splits) > 1:

            mixed_components += 1

            if len(component_split_examples) < 20:

                component_split_examples.append(
                    {
                        "component_size":
                            len(members),

                        "split_counts":
                            dict(splits)
                    }
                )

    # --------------------------------------------------------
    # Useful statistics
    # --------------------------------------------------------

    n_components = len(
        components
    )

    singleton_components = int(
        np.sum(component_sizes == 1)
    )

    components_ge2 = int(
        np.sum(component_sizes >= 2)
    )

    components_ge5 = int(
        np.sum(component_sizes >= 5)
    )

    components_ge10 = int(
        np.sum(component_sizes >= 10)
    )

    components_ge20 = int(
        np.sum(component_sizes >= 20)
    )

    components_ge50 = int(
        np.sum(component_sizes >= 50)
    )

    largest = int(
        component_sizes[-1]
    )

    median_size = float(
        np.median(component_sizes)
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("CONNECTED COMPONENT RESULTS")
    print("=" * 78)

    print(
        "Number of components:",
        n_components
    )

    print(
        "Singleton components:",
        singleton_components
    )

    print(
        "Components >= 2 windows:",
        components_ge2
    )

    print(
        "Components >= 5 windows:",
        components_ge5
    )

    print(
        "Components >= 10 windows:",
        components_ge10
    )

    print(
        "Components >= 20 windows:",
        components_ge20
    )

    print(
        "Components >= 50 windows:",
        components_ge50
    )

    print(
        "Median component size:",
        median_size
    )

    print(
        "Largest component size:",
        largest
    )

    print()
    print(
        "Components containing windows "
        "from multiple supplied splits:",
        mixed_components
    )

    # --------------------------------------------------------
    # Largest components
    # --------------------------------------------------------

    largest_components = sorted(
        components.items(),
        key=lambda kv: len(kv[1]),
        reverse=True
    )[:20]

    largest_summary = []

    print()
    print("Largest components:")

    for rank, (root, members) in enumerate(
        largest_components,
        start=1
    ):

        split_counts = Counter()

        for gid in members:

            split, local_idx = decode_gid(gid)

            split_counts[split] += 1

        summary = {
            "rank": rank,
            "size": len(members),
            "split_counts": dict(
                split_counts
            )
        }

        largest_summary.append(
            summary
        )

        print(
            f"  #{rank:2d}: "
            f"size={len(members):4d}, "
            f"splits={dict(split_counts)}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result = {

        "criterion": {
            "overlap_rows":
                OVERLAP_LEN,

            "exact_match":
                True,

            "description":
                (
                    "Two windows are connected when "
                    "they contain an identical contiguous "
                    "150-row CSI block."
                ),

            "important_interpretation":
                (
                    "Connected components represent "
                    "exact temporal-overlap groups under "
                    "this operational criterion; they are "
                    "not assumed to equal original recording IDs."
                )
        },

        "dataset": {
            "train": list(X_train.shape),
            "val": list(X_val.shape),
            "test": list(X_test.shape),
            "total_windows": total,
            "dtype": str(X_train.dtype)
        },

        "signatures": {
            "total":
                total_signatures,

            "duplicate_hash_buckets":
                duplicate_hashes,

            "exact_overlap_unions":
                len(exact_matches)
        },

        "components": {
            "number":
                n_components,

            "singleton":
                singleton_components,

            "ge_2":
                components_ge2,

            "ge_5":
                components_ge5,

            "ge_10":
                components_ge10,

            "ge_20":
                components_ge20,

            "ge_50":
                components_ge50,

            "median_size":
                median_size,

            "largest_size":
                largest,

            "mixed_supplied_splits":
                mixed_components
        },

        "largest_components":
            largest_summary,

        "mixed_component_examples":
            component_split_examples
    }

    with open(
        OUT,
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    print()
    print("Saved:")
    print(OUT)

    print()
    print("=" * 78)
    print("AUDIT COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
