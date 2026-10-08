from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = ROOT / "data" / "data" / "raw"

FILES = {
    "train": (
        DATA_ROOT / "data" / "X_train.npy",
        DATA_ROOT / "label" / "y_train.npy"
    ),
    "val": (
        DATA_ROOT / "data" / "X_val.npy",
        DATA_ROOT / "label" / "y_val.npy"
    ),
    "test": (
        DATA_ROOT / "data" / "X_test.npy",
        DATA_ROOT / "label" / "y_test.npy"
    )
}

OVERLAP_LEN = 150

OUT = (
    ROOT
    / "results"
    / "temporal_overlap_audit"
    / "temporal_group_label_audit.json"
)


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
                self.parent[self.parent[x]]
            )

            x = self.parent[x]

        return x

    def union(self, a, b):

        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def block_hash(block):

    return hashlib.blake2b(
        block.tobytes(order="C"),
        digest_size=16
    ).digest()


def main():

    print("=" * 78)
    print("UT-HAR TEMPORAL GROUP LABEL AUDIT")
    print("=" * 78)

    arrays = {}
    labels = {}

    offsets = {}
    total = 0

    for split, (xfile, yfile) in FILES.items():

        X = np.load(
            xfile,
            mmap_mode="r"
        )

        y = np.load(
            yfile
        ).astype(np.int64)

        arrays[split] = X
        labels[split] = y

        offsets[split] = total

        total += len(X)

        print(
            f"{split}: X={X.shape}, y={y.shape}"
        )

    # --------------------------------------------------------
    # Build exact-overlap graph
    # --------------------------------------------------------

    index = defaultdict(list)

    print()
    print("Building exact 150-row block index...")

    for split, X in arrays.items():

        offset = offsets[split]

        n_positions = (
            X.shape[1]
            - OVERLAP_LEN
            + 1
        )

        for i in range(len(X)):

            gid = offset + i

            for start in range(n_positions):

                block = X[
                    i,
                    start:start + OVERLAP_LEN
                ]

                index[
                    block_hash(block)
                ].append(
                    (
                        gid,
                        split,
                        i,
                        start
                    )
                )

            if (
                i % 500 == 0
                or i == len(X) - 1
            ):

                print(
                    f"  {split}: "
                    f"{i + 1}/{len(X)}"
                )

    dsu = DSU(total)

    # --------------------------------------------------------
    # Resolve exact matches
    # --------------------------------------------------------

    for h, entries in index.items():

        if len(entries) < 2:
            continue

        verified_groups = []

        for entry in entries:

            gid, split, idx, start = entry

            block = arrays[split][
                idx,
                start:start + OVERLAP_LEN
            ]

            placed = False

            for group in verified_groups:

                rgid, rsplit, ridx, rstart = group[0]

                ref_block = arrays[rsplit][
                    ridx,
                    rstart:rstart + OVERLAP_LEN
                ]

                if np.array_equal(
                    block,
                    ref_block
                ):

                    group.append(entry)
                    placed = True
                    break

            if not placed:

                verified_groups.append(
                    [entry]
                )

        for group in verified_groups:

            if len(group) < 2:
                continue

            first = group[0][0]

            for entry in group[1:]:

                dsu.union(
                    first,
                    entry[0]
                )

    # --------------------------------------------------------
    # Build components
    # --------------------------------------------------------

    components = defaultdict(list)

    for gid in range(total):

        components[
            dsu.find(gid)
        ].append(gid)

    boundaries = []

    for split, X in arrays.items():

        start = offsets[split]

        end = start + len(X)

        boundaries.append(
            (start, end, split)
        )

    def decode(gid):

        for start, end, split in boundaries:

            if start <= gid < end:

                return (
                    split,
                    gid - start
                )

        raise RuntimeError(
            f"Cannot decode gid {gid}"
        )

    # --------------------------------------------------------
    # Analyze labels
    # --------------------------------------------------------

    purities = []

    mixed_label_components = 0

    single_label_components = 0

    component_records = []

    for root, members in components.items():

        label_counts = Counter()
        split_counts = Counter()

        member_records = []

        for gid in members:

            split, idx = decode(gid)

            label = int(
                labels[split][idx]
            )

            label_counts[label] += 1
            split_counts[split] += 1

            member_records.append(
                {
                    "global_id": int(gid),
                    "split": split,
                    "index": int(idx),
                    "label": label
                }
            )

        total_members = len(members)

        dominant_label, dominant_count = (
            label_counts.most_common(1)[0]
        )

        purity = (
            dominant_count
            / total_members
        )

        purities.append(purity)

        if len(label_counts) == 1:

            single_label_components += 1

        else:

            mixed_label_components += 1

        component_records.append(
            {
                "root": int(root),
                "size": total_members,
                "label_counts": {
                    str(k): int(v)
                    for k, v in label_counts.items()
                },
                "dominant_label":
                    int(dominant_label),
                "purity":
                    float(purity),
                "split_counts": {
                    k: int(v)
                    for k, v in split_counts.items()
                },
                "members": member_records
            }
        )

    purities = np.asarray(
        purities,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Purity thresholds
    # --------------------------------------------------------

    purity_100 = int(
        np.sum(purities == 1.0)
    )

    purity_ge_0_95 = int(
        np.sum(purities >= 0.95)
    )

    purity_ge_0_90 = int(
        np.sum(purities >= 0.90)
    )

    purity_ge_0_75 = int(
        np.sum(purities >= 0.75)
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("LABEL CONSISTENCY")
    print("=" * 78)

    print(
        "Total components:",
        len(component_records)
    )

    print(
        "Single-label components:",
        single_label_components
    )

    print(
        "Mixed-label components:",
        mixed_label_components
    )

    print(
        "100% label-pure:",
        purity_100
    )

    print(
        ">=95% purity:",
        purity_ge_0_95
    )

    print(
        ">=90% purity:",
        purity_ge_0_90
    )

    print(
        ">=75% purity:",
        purity_ge_0_75
    )

    print(
        "Mean purity:",
        float(purities.mean())
    )

    print(
        "Median purity:",
        float(np.median(purities))
    )

    print(
        "Minimum purity:",
        float(purities.min())
    )

    # --------------------------------------------------------
    # Most mixed components
    # --------------------------------------------------------

    most_mixed = sorted(
        component_records,
        key=lambda r: (
            r["purity"],
            -r["size"]
        )
    )[:20]

    print()
    print("=" * 78)
    print("MOST MIXED COMPONENTS")
    print("=" * 78)

    for i, record in enumerate(
        most_mixed,
        start=1
    ):

        print(
            f"#{i:2d}: "
            f"size={record['size']:3d}, "
            f"purity={record['purity']:.3f}, "
            f"labels={record['label_counts']}, "
            f"splits={record['split_counts']}"
        )

    # --------------------------------------------------------
    # Largest components with labels
    # --------------------------------------------------------

    largest = sorted(
        component_records,
        key=lambda r: r["size"],
        reverse=True
    )[:20]

    print()
    print("=" * 78)
    print("LARGEST COMPONENTS WITH LABELS")
    print("=" * 78)

    for i, record in enumerate(
        largest,
        start=1
    ):

        print(
            f"#{i:2d}: "
            f"size={record['size']:3d}, "
            f"purity={record['purity']:.3f}, "
            f"labels={record['label_counts']}, "
            f"splits={record['split_counts']}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = {

        "criterion": {
            "exact_contiguous_rows":
                OVERLAP_LEN,

            "description":
                (
                    "Connected components are formed "
                    "from windows sharing an identical "
                    "contiguous 150-row CSI block."
                ),

            "not_original_recording_claim":
                True
        },

        "dataset": {
            "total_windows": total,
            "train": len(arrays["train"]),
            "val": len(arrays["val"]),
            "test": len(arrays["test"])
        },

        "summary": {
            "components":
                len(component_records),

            "single_label_components":
                single_label_components,

            "mixed_label_components":
                mixed_label_components,

            "purity_100_percent":
                purity_100,

            "purity_ge_95_percent":
                purity_ge_0_95,

            "purity_ge_90_percent":
                purity_ge_0_90,

            "purity_ge_75_percent":
                purity_ge_0_75,

            "mean_purity":
                float(purities.mean()),

            "median_purity":
                float(np.median(purities)),

            "minimum_purity":
                float(purities.min())
        },

        "most_mixed_components":
            most_mixed,

        "largest_components":
            largest
    }

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUT,
        "w"
    ) as f:

        json.dump(
            output,
            f,
            indent=2
        )

    print()
    print("Saved:")
    print(OUT)

    print()
    print("=" * 78)
    print("LABEL AUDIT COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
