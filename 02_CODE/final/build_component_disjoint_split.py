from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "data" / "raw"

FILES = {
    "train": (
        DATA / "data" / "X_train.npy",
        DATA / "label" / "y_train.npy",
    ),
    "val": (
        DATA / "data" / "X_val.npy",
        DATA / "label" / "y_val.npy",
    ),
    "test": (
        DATA / "data" / "X_test.npy",
        DATA / "label" / "y_test.npy",
    ),
}

OVERLAP_LEN = 150
SEED = 20261005

OUTDIR = ROOT / "results" / "temporal_overlap_audit"
OUTDIR.mkdir(parents=True, exist_ok=True)


class DSU:
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int32)
        self.size = np.ones(n, dtype=np.int32)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
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
    print("BUILDING COMPONENT-DISJOINT UT-HAR SPLIT")
    print("=" * 78)

    arrays = {}
    labels = {}
    offsets = {}

    total = 0

    for split, (xf, yf) in FILES.items():

        X = np.load(xf, mmap_mode="r")
        y = np.load(yf).astype(np.int64)

        arrays[split] = X
        labels[split] = y

        offsets[split] = total
        total += len(X)

        print(
            f"{split}: "
            f"X={X.shape}, y={y.shape}"
        )

    # --------------------------------------------------------
    # Build exact-overlap graph
    # --------------------------------------------------------

    print()
    print("Building exact-overlap graph...")

    index = defaultdict(list)

    for split, X in arrays.items():

        offset = offsets[split]

        npos = X.shape[1] - OVERLAP_LEN + 1

        for i in range(len(X)):

            gid = offset + i

            for start in range(npos):

                block = X[
                    i,
                    start:start + OVERLAP_LEN
                ]

                index[
                    block_hash(block)
                ].append(
                    (gid, split, i, start)
                )

            if i % 500 == 0 or i == len(X) - 1:
                print(
                    f"  {split}: "
                    f"{i+1}/{len(X)}"
                )

    dsu = DSU(total)

    # --------------------------------------------------------
    # Resolve exact matches
    # --------------------------------------------------------

    for entries in index.values():

        if len(entries) < 2:
            continue

        groups = []

        for entry in entries:

            gid, split, idx, start = entry

            block = arrays[split][
                idx,
                start:start + OVERLAP_LEN
            ]

            placed = False

            for group in groups:

                _, rs, ri, rst = group[0]

                ref = arrays[rs][
                    ri,
                    rst:rst + OVERLAP_LEN
                ]

                if np.array_equal(block, ref):

                    group.append(entry)
                    placed = True
                    break

            if not placed:
                groups.append([entry])

        for group in groups:

            if len(group) < 2:
                continue

            first = group[0][0]

            for entry in group[1:]:
                dsu.union(first, entry[0])

    # --------------------------------------------------------
    # Build components
    # --------------------------------------------------------

    components = defaultdict(list)

    for gid in range(total):
        components[dsu.find(gid)].append(gid)

    print()
    print("Components found:", len(components))

    # --------------------------------------------------------
    # Decode global ID
    # --------------------------------------------------------

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
                return split, gid - start

        raise RuntimeError(gid)

    # --------------------------------------------------------
    # Determine component labels
    # --------------------------------------------------------

    component_info = []

    for root, members in components.items():

        counts = Counter()

        for gid in members:

            split, idx = decode(gid)

            counts[
                int(labels[split][idx])
            ] += 1

        if len(counts) != 1:

            raise RuntimeError(
                "Found mixed-label component!"
            )

        label = next(iter(counts))

        component_info.append(
            {
                "root": int(root),
                "label": int(label),
                "members": members,
                "size": len(members),
            }
        )

    # --------------------------------------------------------
    # Stratified component assignment
    #
    # We assign whole components while trying to match the
    # original WINDOW-level class proportions.
    # --------------------------------------------------------

    rng = np.random.default_rng(SEED)

    by_label = defaultdict(list)

    for comp in component_info:
        by_label[comp["label"]].append(comp)

    assignments = {}

    target_fracs = {
        "train": 0.80,
        "val": 0.10,
        "test": 0.10,
    }

    for label in sorted(by_label):

        comps = by_label[label]

        # Randomized but deterministic ordering.
        rng.shuffle(comps)

        total_windows = sum(
            c["size"] for c in comps
        )

        targets = {
            split: target_fracs[split] * total_windows
            for split in target_fracs
        }

        current = {
            "train": 0,
            "val": 0,
            "test": 0,
        }

        # Larger components first gives more stable
        # approximate stratification.
        comps = sorted(
            comps,
            key=lambda c: c["size"],
            reverse=True
        )

        for comp in comps:

            # Assign to the split with the largest
            # remaining target for this class.
            remaining = {
                split:
                    targets[split] - current[split]
                for split in target_fracs
            }

            chosen = max(
                remaining,
                key=remaining.get
            )

            assignments[
                comp["root"]
            ] = chosen

            current[chosen] += comp["size"]

    # --------------------------------------------------------
    # Create final index lists
    # --------------------------------------------------------

    split_indices = {
        "train": [],
        "val": [],
        "test": [],
    }

    component_split_counts = Counter()

    for comp in component_info:

        split = assignments[
            comp["root"]
        ]

        component_split_counts[split] += 1

        split_indices[split].extend(
            comp["members"]
        )

    # --------------------------------------------------------
    # Sort global IDs
    # --------------------------------------------------------

    for split in split_indices:

        split_indices[split] = sorted(
            split_indices[split]
        )

    # --------------------------------------------------------
    # Decode indices and collect labels
    # --------------------------------------------------------

    decoded = {}

    for split, gids in split_indices.items():

        records = []

        for gid in gids:

            source_split, idx = decode(gid)

            records.append(
                {
                    "source_split": source_split,
                    "source_index": int(idx),
                    "global_id": int(gid),
                    "label": int(
                        labels[source_split][idx]
                    ),
                }
            )

        decoded[split] = records

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    sets = {
        k: set(v)
        for k, v in split_indices.items()
    }

    assert not (
        sets["train"] &
        sets["val"]
    )

    assert not (
        sets["train"] &
        sets["test"]
    )

    assert not (
        sets["val"] &
        sets["test"]
    )

    assert (
        len(
            sets["train"] |
            sets["val"] |
            sets["test"]
        )
        == total
    )

    # Verify no component crosses splits.
    gid_to_split = {}

    for split, gids in split_indices.items():

        for gid in gids:
            gid_to_split[gid] = split

    crossing_components = 0

    for comp in component_info:

        assigned = {
            gid_to_split[gid]
            for gid in comp["members"]
        }

        if len(assigned) != 1:
            crossing_components += 1

    assert crossing_components == 0

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("FINAL SPLIT")
    print("=" * 78)

    summary = {}

    for split in ["train", "val", "test"]:

        recs = decoded[split]

        counts = Counter(
            r["label"]
            for r in recs
        )

        summary[split] = {
            "components":
                int(component_split_counts[split]),

            "windows":
                len(recs),

            "class_counts": {
                str(k): int(v)
                for k, v in sorted(counts.items())
            }
        }

        print(
            f"{split.upper():5s}: "
            f"{len(recs):4d} windows, "
            f"{component_split_counts[split]:3d} components"
        )

        print(
            "      classes:",
            dict(sorted(counts.items()))
        )

    print()
    print(
        "Crossing components:",
        crossing_components
    )

    print(
        "Total windows:",
        sum(
            len(v)
            for v in split_indices.values()
        )
    )

    # --------------------------------------------------------
    # Save index files
    # --------------------------------------------------------

    for split in ["train", "val", "test"]:

        np.save(
            OUTDIR / f"component_disjoint_{split}_global_ids.npy",
            np.asarray(
                split_indices[split],
                dtype=np.int32
            )
        )

    # Also save decoded source indices.
    for split in ["train", "val", "test"]:

        with open(
            OUTDIR /
            f"component_disjoint_{split}.json",
            "w"
        ) as f:

            json.dump(
                decoded[split],
                f,
                indent=2
            )

    manifest = {
        "protocol":
            "component_disjoint_exact_150_row_overlap",

        "seed":
            SEED,

        "overlap_len":
            OVERLAP_LEN,

        "total_windows":
            total,

        "components":
            len(component_info),

        "crossing_components":
            crossing_components,

        "component_counts":
            dict(component_split_counts),

        "summary":
            summary,

        "source_dataset":
            {
                "train": len(arrays["train"]),
                "val": len(arrays["val"]),
                "test": len(arrays["test"]),
            }
    }

    with open(
        OUTDIR /
        "component_disjoint_manifest.json",
        "w"
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2
        )

    print()
    print("Saved split files to:")
    print(OUTDIR)

    print()
    print("=" * 78)
    print("DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()
