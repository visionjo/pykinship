"""
Make train, val, and test image lists for FIW-MM:

Usage:
    make_lists.py <fiw-mm> <output-dir> <splits-csv>

The <fiw-mm> argument is the location of the FIW-MM directory.
The <output-dir> argument where you want the lists to be placed.
The <splits-csv> argument is the path to a CSV containing each FID and
marking it as test, train, or val.

The output directory will contain a subdirectory for each split. Inside
each subdirectory, there will be a probe.json, gallery.json, and true_relatives.json.
"""

import json
import random
from itertools import chain
from pathlib import Path
from typing import Dict, List

import pandas as pd
from docopt import docopt


def choose_probes(fiw_mm: Path) -> Dict[str, Path]:
    """
    Choose a random member of each family to be the probe.

    Parameters
    ------------
    fiw_mm:
        Path to the FIW-MM dir.

    Returns
    -----------
    A dictionary mapping FIDs to the probe for that FID.
    """
    probes = {}
    for fid in fiw_mm.glob("F????"):
        mids = list(fid.glob("MID*"))
        if not mids:
            print(f"{fid.name} has no MIDs. Is this a mistake? Skipping.")
            continue
        probe = random.choice(mids)
        probes[fid.name] = probe

    return probes


def make_galleries(
    probes: Dict[str, Path], fid_splits: pd.DataFrame, fiw_mm: Path
) -> Dict[str, List[Path]]:
    """
    Create a gallery for each split. The gallery for each split consists
    of faces of individuals in the split who are not probes. The list is flat.

    Parameters
    --------------
    probes:
        A dictionary mapping an FID to a probe for that FID.
    fid_splits:
        A dataframe with columns [FID, set], which indicates the split each
        FID belongs to.
    fid_mm:
        The path to the FIW-MM dir.

    Returns
    ---------------
    A dictionary whose keys are the split names, and whose values are a lists of Paths.
    Each Path points to a face in the gallery.
    """
    split_names = fid_splits["set"].unique()
    galleries = {_: [] for _ in split_names}

    for split_name in split_names:
        families_in_split = fid_splits[fid_splits["set"].eq(split_name)]

        for row in families_in_split.itertuples():
            mids = (fiw_mm / row.FID).glob("MID*")
            probe_for_family = probes[row.FID]
            gallery_mids = [_ for _ in mids if _.name != probe_for_family.name]
            gallery_faces = chain.from_iterable([_.glob("*.jpg") for _ in gallery_mids])
            galleries[split_name].extend(gallery_faces)

    return galleries


def make_true_relatives(
    probes: Dict[str, Path], galleries: Dict[str, List[Path]]
) -> Dict[str, List[int]]:
    """
    Make a list of true relatives for each probe. Each true relative faces will be
    represented by its index in the gallery array.

    Parameters
    ----------------
    probes:
        A dictionary mapping FIDs to probes.
    galleries:
        A dictionary mapping split names to faces in the gallery for that split.

    Returns
    ---------------
    A dictionary whose keys are FIDs, and who values are lists of ints. Each key represents
    a probe, since each family has one probe. Each value represents a list of true relatives of
    that probe in the gallery as indices in the gallery.

    """
    true_relatives = {fid: [] for fid in probes}
    for split_name in galleries:
        for idx, face in enumerate(galleries[split_name]):
            fid = face.parent.parent.name
            true_relatives[fid].append(idx)

    return true_relatives


def partition_probes_and_true_rel_by_split(probes, true_relatives, fid_splits):
    """
    Separate probes and true relatives by split.

    Parameters
    ---------------
    probe:
        A dictionary mapping from FID to a probe MID.
    true_relatives:
        A dictionary mapping from FID to a list of true relatives.
    fid_splits:
        A dataframe with columns [FID, set], which indicates the split each
        FID belongs to.

    Returns
    ----------------
    probes_by_split:
        A dictionary with keys being splits, and the values being the split original probe dict.
        As a concrete example, we turn probes = {"F0002: F0002/MID1", "F0003: F0003/MID4" ...}
        into {"train": {"F0002: F0002/MID1"}, "test": {"F0003: F0003/MID4"}.

    true_relatives_by_split:
        A dictionary with keys being splits, and the values being the split original true relatives
        dict. Exactly what happens to probes_by_split.
    """
    split_names = fid_splits["set"].unique()
    probes_by_split = {}
    true_relatives_by_split = {}

    for split_name in split_names:
        fids_in_split = fid_splits[fid_splits["set"].eq(split_name)].FID.values
        probes_by_split[split_name] = {fid: probes[fid] for fid in fids_in_split}
        true_relatives_by_split[split_name] = {
            fid: true_relatives[fid] for fid in fids_in_split
        }

    return probes_by_split, true_relatives_by_split


def clean_paths(probes, galleries, fid_splits):
    split_names = fid_splits["set"].unique()
    for split_name in split_names:
        probes[split_name] = {
            fid: _strip_absolute_path(mid, 2) for fid, mid in probes[split_name].items()
        }
        galleries[split_name] = [_strip_absolute_path(_) for _ in galleries[split_name]]


def _strip_absolute_path(path, keep=3):
    return "/".join(path.parts[-keep:])


def save_to_disk(probes, galleries, true_relatives, output_dir, fid_splits):
    split_names = fid_splits["set"].unique()
    for split_name in split_names:
        dir_for_split = output_dir / split_name
        dir_for_split.mkdir(exist_ok=True)
        with open(dir_for_split / "probes.json", "w+") as f:
            json.dump(probes[split_name], f)

        with open(dir_for_split / "gallery.json", "w+") as f:
            json.dump(galleries[split_name], f)

        with open(dir_for_split / "true_relatives.json", "w+") as f:
            json.dump(true_relatives[split_name], f)


if __name__ == "__main__":
    args = docopt(__doc__)
    fiw_mm = Path(args["<fiw-mm>"])
    output_dir = Path(args["<output-dir>"])
    fid_splits = Path(args["<splits-csv>"])

    fid_splits = pd.read_csv(fid_splits)

    probes = choose_probes(fiw_mm)
    galleries = make_galleries(probes, fid_splits, fiw_mm)
    true_relatives = make_true_relatives(probes, galleries)
    probes, true_relatives = partition_probes_and_true_rel_by_split(
        probes, true_relatives, fid_splits
    )
    clean_paths(probes, galleries, fid_splits)
    save_to_disk(probes, galleries, true_relatives, output_dir, fid_splits)
