"""
PyTorch Dataset for the M&A financial features.

Wraps the preprocessed CSV files (``data/processed/train.csv``, etc.)
into ``torch.utils.data.Dataset`` objects that return
``(feature_tensor, label)`` pairs.
"""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

# Columns that are identifiers / text — NOT numeric features
ID_COLS = {
    "Announce Date", "Target Name", "Acquirer Name",
    "Target Ticker", "Acquirer Ticker", "Currency of Deal",
    "Payment Type", "Deal Attributes",
    "Current Target SIC Code", "Current Acquirer SIC Code",
}


class MADealDataset(Dataset):
    """
    PyTorch Dataset for M&A deal records.

    Parameters
    ----------
    csv_path : str | Path
        Path to a preprocessed CSV (train.csv, val.csv, test.csv).
    target_col : str
        Name of the label column (e.g. "CAR"). If absent, labels are NaN.
    feature_cols : list[str] | None
        Explicit list of feature columns. If None, auto-selects all numeric
        columns that are not in ``ID_COLS`` and not ``target_col``.
    """

    def __init__(
        self,
        csv_path: str | Path,
        target_col: str = "CAR",
        feature_cols: Optional[list[str]] = None,
    ):
        import csv as _csv

        csv_path = Path(csv_path)
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = _csv.DictReader(f)
            headers = list(reader.fieldnames or [])
            rows = list(reader)

        if feature_cols is None:
            feature_cols = [
                h for h in headers
                if h not in ID_COLS and h != target_col
            ]

        self.feature_cols = feature_cols
        self.target_col = target_col

        # Parse features
        features = []
        labels = []
        for row in rows:
            feat = []
            for col in feature_cols:
                val = row.get(col, "").strip()
                feat.append(float(val) if val else 0.0)   # missing → 0
            features.append(feat)

            lbl = row.get(target_col, "").strip()
            labels.append(float(lbl) if lbl else float("nan"))

        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def build_dataloaders(
    cfg: dict,
    splits: dict[str, str] | None = None,
) -> dict[str, DataLoader]:
    """
    Build DataLoaders for train / val / test from config.

    Parameters
    ----------
    cfg : dict
        Loaded config dictionary.
    splits : dict, optional
        Mapping of split name → CSV path. If None, reads from
        ``data/processed/{train,val,test}.csv``.

    Returns
    -------
    dict[str, DataLoader]
    """
    processed_dir = Path(cfg["data"]["processed_dir"])
    batch_size = cfg["training"]["batch_size"]
    target_col = cfg["preprocessing"]["target_column"]

    if splits is None:
        splits = {
            s: str(processed_dir / f"{s}.csv")
            for s in ("train", "val", "test")
        }

    loaders = {}
    for name, path in splits.items():
        ds = MADealDataset(path, target_col=target_col)
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
        )
        print(f"  {name}: {len(ds)} samples, {ds.features.shape[1]} features")

    return loaders
