"""
Block B — FinBERT text embeddings.

Extracts semantic representations from acquirer 10-K filings (MD&A and
Risk Factors sections) using a frozen FinBERT model.

Pipeline:
  1. Load extracted .txt sections from EDGAR filings
  2. Chunk long text into 512-token windows (stride=256)
  3. Extract [CLS] from penultimate layer for each chunk
  4. Mean-pool across chunks → 768-dim vector per section
  5. PCA reduce 768 → N dims (default 64)
  6. Output: text_embeddings.csv

This module is applicable to the US/EDGAR subsample only.
"""

import csv
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ═════════════════════════════════════════════════════════════════════════════════
# FinBERT Embedder
# ═════════════════════════════════════════════════════════════════════════════════

class FinBERTEmbedder:
    """
    Frozen FinBERT feature extractor.

    Loads ProsusAI/finbert, tokenises input text into overlapping 512-token
    chunks, extracts the [CLS] token from the penultimate transformer layer,
    and mean-pools across chunks to produce a single 768-dim vector.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        max_length: int = 512,
        stride: int = 256,
        device: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model identifier.
        max_length : int
            Max tokens per chunk.
        stride : int
            Overlap between consecutive chunks (in tokens).
        device : str, optional
            'cuda', 'mps', or 'cpu'. Auto-detected if None.
        """
        import torch
        from transformers import AutoTokenizer, AutoModel

        # Device selection
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        logger.info(f"Loading FinBERT ({model_name}) on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, output_hidden_states=True
        )
        self.model.eval()
        self.model.to(device)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.max_length = max_length
        self.stride = stride
        self.embedding_dim = self.model.config.hidden_size  # 768

        logger.info(
            f"FinBERT ready: {self.embedding_dim}-dim embeddings, "
            f"chunk={max_length}, stride={stride}, device={device}"
        )

    def embed_text(self, text: str) -> np.ndarray:
        """
        Extract a single embedding vector from a (potentially long) text.

        Steps:
          1. Tokenise the full text
          2. Split into overlapping chunks of max_length tokens
          3. For each chunk: extract [CLS] from penultimate layer
          4. Mean-pool across all chunk embeddings

        Parameters
        ----------
        text : str
            Input text (e.g. full MD&A section, can be very long).

        Returns
        -------
        np.ndarray
            768-dimensional embedding vector.
        """
        import torch

        if not text or not text.strip():
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Tokenise the full text (no truncation yet)
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors=None,
        )
        all_ids = encoded["input_ids"]

        # Build overlapping chunks
        chunks = []
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        usable_len = self.max_length - 2  # reserve space for [CLS] and [SEP]

        if len(all_ids) <= usable_len:
            # Short text — single chunk
            chunks.append([cls_id] + all_ids + [sep_id])
        else:
            # Sliding window
            for start in range(0, len(all_ids), self.stride):
                end = start + usable_len
                chunk_ids = [cls_id] + all_ids[start:end] + [sep_id]
                chunks.append(chunk_ids)
                if end >= len(all_ids):
                    break

        # Extract [CLS] embedding from each chunk
        cls_embeddings = []

        for chunk_ids in chunks:
            # Pad to max_length
            attention_mask = [1] * len(chunk_ids)
            padding = self.max_length - len(chunk_ids)
            if padding > 0:
                chunk_ids = chunk_ids + [self.tokenizer.pad_token_id or 0] * padding
                attention_mask = attention_mask + [0] * padding

            input_ids = torch.tensor([chunk_ids], device=self.device)
            attn_mask = torch.tensor([attention_mask], device=self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attn_mask)

            # Penultimate layer = hidden_states[-2]
            # [CLS] token = index 0
            penultimate = outputs.hidden_states[-2]  # (1, seq_len, 768)
            cls_vec = penultimate[0, 0, :].cpu().numpy()  # (768,)
            cls_embeddings.append(cls_vec)

        # Mean-pool across chunks
        embedding = np.mean(cls_embeddings, axis=0).astype(np.float32)
        return embedding


# ═════════════════════════════════════════════════════════════════════════════════
# Batch Processing
# ═════════════════════════════════════════════════════════════════════════════════

def _load_deal_texts(
    download_log_path: Path,
    filings_dir: Path,
) -> list[dict]:
    """
    Load deal keys and their extracted text file paths from the download log.

    Returns list of dicts with: deal_key, ticker, announce_date, cik,
    item_7_path (or None), item_1a_path (or None).
    """
    deals = []
    with open(download_log_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "success":
                continue

            accession = row.get("accession", "").replace("-", "")
            cik = row.get("cik", "")
            deal_dir = filings_dir / str(cik) / accession

            item7 = deal_dir / "item_7_mda.txt"
            item1a = deal_dir / "item_1a_risk.txt"

            deals.append({
                "deal_key": row.get("deal_key", ""),
                "ticker": row.get("ticker", ""),
                "announce_date": row.get("announce_date", ""),
                "cik": cik,
                "accession": accession,
                "item_7_path": item7 if item7.exists() else None,
                "item_1a_path": item1a if item1a.exists() else None,
            })

    return deals


def build_text_embeddings(
    embedder: FinBERTEmbedder,
    download_log_path: Optional[Path] = None,
    filings_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    limit: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Batch-extract FinBERT embeddings for all deals with text data.

    Parameters
    ----------
    embedder : FinBERTEmbedder
        Initialised embedder instance.
    download_log_path : Path, optional
        Path to download_log.csv.
    filings_dir : Path, optional
        Root filings directory.
    output_path : Path, optional
        Where to save raw embeddings (.npz).
    limit : int, optional
        Max deals to process (for testing).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[dict]]
        (mda_embeddings, rf_embeddings, deal_metadata)
        Shapes: (N, 768), (N, 768), list of N dicts.
    """
    edgar_dir = PROJECT_ROOT / "data" / "external" / "edgar"
    if download_log_path is None:
        download_log_path = edgar_dir / "download_log.csv"
    if filings_dir is None:
        filings_dir = edgar_dir / "filings"
    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "processed" / "raw_embeddings.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load deals with text
    all_deals = _load_deal_texts(download_log_path, filings_dir)
    deals_with_text = [
        d for d in all_deals
        if d["item_7_path"] is not None or d["item_1a_path"] is not None
    ]

    if limit:
        deals_with_text = deals_with_text[:limit]

    n = len(deals_with_text)
    dim = embedder.embedding_dim
    print(f"Processing {n} deals with text data...")

    mda_embeddings = np.zeros((n, dim), dtype=np.float32)
    rf_embeddings = np.zeros((n, dim), dtype=np.float32)

    for i, deal in enumerate(deals_with_text):
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{i+1}/{n}] {deal['ticker']} ({deal['announce_date']})")

        # MD&A embedding
        if deal["item_7_path"]:
            try:
                text = deal["item_7_path"].read_text(encoding="utf-8", errors="replace")
                mda_embeddings[i] = embedder.embed_text(text)
            except Exception as e:
                logger.warning(f"Error embedding MD&A for {deal['ticker']}: {e}")

        # Risk Factors embedding
        if deal["item_1a_path"]:
            try:
                text = deal["item_1a_path"].read_text(encoding="utf-8", errors="replace")
                rf_embeddings[i] = embedder.embed_text(text)
            except Exception as e:
                logger.warning(f"Error embedding RF for {deal['ticker']}: {e}")

    # Save raw embeddings
    metadata_for_save = [
        {"deal_key": d["deal_key"], "ticker": d["ticker"],
         "announce_date": d["announce_date"], "cik": d["cik"]}
        for d in deals_with_text
    ]

    np.savez_compressed(
        output_path,
        mda=mda_embeddings,
        rf=rf_embeddings,
    )
    # Save metadata alongside
    meta_path = output_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(metadata_for_save, f, indent=2)

    print(f"✅ Raw embeddings saved → {output_path} ({n} deals × {dim} dims)")
    return mda_embeddings, rf_embeddings, deals_with_text


# ═════════════════════════════════════════════════════════════════════════════════
# PCA Dimensionality Reduction
# ═════════════════════════════════════════════════════════════════════════════════

def reduce_dimensions(
    mda_embeddings: np.ndarray,
    rf_embeddings: np.ndarray,
    deal_metadata: list[dict],
    n_components: int = 64,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    PCA-reduce raw 768-dim embeddings and save as CSV.

    Fits PCA on deals where the embedding is non-zero (has actual text),
    transforms all embeddings, and writes text_embeddings.csv.

    Parameters
    ----------
    mda_embeddings : np.ndarray
        (N, 768) raw MD&A embeddings.
    rf_embeddings : np.ndarray
        (N, 768) raw Risk Factors embeddings.
    deal_metadata : list[dict]
        Deal info (deal_key, ticker, announce_date).
    n_components : int
        Number of PCA dimensions per section. Total = 2 * n_components.
    output_dir : Path, optional
        Output directory. Defaults to data/processed/.

    Returns
    -------
    Path
        Path to text_embeddings.csv.
    """
    from sklearn.decomposition import PCA

    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(deal_metadata)

    # ── PCA for MD&A ────────────────────────────────────────────────
    # Only fit on deals where we have actual embeddings (non-zero rows)
    mda_mask = np.any(mda_embeddings != 0, axis=1)
    mda_valid = mda_embeddings[mda_mask]

    if len(mda_valid) > n_components:
        pca_mda = PCA(n_components=n_components, random_state=42)
        pca_mda.fit(mda_valid)
        mda_reduced = np.zeros((n, n_components), dtype=np.float32)
        mda_reduced[mda_mask] = pca_mda.transform(mda_valid)
        mda_var = sum(pca_mda.explained_variance_ratio_) * 100
        print(f"  MD&A PCA: {len(mda_valid)} deals, "
              f"{n_components} components, {mda_var:.1f}% variance explained")
    else:
        logger.warning(f"Not enough MD&A embeddings ({len(mda_valid)}) for PCA")
        pca_mda = None
        mda_reduced = mda_embeddings[:, :n_components]
        mda_var = 0

    # ── PCA for Risk Factors ──────────────────────────────────────────
    rf_mask = np.any(rf_embeddings != 0, axis=1)
    rf_valid = rf_embeddings[rf_mask]

    if len(rf_valid) > n_components:
        pca_rf = PCA(n_components=n_components, random_state=42)
        pca_rf.fit(rf_valid)
        rf_reduced = np.zeros((n, n_components), dtype=np.float32)
        rf_reduced[rf_mask] = pca_rf.transform(rf_valid)
        rf_var = sum(pca_rf.explained_variance_ratio_) * 100
        print(f"  RF PCA:  {len(rf_valid)} deals, "
              f"{n_components} components, {rf_var:.1f}% variance explained")
    else:
        logger.warning(f"Not enough RF embeddings ({len(rf_valid)}) for PCA")
        pca_rf = None
        rf_reduced = rf_embeddings[:, :n_components]
        rf_var = 0

    # ── Save PCA models ────────────────────────────────────────────
    pca_path = output_dir / "pca_models.pkl"
    with open(pca_path, "wb") as f:
        pickle.dump({"pca_mda": pca_mda, "pca_rf": pca_rf}, f)
    print(f"  PCA models saved → {pca_path}")

    # ── Write text_embeddings.csv ──────────────────────────────────
    output_csv = output_dir / "text_embeddings.csv"

    mda_cols = [f"mda_pca_{i}" for i in range(n_components)]
    rf_cols = [f"rf_pca_{i}" for i in range(n_components)]
    header = ["deal_key", "ticker", "announce_date", "has_mda", "has_rf"] + mda_cols + rf_cols

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for i, deal in enumerate(deal_metadata):
            row = [
                deal["deal_key"],
                deal["ticker"],
                deal["announce_date"],
                bool(mda_mask[i]),
                bool(rf_mask[i]),
            ]
            row.extend(mda_reduced[i].tolist())
            row.extend(rf_reduced[i].tolist())
            writer.writerow(row)

    print(f"✅ Text embeddings saved → {output_csv}")
    print(f"   {n} deals × {2 * n_components} dims (MD&A:{n_components} + RF:{n_components})")
    return output_csv
