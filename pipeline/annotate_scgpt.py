from __future__ import annotations

import os
if not hasattr(os, 'sched_getaffinity'):
    os.sched_getaffinity = lambda x: {0}

import torch.utils.data
_original_dataloader = torch.utils.data.DataLoader.__init__

def _patched_dataloader(self, *args, **kwargs):
    kwargs['num_workers'] = 0
    _original_dataloader(self, *args, **kwargs)

torch.utils.data.DataLoader.__init__ = _patched_dataloader

import scanpy as sc
import numpy as np
import torch
from pathlib import Path


def run_scgpt_embedding(input_path: str, output_path: str, model_dir: str = "models/scGPT_human"):
    from scgpt.tasks import embed_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    adata = sc.read_h5ad(input_path)
    print(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")

    # 用 scGPT 生成细胞嵌入
    adata = embed_data(
        adata,
        model_dir=model_dir,
        gene_col="index",
        batch_size=64,
        device=device,
        use_fast_transformer=False,
    )

    # 用 scGPT 嵌入重新做 UMAP
    sc.pp.neighbors(adata, use_rep="X_scGPT")
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.7, key_added="leiden_scgpt")

    adata.write_h5ad(output_path, compression="gzip")
    print(f"Done! Saved to {output_path}")


if __name__ == "__main__":
    run_scgpt_embedding(
        input_path="data/pbmc3k_annotated.h5ad",
        output_path="data/pbmc3k_scgpt.h5ad"
    )
