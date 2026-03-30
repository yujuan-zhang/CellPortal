from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
def run_preprocess(input_path: str, output_path: str):
    
    sc.settings.verbosity = 3
    sc.set_figure_params(dpi=80, facecolor="white")



    adata = sc.read_10x_mtx(
        input_path,
        var_names="gene_symbols",
        cache=True,
    )

    adata.var_names_make_unique()

    # QC
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    # Filtering
    adata = adata[
        (adata.obs.n_genes_by_counts < 2500) & (adata.obs.n_genes_by_counts > 200) & (adata.obs.pct_counts_mt < 5),
        :,
    ].copy()
    adata.layers["counts"] = adata.X.copy()

    # Normalization
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Highly variable genes
    sc.pp.highly_variable_genes(
        adata,
        layer="counts",
        n_top_genes=2000,
        min_mean=0.0125,
        max_mean=3,
        min_disp=0.5,
        flavor="seurat_v3",
    )

    # Scaling
    adata.layers["scaled"] = adata.X.toarray()
    sc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"], layer="scaled")
    sc.pp.scale(adata, max_value=10, layer="scaled")

    # PCA
    sc.pp.pca(adata, layer="scaled", svd_solver="arpack")

    # Neighborhood graph
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

    # UMAP
    sc.tl.umap(adata)

    # Clustering
    sc.tl.leiden(
        adata,
        resolution=0.7,
        random_state=0,
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )
    adata.obs["leiden"] = adata.obs["leiden"].copy()
    adata.uns["leiden"] = adata.uns["leiden"].copy()
    adata.obsm["X_umap"] = adata.obsm["X_umap"].copy()

    # Marker genes
    sc.tl.rank_genes_groups(adata, "leiden", mask_var="highly_variable", method="wilcoxon")

    # Cell type annotation
    new_cluster_names = [
        "CD4 T",
        "B",
        "CD14+ Monocytes",
        "NK",
        "CD8 T",
        "FCGR3A+ Monocytes",
        "Dendritic",
        "Megakaryocytes",
    ]
    adata.rename_categories("leiden", new_cluster_names)

    # Save
    adata.write(output_path, compression="gzip")
    print(f"Done! Saved to {output_path}")

if __name__ == "__main__":
    run_preprocess(
        input_path="data/filtered_gene_bc_matrices/hg19/",
        output_path="write/pbmc3k.h5ad"
    )
