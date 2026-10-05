"""Count-data QC and size-aware Scanpy analysis used by the Streamlit app."""
import numpy as np
import scanpy as sc
from scipy import sparse


def needs_preprocessing(adata):
    has_clustering = 'leiden' in adata.obs or 'louvain' in adata.obs
    return not has_clustering or 'X_umap' not in adata.obsm


def _check_size(adata):
    if adata.n_obs < 3 or adata.n_vars < 2:
        raise ValueError('Analysis needs at least 3 retained cells and 2 genes. '
                         'Check your input and mitochondrial threshold.')


def filter_counts(adata, max_pct_mt):
    """Apply the UI threshold to raw counts without mutating the input."""
    if not 0 <= max_pct_mt <= 100:
        raise ValueError('The mitochondrial threshold must be between 0 and 100.')
    a = adata.copy()
    values = a.X.data if sparse.issparse(a.X) else np.asarray(a.X)
    if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values, np.rint(values)):
        raise ValueError('Unprocessed input X must contain finite, nonnegative raw counts. '
                         'Do not pass scaled or log-normalized values as raw counts.')
    before = a.n_obs
    totals = np.asarray(a.X.sum(axis=1)).ravel()
    a = a[totals > 0].copy()
    _check_size(a)
    # Gene IDs may be Ensembl IDs if the supplied symbol column is available.
    if 'mt' in a.var:
        if not a.var['mt'].isin([True, False]).all() or a.var['mt'].isna().any():
            raise ValueError('var["mt"] must contain Boolean mitochondrial annotations.')
        a.var['mt'] = a.var['mt'].astype(bool)
    else:
        symbols = a.var_names
        for column in ['gene_symbols', 'gene_names']:
            if column in a.var:
                symbols = a.var[column].astype(str)
                break
        a.var['mt'] = np.asarray([str(symbol).upper().startswith('MT-') for symbol in symbols])
    if not a.var['mt'].any():
        raise ValueError('No mitochondrial genes were identified. Supply MT- gene symbols '
                         'or an explicit Boolean var["mt"] annotation.')
    sc.pp.calculate_qc_metrics(a, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    a = a[a.obs['pct_counts_mt'] <= max_pct_mt].copy()
    _check_size(a)
    sc.pp.filter_genes(a, min_cells=1)
    _check_size(a)
    a.uns['cellportal_qc'] = {
        'input_cells': before, 'retained_cells': a.n_obs,
        'removed_cells': before - a.n_obs, 'max_pct_mt': float(max_pct_mt),
    }
    return a


def _gene_variance(a):
    if sparse.issparse(a.X):
        mean = np.asarray(a.X.mean(axis=0)).ravel()
        return np.maximum(np.asarray(a.X.multiply(a.X).mean(axis=0)).ravel() - mean ** 2, 0)
    return np.asarray(a.X).var(axis=0)


def _neighbors(a, n_neighbors):
    _check_size(a)
    if 'X_pca' not in a.obsm:
        sc.tl.pca(a, n_comps=min(40, a.n_obs - 1, a.n_vars - 1), svd_solver='arpack')
    available_pcs = a.obsm['X_pca'].shape[1]
    if available_pcs < 1:
        raise ValueError('No PCA dimensions are available for clustering.')
    sc.pp.neighbors(a, n_neighbors=min(int(n_neighbors), a.n_obs - 1),
                    n_pcs=min(40, available_pcs))


def run_preprocessing(adata, n_neighbors, resolution, max_pct_mt, progress=None, status=None):
    def update(pct, message):
        if status is not None:
            status.info(message)
        if progress is not None:
            progress.progress(pct)

    update(5, 'Filtering zero-count cells and mitochondrial counts...')
    a = filter_counts(adata, max_pct_mt)
    update(10, 'Normalizing retained counts...')
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    # Preserve every retained gene for annotation and marker analysis.
    a.raw = a.copy()
    update(15, 'Selecting highly variable genes...')
    variance = _gene_variance(a)
    variable = np.flatnonzero(variance > 0)
    if len(variable) < 2:
        raise ValueError('At least 2 genes must vary across the retained cells.')
    try:
        sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor='seurat')
        selected = np.flatnonzero(a.var['highly_variable'].to_numpy() & (variance > 0))
    except ValueError:
        selected = np.array([], dtype=int)
    if len(selected) < 2:
        selected = variable[np.argsort(variance[variable])[-2000:]]
    a = a[:, selected].copy()
    update(25, 'Scaling and computing size-aware PCA...')
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=min(40, a.n_obs - 1, a.n_vars - 1), svd_solver='arpack')
    update(50, 'Computing neighborhood graph...')
    _neighbors(a, n_neighbors)
    update(70, 'Computing UMAP...')
    sc.tl.umap(a, init_pos='random' if a.n_obs <= 3 else 'spectral')
    update(85, 'Running Leiden clustering...')
    sc.tl.leiden(a, resolution=resolution)
    update(100, 'Done!')
    return a


def recluster_processed(adata, n_neighbors, resolution):
    """Recluster existing representations; do not invent counts for QC."""
    a = adata.copy()
    _neighbors(a, n_neighbors)
    sc.tl.umap(a, init_pos='random' if a.n_obs <= 3 else 'spectral')
    sc.tl.leiden(a, resolution=resolution)
    return a
