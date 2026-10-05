"""Run a small synthetic raw-count example and verify its QC result."""
from pathlib import Path
import json
import sys
import time
import numpy as np
import pandas as pd
from anndata import AnnData

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'app'))
from analysis import run_preprocessing


def main():
    start = time.monotonic()
    counts = pd.read_csv(HERE / 'example_input.csv', index_col='cell_id')
    a = AnnData(counts.astype(float))
    result = run_preprocessing(a, n_neighbors=10, resolution=0.7, max_pct_mt=5)
    summary = dict(result.uns['cellportal_qc'])
    summary['retained_cell_ids'] = result.obs_names.tolist()
    expected = json.loads((HERE / 'expected_output.json').read_text())
    if summary != expected:
        raise AssertionError(f'Unexpected QC result: {summary}')
    if not np.isfinite(result.obsm['X_umap']).all() or 'leiden' not in result.obs:
        raise AssertionError('Clustering or UMAP failed.')
    output = HERE / 'output'
    output.mkdir(exist_ok=True)
    result.write_h5ad(output / 'example_analysis.h5ad')
    (output / 'qc_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'PASS: retained {result.n_obs} cells; results in {output}; {time.monotonic() - start:.2f}s')


if __name__ == '__main__':
    main()
