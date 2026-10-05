import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from analysis import filter_counts, run_preprocessing, recluster_processed


def example(sparse_matrix=False):
    df = pd.read_csv(ROOT / 'examples/example_input.csv', index_col='cell_id')
    a = AnnData(df.astype(float))
    if sparse_matrix:
        a.X = sparse.csr_matrix(a.X)
    return a


class AnalysisTests(unittest.TestCase):
    def test_dense_and_sparse_filtering_preserves_input(self):
        for use_sparse in [False, True]:
            with self.subTest(sparse=use_sparse):
                a = example(use_sparse)
                before = a.X.copy()
                result = filter_counts(a, 5)
                self.assertEqual(result.obs_names.tolist(), [f'cell{i:02}' for i in range(1, 10)])
                self.assertTrue((result.obs.pct_counts_mt <= 5).all())
                self.assertEqual(a.n_obs, 12)
                np.testing.assert_array_equal(a.X.toarray() if use_sparse else a.X,
                                              before.toarray() if use_sparse else before)
                self.assertNotIn('pct_counts_mt', a.obs)

    def test_threshold_boundary(self):
        a = AnnData(np.array([[5,95], [6,94], [0,100], [1,99], [2,98], [0,0]], dtype=float))
        a.var_names = ['MT-CO1', 'GeneA']
        result = filter_counts(a, 5)
        self.assertEqual(result.obs_names.tolist(), ['0', '2', '3', '4'])

    def test_invalid_input_and_too_strict_filter(self):
        a = example()
        a.X[0, 0] = -1
        with self.assertRaisesRegex(ValueError, 'raw counts'):
            filter_counts(a, 5)
        a = example()
        a.var_names = [f'ENSG{i}' for i in range(a.n_vars)]
        with self.assertRaisesRegex(ValueError, 'No mitochondrial'):
            filter_counts(a, 5)
        with self.assertRaisesRegex(ValueError, 'at least 3'):
            filter_counts(example(), 0)

    def test_symbol_column_and_explicit_annotation(self):
        a = example(True)
        symbols = a.var_names.copy()
        a.var_names = [f'ENSG{i}' for i in range(a.n_vars)]
        a.var['gene_symbols'] = symbols
        self.assertEqual(filter_counts(a, 5).n_obs, 9)
        a.var.drop(columns='gene_symbols', inplace=True)
        a.var['mt'] = [True, False, False, False, False, False]
        self.assertEqual(filter_counts(a, 5).n_obs, 9)

    def test_preprocessing_keeps_annotation_expression_and_adapts_pca(self):
        result = run_preprocessing(example(True), 50, 0.7, 5)
        self.assertEqual(result.n_obs, 9)
        self.assertEqual(result.raw.n_vars, 6)
        self.assertEqual(result.raw.obs_names.tolist(), result.obs_names.tolist())
        self.assertLessEqual(result.obsm['X_pca'].shape[1], 5)
        self.assertTrue(np.isfinite(result.obsm['X_umap']).all())
        self.assertIn('leiden', result.obs)
        # Processed subsets can have fewer cells than the original PCA dimensions.
        small = recluster_processed(result[:3].copy(), 50, 0.7)
        self.assertEqual(small.n_obs, 3)
        self.assertTrue(np.isfinite(small.obsm['X_umap']).all())


if __name__ == '__main__':
    unittest.main()
