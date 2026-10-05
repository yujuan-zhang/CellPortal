# AI-Powered Single-Cell Analysis Platform

## What it does

Explore single-cell RNA-seq data in a browser: filter low-quality cells,
find cell clusters, inspect marker genes and, optionally, assign cell types.
The starting point is a gene-expression matrix in AnnData format, not raw
sequencing reads. You can use the PBMC demo or upload a compatible `.h5ad` file.

For raw counts, the analysis follows this route:

```text
Counts → cell-quality filtering → normalization and log1p
       → variable genes and PCA → neighbor graph → Leiden clusters and UMAP
       → marker tables and optional cell-type annotation
```

The app exposes QC and clustering settings so you can inspect how they affect
the retained cells and clusters. Already processed inputs follow a separate
reclustering route rather than repeating raw-count QC. Results are explored
in the UMAP, composition and marker views; the small command-line example
checks the preprocessing steps on synthetic counts.

Optional CellTypist annotation uses a pretrained external model. It does not
train a new annotation model by default. Chat needs separate AWS Bedrock
configuration; it is not needed for the expression-analysis example. GEO,
annotation and chat paths are separate from the verified local start path.

🌐 **Live Demo**: https://singlecell-ai.streamlit.app/

## Input

**Input:** `.h5ad` AnnData with cells in rows and genes in columns. Unprocessed inputs must contain finite, nonnegative integer counts in `X`. Mitochondrial genes must have `MT-`/`mt-` symbols in `var_names`, `var["gene_symbols"]` or `var["gene_names"]`, or an explicit Boolean `var["mt"]` annotation. Missing mitochondrial annotations cause a clear error rather than silently assuming zero mitochondrial counts.

## Output

**Output:** on-screen UMAP, cell composition and marker tables. The current app does not provide a dedicated full-result `.h5ad` export or per-cell prediction CSV download. Do not expect files to appear just by viewing the tabs.

## Try it

### Local quick start

From the project root, use Python 3.11 in a dedicated environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app/app.py
```

Open `http://localhost:8501` and keep **Use demo dataset (PBMC 3k)** selected. A local `data/pbmc3k_annotated.h5ad` is used when present; otherwise Scanpy downloads its processed PBMC demo. The local example showed 2,638 cells and 32,738 genes, UMAP and a composition table in about 29 seconds (Python 3.11, installed dependencies; includes a cold font cache). Cached or downloaded demos can have different gene counts. Downloads and installation are excluded.

On a raw-data run, zero-count cells are removed and cells above **Max Mitochondrial %** are filtered before normalization. The displayed QC summary reports retained cells and the applied threshold. Changes take effect when **Re-run Analysis** is clicked. At least 3 cells and 2 variable genes must remain; PCA and neighborhood sizes adapt to the retained data instead of requiring 40 PCs. The threshold is a user setting, not a universal biological QC recommendation.

If `obs` already contains `leiden` or `louvain` and `obsm` contains `X_umap`, the app treats the data as processed. The mitochondrial control is disabled for that data; re-running updates clustering without reconstructing raw-count QC. Annotation requires log1p-normalized expression in `adata.raw`; raw-count preprocessing saves this matrix before variable-gene selection and scaling.

CellTypist annotation is optional and downloads a selected model on first use. Chat requires configured AWS Bedrock access in Streamlit secrets; the demo analysis works without invoking chat. scGPT, Geneformer and FASTQ processing are not part of this verified start path. The local demo and small synthetic dense/sparse count examples are checked separately. GEO ingestion, CellTypist model execution and arbitrary uploaded datasets remain unverified.

#### Verify the raw-count example

```bash
python examples/run_example.py
python -m unittest discover -s tests -v
```

The included `examples/example_input.csv` contains 12 synthetic cells (rows) and 6 gene-symbol columns. It is an execution fixture, not a biological dataset. The example runs the same preprocessing as the app at a 5% mitochondrial threshold, writes `examples/output/example_analysis.h5ad` and a QC summary, and checks `examples/expected_output.json`. Two high-mitochondrial cells and one zero-count cell should be removed, leaving 9 cells. Cluster labels and UMAP coordinates are not used as exact reference values. No model download, AWS access or GEO request is needed.

The synthetic analysis took about 8 seconds on the local CPU with dependencies and a JIT cache available (about 13 seconds including Python startup). Initial compilation, installation and larger datasets take additional time. This is a functional check, not validation of biological cluster quality.

### Features

- **Automated Pipeline**: Standardized QC, normalization, dimensionality reduction, and clustering powered by Scanpy
- **AI Cell Annotation**: Integrates multiple AI models for automatic cell type identification
  - CellTypist — immune cell classification with 40+ pretrained models
  - scGPT — large language model for single-cell biology *(Coming Soon)*
  - Geneformer — gene expression perturbation prediction *(Coming Soon)*
- **Interactive Visualization**: UMAP plots, cell type composition charts, and marker gene heatmaps
- **Parameter Control**: Dynamically adjust clustering resolution, neighbor count, and QC thresholds with real-time updates
- **Dual Annotation Comparison**: Side-by-side comparison of manual annotation (Scanpy) vs. AI annotation (CellTypist)
- **Cloud-Native**: Data stored on AWS S3, deployed on Streamlit Cloud

---

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Downstream Analysis | Scanpy, AnnData |
| AI Annotation | CellTypist, scGPT, Geneformer |
| Frontend | Streamlit |
| Cloud Storage | AWS S3 |
| Containerization | Docker *(Coming Soon)* |
| Orchestration | Apache Airflow *(Coming Soon)* |
| MLOps | MLflow *(Coming Soon)* |
| Deployment | AWS ECS / Fargate *(Coming Soon)* |

---

### Architecture

```
Raw Data (FASTQ / Count Matrix / h5ad)
    ↓
Data Ingestion (GEO / SRA / Internal)
    ↓
Upstream Analysis (Nextflow + STARsolo)     ← Coming Soon
    ↓
Downstream Analysis (Scanpy Pipeline)
    ├── QC & Filtering
    ├── Normalization
    ├── Dimensionality Reduction (PCA + UMAP)
    └── Clustering (Leiden)
    ↓
AI Annotation
    ├── CellTypist (Active)
    ├── scGPT (Coming Soon)
    └── Geneformer (Coming Soon)
    ↓
Interactive Visualization (Streamlit)
```

---

### Demo Dataset

[PBMC 3k](https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/pbmc3k) from 10x Genomics — 2,638 peripheral blood mononuclear cells from a healthy donor.

---

### Project Structure

```
CellPortal/
├── pipeline/
│   ├── ingest.py             # Data ingestion and input routing
│   └── annotate_scgpt.py     # scGPT cell type annotation
├── app/
│   ├── app.py                # Streamlit frontend
│   └── analysis.py           # Count QC and size-aware preprocessing
├── Dockerfile
├── requirements.txt
├── runtime.txt
└── README.md
```

---

### Roadmap

- [x] Scanpy downstream pipeline
- [x] CellTypist AI annotation
- [x] Streamlit interactive frontend
- [x] AWS S3 cloud storage
- [ ] scGPT integration
- [ ] Geneformer integration
- [ ] Nextflow NGS upstream pipeline
- [ ] Docker containerization
- [ ] AWS ECS deployment
- [ ] MLflow experiment tracking
- [ ] Airflow pipeline orchestration

