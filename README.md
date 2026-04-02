# AI-Powered Single-Cell Analysis Platform

A end-to-end single-cell RNA sequencing (scRNA-seq) analysis platform that automates the full workflow from raw data ingestion to cell type annotation and interactive visualization.

🌐 **Live Demo**: https://singlecell-ai.streamlit.app/

---

## Features

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

## Tech Stack

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

## Architecture

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

## Demo Dataset

[PBMC 3k](https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/pbmc3k) from 10x Genomics — 2,638 peripheral blood mononuclear cells from a healthy donor.

---

## Project Structure

```
CellPortal/
├── pipeline/
│   ├── preprocess.py     # Scanpy QC, normalization, clustering
│   ├── annotate.py       # CellTypist cell type annotation
│   └── router.py         # Input routing (FASTQ / matrix / h5ad)
├── app/
│   └── app.py            # Streamlit frontend
├── requirements.txt
└── README.md
```

---

## Roadmap

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
