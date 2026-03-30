import streamlit as st
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI-Powered Single-Cell Analysis Platform", layout="wide")
st.title("AI-Powered Single-Cell Analysis Platform")

@st.cache_data
def load_data():
    
    adata = sc.read_h5ad("scanpy-tutorials-main/write/pbmc3k_annotated.h5ad")

    return adata

adata = load_data()
st.success(f"Data loaded: {adata.n_obs} cells, {adata.n_vars} genes")

tab1, tab2, tab3 = st.tabs(["UMAP", "Cell Type Composition", "Marker Genes"])


# 侧边栏参数
st.sidebar.header("Analysis Parameters")
resolution = st.sidebar.slider("Clustering Resolution", 0.1, 2.0, 0.7, 0.1)
n_neighbors = st.sidebar.slider("N Neighbors", 5, 50, 10, 5)
max_pct_mt = st.sidebar.slider("Max Mitochondrial %", 1, 20, 5, 1)

run_button = st.sidebar.button("Re-run Analysis")


with tab1:
    st.subheader("UMAP - Cell Clusters")
    if run_button:
        with st.spinner("Re-running clustering and annotation..."):
            adata_run = adata.copy()
            sc.pp.neighbors(adata_run, n_neighbors=n_neighbors, n_pcs=40)
            sc.tl.umap(adata_run)
            sc.tl.leiden(adata_run, resolution=resolution)

            import celltypist
            from celltypist import models
            adata_pred = adata_run.copy()
            sc.pp.normalize_total(adata_pred, target_sum=1e4)
            sc.pp.log1p(adata_pred)
            model = models.Model.load(model="Immune_All_Low.pkl")
            predictions = celltypist.annotate(adata_pred, model=model, majority_voting=True)
            adata_run.obs["cell_type"] = predictions.predicted_labels["majority_voting"]
            st.session_state["adata_run"] = adata_run

    if "adata_run" in st.session_state:
        adata_run = st.session_state["adata_run"]
    else:
        adata_run = adata

    st.info(f"Current clusters: {adata_run.obs['leiden'].nunique()}")
    color_option = st.selectbox("Color by", ["Manual Annotation cell type (scanpy)", "AI Annotation (CellTypist)"])
    color_by = "leiden" if "scanpy" in color_option else "cell_type"
    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.umap(adata_run, color=color_by, legend_loc="on data", frameon=False, ax=ax, show=False)
    st.pyplot(fig)



with tab2:
    st.subheader("Cell Type Composition")
    cell_counts = adata.obs["leiden"].value_counts().reset_index()
    cell_counts.columns = ["Cell Type", "Count"]
    cell_counts["Percentage"] = (cell_counts["Count"] / cell_counts["Count"].sum() * 100).round(2)
    st.dataframe(cell_counts, use_container_width=True)
    st.bar_chart(cell_counts.set_index("Cell Type")["Count"])

with tab3:
    st.subheader("Top Marker Genes per Cluster")
    marker_df = pd.DataFrame(adata.uns["rank_genes_groups"]["names"]).head(10)
    st.dataframe(marker_df, use_container_width=True)
