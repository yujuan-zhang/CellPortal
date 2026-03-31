import streamlit as st
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI-Powered Single-Cell Analysis Platform", layout="wide")
st.title("AI-Powered Single-Cell Analysis Platform")
with st.container(border=True):
    st.subheader("Load Data")
    input_mode = st.radio(
        "Select input method",
        ["Use demo dataset (PBMC 3k)", "Upload .h5ad file", "Enter GEO Accession ID"]
    )
@st.cache_data
def load_data():
    import boto3
    import os
    s3 = boto3.client(
        "s3",
        aws_access_key_id=st.secrets["aws"]["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"],
        region_name=st.secrets["aws"]["AWS_DEFAULT_REGION"]
    )

    local_path = "/tmp/pbmc3k_annotated.h5ad"
    if not os.path.exists(local_path):
        s3.download_file("cellportal-data", "processed/pbmc3k_annotated.h5ad", local_path)
    adata = sc.read_h5ad(local_path)
    # 只保留必要数据，节省内存
    del adata.layers
    return adata

   

if input_mode == "Use demo dataset (PBMC 3k)":
    adata = load_data()

elif input_mode == "Upload .h5ad file":
    uploaded_file = st.file_uploader("Upload .h5ad file", type=["h5ad"])
    if uploaded_file is not None:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        adata = sc.read_h5ad(tmp_path)
        st.success(f"Data loaded: {adata.n_obs} cells, {adata.n_vars} genes")
    else:
        st.info("Please upload a .h5ad file to continue.")
        st.stop()

elif input_mode == "Enter GEO Accession ID":
    geo_id = st.text_input("Enter GEO Accession ID (e.g. GSE176078)")
    if geo_id:
        st.info(f"GEO download coming soon. Please upload the processed .h5ad file for {geo_id} manually for now.")
        st.stop()
    else:
        st.info("Please enter a GEO Accession ID.")
        st.stop()

st.success(f"Data loaded: {adata.n_obs} cells, {adata.n_vars} genes")

tab1, tab2, tab3 = st.tabs(["UMAP", "Cell Type Composition", "Marker Genes"])


# 侧边栏介绍
st.sidebar.title("About")
st.sidebar.markdown("""
An end-to-end scRNA-seq analysis platform powered by AI.

**AI Annotation Models:**
- ✅ CellTypist
- 🔜 scGPT *(Coming Soon)*
- 🔜 Geneformer *(Coming Soon)*

🌐 [GitHub](https://github.com/yujuan-zhang/CellPortal)
""")
st.sidebar.divider()

# 侧边栏参数
st.sidebar.header("Analysis Parameters")
resolution = st.sidebar.slider("Clustering Resolution", 0.1, 2.0, 0.7, 0.1)
n_neighbors = st.sidebar.slider("N Neighbors", 5, 50, 10, 5)
max_pct_mt = st.sidebar.slider("Max Mitochondrial %", 1, 20, 5, 1)

run_button = st.sidebar.button("Re-run Analysis")


with tab1:
    st.subheader("UMAP - Cell Clusters")
    if run_button:
        progress = st.progress(0)
        status = st.empty()

        status.info("Step 1/4: Copying data...")
        adata_run = adata.copy()
        progress.progress(10)

        status.info("Step 2/4: Computing neighborhood graph...")
        sc.pp.neighbors(adata_run, n_neighbors=n_neighbors, n_pcs=40)
        progress.progress(40)

        status.info("Step 3/4: Computing UMAP...")
        sc.tl.umap(adata_run)
        progress.progress(60)

        status.info("Step 4/4: Running Leiden clustering...")
        sc.tl.leiden(adata_run, resolution=resolution)
        progress.progress(80)

        adata_run.obs["cell_type"] = adata.obs["cell_type"]
        progress.progress(100)

        status.success("Analysis complete!")
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
