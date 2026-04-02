import streamlit as st
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import os

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
    del adata.layers
    return adata


if input_mode == "Use demo dataset (PBMC 3k)":
    adata = load_data()
    st.success(f"Data loaded: {adata.n_obs} cells, {adata.n_vars} genes")

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
    geo_id = st.text_input("Enter GEO Accession ID (e.g. GSE84133)")
    if geo_id:
        with st.spinner(f"Downloading {geo_id} from GEO..."):
            import sys
            pipeline_path = os.path.join(os.path.dirname(__file__), "..", "pipeline")
            if pipeline_path not in sys.path:
                sys.path.insert(0, pipeline_path)
            from ingest import download_geo, get_geo_tissue_hint, auto_select_celltypist_model
            result_path = download_geo(geo_id, output_dir="/tmp")
            if result_path.endswith(".h5ad"):
                adata = sc.read_h5ad(result_path)
            elif os.path.isdir(result_path):
                import glob
                mtx_files = glob.glob(f"{result_path}/**/*matrix.mtx*", recursive=True)
                if mtx_files:
                    mtx_dir = os.path.dirname(mtx_files[0])
                    adata = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", cache=True)
                else:
                    st.error("Could not find matrix file. Please upload manually.")
                    st.stop()
            else:
                st.error("Could not load data. Please upload manually.")
                st.stop()
            tissue_text = get_geo_tissue_hint(geo_id, output_dir="/tmp")
            st.session_state["celltypist_model"] = auto_select_celltypist_model(tissue_text)
        adata.obs_names_make_unique()
        st.session_state["adata_raw"] = adata.copy()
        st.success(f"Data loaded: {adata.n_obs} cells, {adata.n_vars} genes")
    else:
        st.info("Please enter a GEO Accession ID.")
        st.stop()

tab1, tab2, tab3 = st.tabs(["UMAP", "Cell Type Composition", "Marker Genes"])


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

st.sidebar.header("Analysis Parameters")
resolution = st.sidebar.slider("Clustering Resolution", 0.1, 2.0, 0.7, 0.1)
n_neighbors = st.sidebar.slider("N Neighbors", 5, 50, 10, 5)
max_pct_mt = st.sidebar.slider("Max Mitochondrial %", 1, 20, 5, 1)

run_button = st.sidebar.button("Re-run Analysis")


def needs_preprocessing(a):
    return "leiden" not in a.obs.columns or "X_umap" not in a.obsm

def run_preprocessing(a, n_neighbors, resolution, progress=None, status=None):
    def _update(pct, msg):
        if status: status.info(msg)
        if progress: progress.progress(pct)

    _update(5, "Preprocessing: normalizing...")
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)

    _update(15, "Preprocessing: selecting highly variable genes...")
    sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat")
    a = a[:, a.var.highly_variable].copy()

    _update(25, "Preprocessing: scaling & PCA...")
    a.raw = a  # save log1p normalized matrix for CellTypist
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a)

    _update(50, "Computing neighborhood graph...")
    sc.pp.neighbors(a, n_neighbors=n_neighbors, n_pcs=40)

    _update(70, "Computing UMAP...")
    sc.tl.umap(a)

    _update(85, "Running Leiden clustering...")
    sc.tl.leiden(a, resolution=resolution)

    _update(100, "Done!")
    return a


with tab1:
    st.subheader("UMAP - Cell Clusters")

    # Auto-preprocess if data has no leiden/umap (e.g. raw GEO data)
    if needs_preprocessing(adata) and "adata_run" not in st.session_state:
        st.info("Raw data detected — running preprocessing automatically...")
        progress = st.progress(0)
        status = st.empty()
        adata_run = run_preprocessing(adata.copy(), n_neighbors, resolution, progress, status)
        status.success("Preprocessing complete!")
        st.session_state["adata_run"] = adata_run

    if run_button:
        progress = st.progress(0)
        status = st.empty()

        if needs_preprocessing(adata):
            status.info("Step 1/5: Preprocessing raw data...")
            adata_run = run_preprocessing(adata.copy(), n_neighbors, resolution, progress, status)
        else:
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

            if "cell_type" in adata.obs.columns:
                adata_run.obs["cell_type"] = adata.obs["cell_type"]
            progress.progress(100)

        status.success("Analysis complete!")
        st.session_state["adata_run"] = adata_run

    if "adata_run" in st.session_state:
        adata_run = st.session_state["adata_run"]
    else:
        adata_run = adata

    st.info(f"Current clusters: {adata_run.obs['leiden'].nunique()}")

    color_options = ["Leiden Clusters (Scanpy)"]
    if "cell_type" in adata_run.obs.columns:
        color_options.append("Cell Type Annotation (CellTypist)")
    color_option = st.selectbox("Color by", color_options)
    color_by = "leiden" if "Leiden" in color_option else "cell_type"
    st.session_state["color_by"] = color_by

    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.umap(adata_run, color=color_by, legend_loc="on data", frameon=False, ax=ax, show=False)
    st.pyplot(fig)

    # CellTypist annotation section
    st.divider()
    st.subheader("Cell Type Annotation")
    selected_model = st.session_state.get("celltypist_model", "Immune_All_Low.pkl")
    st.caption(f"Auto-selected model: **{selected_model}**")
    if st.button("Run CellTypist Annotation"):
        with st.spinner("Running CellTypist annotation..."):
            import celltypist
            from celltypist import models
            models.download_models(model=selected_model, force_update=False)
            model = models.Model.load(model=selected_model)
            # CellTypist needs log1p normalized data — start from raw counts
            adata_ct = st.session_state.get("adata_raw", adata).copy()
            sc.pp.normalize_total(adata_ct, target_sum=1e4)
            sc.pp.log1p(adata_ct)
            predictions = celltypist.annotate(adata_ct, model=model, majority_voting=True)
            adata_run.obs["cell_type"] = predictions.predicted_labels["majority_voting"].values
            st.session_state["adata_run"] = adata_run
        st.success("Annotation complete! Select 'Cell type annotation' in Color by above.")
        st.rerun()


with tab2:
    st.subheader("Cell Type Composition")
    # Prefer cell_type annotation over leiden cluster numbers
    if "cell_type" in adata_run.obs.columns:
        count_col = "cell_type"
    else:
        count_col = "leiden"
    cell_counts = adata_run.obs[count_col].value_counts().reset_index()
    cell_counts.columns = ["Cell Type", "Count"]
    cell_counts["Percentage"] = (cell_counts["Count"] / cell_counts["Count"].sum() * 100).round(2)
    st.dataframe(cell_counts, use_container_width=True)
    st.bar_chart(cell_counts.set_index("Cell Type")["Count"])

with tab3:
    st.subheader("Top Marker Genes per Cluster")
    sample_var = str(adata_run.var_names[0]) if len(adata_run.var_names) > 0 else ""
    looks_like_barcode = any(p in sample_var for p in ["final_cell", "AAACCT", "barcode", ".final."])
    if looks_like_barcode:
        st.warning("Gene names were not detected in this dataset. Marker genes cannot be displayed.")
    else:
        # Use the same grouping as Color by selection
        groupby = st.session_state.get("color_by", "leiden")
        if groupby not in adata_run.obs.columns:
            groupby = "leiden"

        # Recompute if groupby changed or not yet computed
        current_groupby = adata_run.uns.get("rank_genes_groups", {}).get("params", {}).get("groupby")
        if current_groupby != groupby:
            with st.spinner(f"Computing marker genes by {groupby}..."):
                sc.tl.rank_genes_groups(adata_run, groupby, method="wilcoxon")
                st.session_state["adata_run"] = adata_run

        marker_df = pd.DataFrame(adata_run.uns["rank_genes_groups"]["names"]).head(10)
        st.dataframe(marker_df, use_container_width=True)
