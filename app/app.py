import streamlit as st
import traceback
import os
import sys
import hashlib
sys.path.insert(0, os.path.dirname(__file__))
import chat

st.set_page_config(page_title="AI-Powered Single-Cell Analysis Platform", layout="wide")

try:
    import scanpy as sc
    from analysis import needs_preprocessing, run_preprocessing, recluster_processed
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as _import_err:
    st.error(f"Import failed: {_import_err}")
    st.code(traceback.format_exc())
    st.stop()

st.title("AI-Powered Single-Cell Analysis Platform")
with st.container(border=True):
    st.subheader("Load Data")
    input_mode = st.radio(
        "Select input method",
        ["Use demo dataset (PBMC 3k)", "Upload .h5ad file", "Enter GEO Accession ID"]
    )
@st.cache_resource
def load_data():
    local_path = os.path.join(os.path.dirname(__file__), "..", "data", "pbmc3k_annotated.h5ad")
    if os.path.exists(local_path):
        adata = sc.read_h5ad(local_path)
        if hasattr(adata, 'layers'):
            del adata.layers
    else:
        adata = sc.datasets.pbmc3k_processed()
    return adata


if input_mode == "Use demo dataset (PBMC 3k)":
    try:
        adata = load_data()
        source_key = "demo"
        st.success(f"Data loaded: {adata.n_obs} cells, {adata.n_vars} genes")
    except Exception as e:
        st.error(f"Failed to load demo dataset: {e}")
        st.exception(e)
        st.stop()

elif input_mode == "Upload .h5ad file":
    uploaded_file = st.file_uploader("Upload .h5ad file", type=["h5ad"])
    if uploaded_file is not None:
        import tempfile
        payload = uploaded_file.getvalue()
        source_key = "upload:" + hashlib.sha256(payload).hexdigest()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp:
                tmp.write(payload)
                tmp_path = tmp.name
            adata = sc.read_h5ad(tmp_path)
        except Exception as error:
            st.error(f"Could not read the uploaded .h5ad file: {error}")
            st.stop()
        finally:
            if tmp_path is not None:
                os.unlink(tmp_path)
        st.success(f"Data loaded: {adata.n_obs} cells, {adata.n_vars} genes")
    else:
        st.info("Please upload a .h5ad file to continue.")
        st.stop()

elif input_mode == "Enter GEO Accession ID":
    geo_id = st.text_input("Enter GEO Accession ID (e.g. GSE84133)")
    load_btn = st.button("Load GEO Data", disabled=not bool(geo_id))

    if load_btn and geo_id:
        import sys
        pipeline_path = os.path.join(os.path.dirname(__file__), "..", "pipeline")
        if pipeline_path not in sys.path:
            sys.path.insert(0, pipeline_path)
        from ingest import download_geo, get_geo_tissue_hint, auto_select_celltypist_model

        with st.spinner(f"Downloading {geo_id} from GEO..."):
            try:
                result_path = download_geo(geo_id, output_dir="/tmp")
                if result_path.endswith(".h5ad"):
                    _geo_adata = sc.read_h5ad(result_path)
                elif os.path.isdir(result_path):
                    import glob
                    mtx_files = glob.glob(f"{result_path}/**/*matrix.mtx*", recursive=True)
                    if mtx_files:
                        mtx_dir = os.path.dirname(mtx_files[0])
                        _geo_adata = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", cache=True)
                    else:
                        st.error("Could not find matrix file in the downloaded GEO dataset.")
                        st.stop()
                else:
                    st.error("Could not load data from GEO. Please upload manually.")
                    st.stop()
                _geo_adata.obs_names_make_unique()
                tissue_text = get_geo_tissue_hint(geo_id, output_dir="/tmp")
                st.session_state["geo_adata"] = _geo_adata
                st.session_state["geo_id_loaded"] = geo_id
                st.session_state["adata_raw"] = _geo_adata.copy()
                st.session_state["celltypist_model"] = auto_select_celltypist_model(tissue_text)
                st.session_state.pop("adata_run", None)
            except Exception as e:
                st.error(f"GEO loading failed: {e}")
                st.stop()
        st.rerun()

    loaded_geo_id = st.session_state.get("geo_id_loaded", "")
    if loaded_geo_id and geo_id and loaded_geo_id == geo_id and "geo_adata" in st.session_state:
        adata = st.session_state["geo_adata"]
        source_key = "geo:" + loaded_geo_id
        st.success(f"Data loaded: {adata.n_obs} cells, {adata.n_vars} genes")
    else:
        st.info("Please enter a GEO Accession ID and click Load.")
        st.stop()

# A previous analysis must never be displayed for a different input.
if st.session_state.get("analysis_source_key") != source_key:
    st.session_state.pop("adata_run", None)
    st.session_state.pop("color_by", None)
    st.session_state["analysis_source_key"] = source_key

tab1, tab2, tab3 = st.tabs(["UMAP", "Cell Type Composition", "Marker Genes"])


st.sidebar.title("About")
st.sidebar.markdown("""
An end-to-end scRNA-seq analysis platform powered by AI.

**AI Annotation Models:**
- ✅ CellTypist
- 🔜 scGPT *(GPU required, not available on free tier)*
- 🔜 Geneformer *(Coming Soon)*

🌐 [GitHub](https://github.com/yujuan-zhang/CellPortal)
""")
st.sidebar.divider()

st.sidebar.header("Analysis Parameters")
resolution = st.sidebar.slider("Clustering Resolution", 0.1, 2.0, 0.7, 0.1)
n_neighbors = st.sidebar.slider("N Neighbors", 5, 50, 10, 5)
max_pct_mt = st.sidebar.slider("Max Mitochondrial %", 1, 20, 5, 1,
                               disabled=not needs_preprocessing(adata))
st.sidebar.caption("Mitochondrial filtering applies to raw counts when analysis runs. "
                   "It is disabled for already processed data.")

run_button = st.sidebar.button("Re-run Analysis")



with tab1:
    st.subheader("UMAP - Cell Clusters")

    should_run = run_button or (needs_preprocessing(adata) and "adata_run" not in st.session_state)
    if should_run:
        progress = st.progress(0)
        status = st.empty()
        try:
            if needs_preprocessing(adata):
                result = run_preprocessing(adata, n_neighbors, resolution, max_pct_mt,
                                           progress=progress, status=status)
            else:
                status.info("Reclustering processed data; raw-count QC is not applied.")
                result = recluster_processed(adata, n_neighbors, resolution)
                progress.progress(100)
        except ValueError as error:
            st.error(f"Analysis could not run: {error}")
            st.stop()
        st.session_state["adata_run"] = result
        status.success("Analysis complete!")

    adata_run = st.session_state.get("adata_run", adata).copy()
    qc = adata_run.uns.get("cellportal_qc")
    if qc:
        st.caption(f"QC retained {qc['retained_cells']} of {qc['input_cells']} cells "
                   f"at mitochondrial counts ≤ {qc['max_pct_mt']:g}%.")

    cluster_col = "leiden" if "leiden" in adata_run.obs.columns else "louvain"
    st.info(f"Current clusters: {adata_run.obs[cluster_col].nunique()}")

    color_options = ["Clusters (Scanpy)"]
    if "cell_type" in adata_run.obs.columns:
        color_options.append("Cell Type Annotation (CellTypist)")
    color_option = st.selectbox("Color by", color_options)
    color_by = cluster_col if "Clusters" in color_option else "cell_type"
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
            # raw is saved after count normalization and before HVG selection/scaling.
            if adata_run.raw is None:
                st.error("Annotation needs log1p-normalized expression in adata.raw. "
                         "Upload raw counts for preprocessing or supply this matrix.")
                st.stop()
            adata_ct = adata_run.raw.to_adata()
            import celltypist
            from celltypist import models
            models.download_models(model=selected_model, force_update=False)
            model = models.Model.load(model=selected_model)
            predictions = celltypist.annotate(adata_ct, model=model, majority_voting=True)
            adata_run.obs["cell_type"] = predictions.predicted_labels["majority_voting"].reindex(adata_run.obs_names).values
            st.session_state["adata_run"] = adata_run
        st.success("Annotation complete! Select 'Cell type annotation' in Color by above.")
        st.rerun()


with tab2:
    st.subheader("Cell Type Composition")
    # Prefer cell_type annotation over leiden cluster numbers
    if "cell_type" in adata_run.obs.columns:
        count_col = "cell_type"
    else:
        count_col = "leiden" if "leiden" in adata_run.obs.columns else "louvain"
    cell_counts = adata_run.obs[count_col].value_counts().reset_index()
    cell_counts.columns = ["Cell Type", "Count"]
    cell_counts["Percentage"] = (cell_counts["Count"] / cell_counts["Count"].sum() * 100).round(2)
    st.dataframe(cell_counts, width='stretch')
    st.bar_chart(cell_counts.set_index("Cell Type")["Count"])

with tab3:
    st.subheader("Top Marker Genes per Cluster")
    sample_var = str(adata_run.var_names[0]) if len(adata_run.var_names) > 0 else ""
    looks_like_barcode = any(p in sample_var for p in ["final_cell", "AAACCT", "barcode", ".final."])
    if looks_like_barcode:
        st.warning("Gene names were not detected in this dataset. Marker genes cannot be displayed.")
    else:
        # Use the same grouping as Color by selection
        default_cluster = "leiden" if "leiden" in adata_run.obs.columns else "louvain"
        groupby = st.session_state.get("color_by", default_cluster)
        if groupby not in adata_run.obs.columns:
            groupby = default_cluster

        # Compute marker genes on demand
        current_groupby = adata_run.uns.get("rank_genes_groups", {}).get("params", {}).get("groupby")
        if current_groupby != groupby:
            if st.button("Compute Marker Genes"):
                with st.spinner(f"Computing marker genes by {groupby}..."):
                    try:
                        sc.tl.rank_genes_groups(adata_run, groupby, method="wilcoxon")
                        st.session_state["adata_run"] = adata_run
                        st.rerun()
                    except Exception as e:
                        st.error(f"Marker gene computation failed: {e}")
        else:
            try:
                marker_df = pd.DataFrame(adata_run.uns["rank_genes_groups"]["names"]).head(10)
                st.dataframe(marker_df, width='stretch')
            except Exception as e:
                st.error(f"Could not display marker genes: {e}")

chat.render(adata_run)
