from __future__ import annotations

import os
import scanpy as sc
import anndata as ad


def _download_file(url: str, dest: str) -> bool:
    """Download url to dest using curl. Returns True on success."""
    import subprocess
    r = subprocess.run(["curl", "-L", "-s", "-o", dest, url])
    return r.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0


def _decompress_gz(path: str) -> str:
    """Decompress a .gz file (not .tar.gz) and return new path."""
    import gzip, shutil
    out = path[:-3]
    if not os.path.exists(path):
        return out  # already decompressed
    if os.path.exists(out):
        os.remove(path)
        return out  # output already exists, just remove the gz
    with gzip.open(path, "rb") as f_in, open(out, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(path)
    return out


def _extract_tar(path: str, dest_dir: str) -> None:
    """Extract a .tar or .tar.gz archive into dest_dir."""
    import tarfile
    with tarfile.open(path) as tar:
        tar.extractall(dest_dir)
    os.remove(path)


def _download_urls(urls: list[str], out_dir: str) -> list[str]:
    """Download a list of URLs into out_dir. Returns list of local paths."""
    downloaded = []
    for url in urls:
        if not url or url == "NONE":
            continue
        filename = os.path.basename(url)
        local = os.path.join(out_dir, filename)
        print(f"  Downloading {filename}...")
        already_exists = os.path.exists(local) and os.path.getsize(local) > 0
        if already_exists or _download_file(url, local):
            if local.endswith(".tar.gz") or local.endswith(".tar"):
                print(f"  Extracting {filename}...")
                _extract_tar(local, out_dir)
                # Decompress any .gz files extracted from the tar (deduplicate paths)
                import glob as _glob
                gz_files = set(_glob.glob(f"{out_dir}/**/*.gz", recursive=True))
                for gz_file in sorted(gz_files):
                    if not gz_file.endswith(".tar.gz") and "soft.gz" not in gz_file:
                        if os.path.exists(gz_file):
                            print(f"  Decompressing {os.path.basename(gz_file)}...")
                            _decompress_gz(gz_file)
            elif local.endswith(".gz"):
                local = _decompress_gz(local)
                downloaded.append(local)
            else:
                downloaded.append(local)
        else:
            print(f"  Warning: failed to download {url}")
    return downloaded


def _collect_geo_urls(gse) -> list[str]:
    """Collect all supplementary file URLs from a GEOparse GSE object."""
    urls = list(gse.metadata.get("supplementary_file", []))
    for gsm in gse.gsms.values():
        urls.extend(gsm.metadata.get("supplementary_file", []))
    return urls


def _collect_ffq_urls(geo_id: str) -> list[str]:
    """Fallback: use ffq CLI to get FTP download URLs."""
    import subprocess, json
    try:
        r = subprocess.run(
            ["ffq", "--ftp", geo_id],
            capture_output=True, text=True, timeout=60
        )
        if r.returncode != 0:
            return []
        data = json.loads(r.stdout)

        urls = []
        def _walk(obj):
            if isinstance(obj, dict):
                if "url" in obj:
                    urls.append(obj["url"])
                for v in obj.values():
                    _walk(v)
            elif isinstance(obj, list):
                for item in obj:
                    _walk(item)
        _walk(data)
        return urls
    except Exception as e:
        print(f"ffq fallback failed: {e}")
        return []


def _find_10x_dir(base_dir: str) -> str | None:
    """Return the directory containing a 10x matrix.mtx file, or None."""
    import glob
    mtx_files = glob.glob(f"{base_dir}/**/*matrix.mtx*", recursive=True)
    return os.path.dirname(mtx_files[0]) if mtx_files else None


def download_geo(geo_id: str, output_dir: str = "/tmp") -> str:
    import GEOparse

    print(f"Fetching {geo_id} from GEO...")
    out_path = os.path.join(output_dir, geo_id)
    os.makedirs(out_path, exist_ok=True)

    # Return cached h5ad if it already exists
    cached_h5ad = os.path.join(out_path, f"{geo_id}_merged.h5ad")
    if os.path.exists(cached_h5ad):
        print(f"Using cached file: {cached_h5ad}")
        return cached_h5ad

    # Step 1: get metadata via GEOparse
    gse = GEOparse.get_GEO(geo=geo_id, destdir=out_path, silent=True)

    # Step 2: collect supplementary file URLs (GSE + GSM level)
    urls = _collect_geo_urls(gse)
    print(f"Found {len(urls)} supplementary file URLs via GEOparse")

    # Step 3: if no URLs found, fall back to ffq
    if not urls:
        print("No supplementary URLs from GEOparse, trying ffq...")
        urls = _collect_ffq_urls(geo_id)
        print(f"Found {len(urls)} URLs via ffq")

    # Step 4: filter to relevant file types only
    relevant_exts = (".h5ad", ".h5", ".loom", ".tar.gz", ".tar",
                     ".mtx", ".mtx.gz", ".tsv", ".tsv.gz", ".csv", ".csv.gz",
                     "barcodes.tsv", "features.tsv", "genes.tsv")
    filtered = [u for u in urls if any(u.lower().endswith(e) or e in u.lower() for e in relevant_exts)]
    if not filtered:
        filtered = urls  # download everything if no filter matches

    # Limit to first dataset if too many (e.g., many samples)
    MAX_FILES = 20
    if len(filtered) > MAX_FILES:
        print(f"Limiting download to first {MAX_FILES} of {len(filtered)} files")
        filtered = filtered[:MAX_FILES]

    # Step 5: download
    downloaded = _download_urls(filtered, out_path)

    # Step 6: check for h5ad
    h5ad_files = [f for f in downloaded if f.endswith(".h5ad")]
    if h5ad_files:
        return h5ad_files[0]

    # Step 7: check for 10x mtx
    mtx_dir = _find_10x_dir(out_path)
    if mtx_dir:
        return mtx_dir

    # Step 8: check for CSV files → merge into one AnnData and save as h5ad
    import glob
    csv_files = glob.glob(f"{out_path}/**/*.csv", recursive=True)
    if csv_files:
        print(f"Found {len(csv_files)} CSV file(s), merging into AnnData...")
        h5ad_out = os.path.join(out_path, f"{geo_id}_merged.h5ad")
        _merge_csv_to_h5ad(csv_files, h5ad_out)
        return h5ad_out

    return out_path


def _merge_csv_to_h5ad(csv_files: list[str], output_path: str) -> None:
    """Read GEO-style CSV files and merge into a single h5ad file."""
    import pandas as pd
    import numpy as np

    adatas = []
    for f in csv_files:
        sample_name = os.path.basename(f).split("_")[0]  # e.g. GSM2230757
        print(f"  Reading {os.path.basename(f)}...")
        df = pd.read_csv(f, index_col=0)

        # Check for label in last ROW (genes×cells format: non-numeric last row)
        row_labels = None
        last_row = df.iloc[-1, :]
        non_numeric_frac = last_row.apply(lambda x: not str(x).replace('.','').replace('-','').isdigit()).mean()
        if non_numeric_frac > 0.8:
            row_labels = last_row.values
            df = df.iloc[:-1, :]

        # Check for label in last COLUMN (cells×genes format: object dtype)
        label_col = None
        if df.iloc[:, -1].dtype == object:
            label_col = df.columns[-1]
            col_labels = df[label_col]
            df = df.drop(columns=[label_col])

        # Detect orientation: barcode-like index → cells×genes; more rows than cols → genes×cells
        sample_idx = str(df.index[0]) if len(df.index) > 0 else ""
        is_barcode_rows = any(p in sample_idx for p in ["final_cell", "AAACCT", "barcode", "-1", "_"])
        if not is_barcode_rows and df.shape[0] > df.shape[1]:
            df = df.T
            label_col = None  # col labels were per-gene, not per-cell
        else:
            row_labels = None  # row labels were per-gene, not per-cell

        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

        a = ad.AnnData(X=df.values.astype(np.float32),
                       obs=pd.DataFrame(index=df.index),
                       var=pd.DataFrame(index=df.columns))
        a.obs["sample"] = sample_name
        if label_col is not None:
            a.obs["cell_type"] = col_labels.values
        elif row_labels is not None:
            a.obs["cell_type"] = row_labels

        adatas.append(a)

    merged = ad.concat(adatas, join="outer", fill_value=0)
    merged.var_names_make_unique()
    # Ensure all obs string columns are actually str (no NaN) before writing
    for col in merged.obs.columns:
        if merged.obs[col].dtype == object:
            merged.obs[col] = merged.obs[col].fillna("unknown").astype(str)
    merged.write_h5ad(output_path, compression="gzip")
    print(f"Saved merged AnnData: {merged.n_obs} cells, {merged.n_vars} genes → {output_path}")


def get_geo_tissue_hint(geo_id: str, output_dir: str = "/tmp") -> str:
    """Return a free-text description of the GEO dataset for model selection."""
    import GEOparse
    out_path = os.path.join(output_dir, geo_id)
    os.makedirs(out_path, exist_ok=True)
    try:
        gse = GEOparse.get_GEO(geo=geo_id, destdir=out_path, silent=True)
        parts = []
        for key in ("title", "summary", "overall_design"):
            parts += gse.metadata.get(key, [])
        return " ".join(parts).lower()
    except Exception:
        return ""


def auto_select_celltypist_model(tissue_text: str) -> str:
    """Pick the best CellTypist model based on tissue/cell keywords."""
    TISSUE_MODEL_MAP = [
        (["pancreatic islet", "islet", "beta cell", "acinar", "endocrine pancrea"],
                                                          "Adult_Human_PancreaticIslet.pkl"),
        (["fetal pancrea", "embryonic pancrea"],          "Fetal_Human_Pancreas.pkl"),
        (["pbmc", "peripheral blood mononuclear"],        "Immune_All_Low.pkl"),
        (["blood", "t cell", "b cell", "nk cell", "immune cell"],
                                                          "Immune_All_High.pkl"),
        (["hippocampus"],                                 "Human_AdultAged_Hippocampus.pkl"),
        (["prefrontal cortex"],                           "Adult_Human_PrefrontalCortex.pkl"),
        (["brain", "neuron", "cortex", "cerebr"],         "Developing_Human_Brain.pkl"),
        (["lung", "airway", "alveol", "bronch"],          "Human_Lung_Atlas.pkl"),
        (["intestin", "colon", "gut", "bowel"],           "Cells_Intestinal_Tract.pkl"),
        (["liver", "hepat"],                              "Healthy_Human_Liver.pkl"),
        (["skin", "dermis", "epiderm"],                   "Adult_Human_Skin.pkl"),
        (["heart", "cardiac", "myocard"],                 "Healthy_Adult_Heart.pkl"),
        (["tonsil"],                                      "Cells_Human_Tonsil.pkl"),
        (["breast"],                                      "Cells_Adult_Breast.pkl"),
        (["thymus"],                                      "Developing_Human_Thymus.pkl"),
        (["retina", "ocular", "eye"],                     "Human_Developmental_Retina.pkl"),
        (["fetal", "embryo"],                             "Pan_Fetal_Human.pkl"),
        (["mouse", "murine"],                             "Mouse_Whole_Brain.pkl"),
    ]
    for keywords, model in TISSUE_MODEL_MAP:
        if any(kw in tissue_text for kw in keywords):
            return model
    return "Immune_All_Low.pkl"  # default fallback


if __name__ == "__main__":
    result = download_geo("GSE84133", output_dir="/tmp")
    print(f"Downloaded to: {result}")
