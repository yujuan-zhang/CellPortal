import scanpy as sc
import celltypist
from celltypist import models


def run_annotate(input_path: str, output_path: str):
    # 加载数据
    adata = sc.read_h5ad(input_path)

    # 准备：CellTypist 需要归一化到 10000 的数据
    adata_pred = adata.copy()
    sc.pp.normalize_total(adata_pred, target_sum=1e4)
    sc.pp.log1p(adata_pred)

    # 加载模型并预测
    model = models.Model.load(model="Immune_All_Low.pkl")
    predictions = celltypist.annotate(adata_pred, model=model, majority_voting=True)

    # 把注释结果写回原始数据
    adata.obs["cell_type"] = predictions.predicted_labels["majority_voting"]

    # 保存
    adata.write_h5ad(output_path, compression="gzip")
    print(f"Done! Saved to {output_path}")


if __name__ == "__main__":
    run_annotate(
        input_path="scanpy-tutorials-main/write/pbmc3k.h5ad",
        output_path="scanpy-tutorials-main/write/pbmc3k_annotated.h5ad"
    )
