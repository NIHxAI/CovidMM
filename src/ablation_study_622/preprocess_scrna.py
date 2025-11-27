#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess scRNA pseudobulk data for multimodal fusion
- Filters top 2000 HVGs
- Performs KEGG enrichment using the 2000 HVGs (once, globally)
- Selects top 50 pathways by adjusted p-value
- Aggregates expressions (sum) of genes within each pathway for all samples
- Merges with Phenodata + RID list
Author: ssh010214
"""

import os
import re
import pandas as pd
from gseapy import enrichr

# ============================================================
# 1️⃣ PATH 설정
# ============================================================
bulk_dir = "/data/bulk"
expr_path = os.path.join(bulk_dir, "pseudobulk_long.tsv")
pheno_path = os.path.join(bulk_dir, "Phenodata.tsv")
rid_map_path = os.path.join(bulk_dir, "scRNA_rid_list.csv")
save_path = "/data/processed/scrna_processed.csv"

os.makedirs(os.path.dirname(save_path), exist_ok=True)

print("[INFO] Loading files...")
expr = pd.read_csv(expr_path, sep="\t")
pheno = pd.read_csv(pheno_path, sep="\t")
rid_map = pd.read_csv(rid_map_path)

# 문자열 정리
expr["Sample"] = expr["Sample"].astype(str).str.strip()
pheno["Sample"] = pheno["Sample"].astype(str).str.strip()
rid_map["RID_with_time"] = rid_map["RID_with_time"].astype(str).str.strip()

# ============================================================
# 2️⃣ HVG (Highly Variable Genes) 선택 (상위 2000)
# ============================================================
print("[INFO] Selecting top 2000 HVGs...")
gene_var = expr.groupby("Gene")["Expression"].var().sort_values(ascending=False)
top_genes = list(gene_var.head(2000).index)
expr = expr[expr["Gene"].isin(top_genes)]
print(f"[INFO] Retained {len(top_genes)} HVGs out of {len(gene_var)} genes.")

# ============================================================
# 3️⃣ KEGG Pathway Enrichment (global on HVG list)
# ============================================================
print("[INFO] Performing KEGG pathway enrichment (one global test on 2000 HVGs)...")
enr = enrichr(
    gene_list=top_genes,
    gene_sets=["KEGG_2021_Human"],
    cutoff=0.05
)
enr_results = enr.results.sort_values("Adjusted P-value", ascending=True)
top_pathways = enr_results.head(50)["Term"].tolist()
print(f"[INFO] Selected top {len(top_pathways)} KEGG pathways (by adjusted p-value).")

# pathway → gene 매핑
pathway2genes = {}
for _, row in enr_results.iterrows():
    if row["Term"] in top_pathways:
        genes = [g.strip() for g in row["Genes"].split(";")]
        pathway2genes[row["Term"]] = [g for g in genes if g in top_genes]

# ============================================================
# 4️⃣ Pivot: Sample × (Gene_CellType)
# ============================================================
print("[INFO] Pivoting expression matrix (Sample × Gene_CellType)...")
pivot = expr.pivot_table(
    index="Sample",
    columns=["Gene", "CellType"],
    values="Expression",
    fill_value=0
)
pivot.columns = [f"{g}_{c}" for g, c in pivot.columns]
pivot.reset_index(inplace=True)

# ============================================================
# 5️⃣ Pathway별 gene 합산 (모든 샘플 공통 gene 기준)
# ============================================================
print("[INFO] Aggregating gene expressions within top pathways (sum per celltype)...")

celltypes = sorted(set([c.split("_")[-1] for c in pivot.columns if "_" in c]))
pathway_features = []

for pw, genes in pathway2genes.items():
    clean_pw = re.sub(r"[^A-Za-z0-9_]+", "_", pw)
    for ct in celltypes:
        cols = [f"{g}_{ct}" for g in genes if f"{g}_{ct}" in pivot.columns]
        if len(cols) > 0:
            pivot[f"{clean_pw}_{ct}"] = pivot[cols].sum(axis=1)
            pathway_features.append(f"{clean_pw}_{ct}")

pivot = pivot[["Sample"] + pathway_features]
print(f"[INFO] Final feature dimension: {len(pathway_features)}")

# ============================================================
# 6️⃣ Phenodata 및 RID 병합
# ============================================================
merged = pd.merge(pivot, pheno, on="Sample", how="left")
rid_map = rid_map.rename(columns={"RID_with_time": "Sample"})
merged = pd.merge(merged, rid_map, on="Sample", how="left")

# ✅ Status/Severity 중복 정리
status_cols = [c for c in merged.columns if "Status" in c or "Severity" in c]
if len(status_cols) > 1:
    print(f"[WARN] Multiple status-related columns detected: {status_cols}")
    for cand in ["Status_x", "Status", "Severity_x", "Severity_y"]:
        if cand in merged.columns:
            merged["Severity"] = merged[cand]
            break
    merged = merged.loc[:, ~((merged.columns.str.contains("Status|Severity")) & (merged.columns != "Severity"))]

# ============================================================
# 7️⃣ RID / TIME / DATE 정리
# ============================================================
def extract_rid(sample):
    m = re.match(r"^(COV-[A-Z]+-\d{3})(\d)$", sample)
    return m.group(1) if m else sample

def extract_time(sample):
    m = re.search(r"(\d)$", sample)
    return int(m.group(1)) if m else None

merged["RID"] = merged["Sample"].apply(extract_rid)
merged["TIME"] = merged["Sample"].apply(extract_time)

# DATE 처리 (NaT 방지)
if "DATE" in merged.columns:
    merged["DATE"] = merged["DATE"].astype(str).str.strip()
    merged["DATE"] = merged["DATE"].replace("nan", pd.NA)
    merged["DATE"] = pd.to_datetime(merged["DATE"], errors="coerce", infer_datetime_format=True)
    if merged["DATE"].isna().all():
        print("[WARN] All DATE values are NaT. Generating pseudo-dates by RID/TIME order.")
        merged["DATE"] = merged.groupby("RID")["TIME"].transform(
            lambda x: pd.to_datetime("2020-01-01") + pd.to_timedelta(x, unit="D")
        )
else:
    merged["DATE"] = pd.NaT

# ============================================================
# 8️⃣ 정렬 및 저장
# ============================================================
meta_cols = ["RID", "TIME", "DATE", "Severity"]
feature_cols = [c for c in merged.columns if c not in meta_cols + ["Sample"]]
merged = merged[meta_cols + feature_cols]
merged = merged.sort_values(by=["RID", "DATE", "TIME"]).reset_index(drop=True)

merged.to_csv(save_path, index=False)
print(f"[✅] Saved processed scRNA pseudobulk → {save_path}")
print(f"[INFO] Final shape: {merged.shape}")
print(merged.head(5)[['RID', 'TIME', 'DATE', 'Severity']])

