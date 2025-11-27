#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess scRNA pseudobulk data for multimodal fusion
- Combines pseudobulk_long.tsv + Phenodata.tsv + scRNA_rid_list.csv
- Filters top 2000 HVGs
- Extracts RID (COV-XXX-0011 → COV-XXX-001) and TIME (→ 1)
- Adds DATE, Severity for time-series multimodal alignment
- Saves to data/processed/scrna_processed_all.csv
Author: ssh010214
"""

import os
import re
import pandas as pd

# ============================================================
# 1️⃣ PATH 설정
# ============================================================
bulk_dir = "/data/bulk"
expr_path = os.path.join(bulk_dir, "pseudobulk_long.tsv")
pheno_path = os.path.join(bulk_dir, "Phenodata.tsv")
rid_map_path = os.path.join(bulk_dir, "scRNA_rid_list.csv")
save_path = "/data/processed/scrna_processed_all.csv"

os.makedirs(os.path.dirname(save_path), exist_ok=True)

print("[INFO] Loading files...")
expr = pd.read_csv(expr_path, sep="\t")
pheno = pd.read_csv(pheno_path, sep="\t")
rid_map = pd.read_csv(rid_map_path)

# --- 문자열 공백 제거 및 타입 통일
expr["Sample"] = expr["Sample"].astype(str).str.strip()
pheno["Sample"] = pheno["Sample"].astype(str).str.strip()
rid_map["RID_with_time"] = rid_map["RID_with_time"].astype(str).str.strip()

# ============================================================
# 2️⃣ HVG (Highly Variable Genes) 선택 (상위 2000)
# ============================================================
print("[INFO] Selecting top 2000 HVGs...")
gene_var = expr.groupby("Gene")["Expression"].var().sort_values(ascending=False)
top_genes = set(gene_var.head(2000).index)
expr = expr[expr["Gene"].isin(top_genes)]
print(f"[INFO] Retained {len(top_genes)} HVGs out of {len(gene_var)} genes.")

# ============================================================
# 3️⃣ Pivot: Sample × (Gene_CellType)
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
# 4️⃣ Phenodata 병합
# ============================================================
print("[INFO] Merging Phenodata and RID map...")
# 필요 컬럼만 선택
use_cols = [c for c in ["Sample", "Status", "DATE"] if c in pheno.columns]
pheno = pheno[use_cols].copy()

# rid_map 준비
rid_map = rid_map.rename(columns={"RID_with_time": "Sample"})
rid_map = rid_map.loc[:, ~rid_map.columns.str.contains("Status|DATE", case=False)]

# 병합
merged = pd.merge(pivot, pheno, on="Sample", how="left")
merged = pd.merge(merged, rid_map, on="Sample", how="left")

# ============================================================
# 5️⃣ RID / TIME 분리 및 정리
# ============================================================

def extract_rid(sample):
    m = re.match(r"^(COV-[A-Z]+-\d{3})(\d)$", str(sample))
    return m.group(1) if m else sample

def extract_time(sample):
    m = re.search(r"(\d)$", str(sample))
    return int(m.group(1)) if m else None

merged["RID"] = merged["Sample"].apply(extract_rid)
merged["TIME"] = merged["Sample"].apply(extract_time)

# ============================================================
# 6️⃣ 중복 Status / Severity 정리
# ============================================================

status_cols = [c for c in merged.columns if "Status" in c or "Severity" in c]
if len(status_cols) > 1:
    print(f"[WARN] Multiple status-related columns detected: {status_cols}")
    # 우선순위: Status_x → Status → Severity_x → Severity_y
    main_col = None
    for cand in ["Status_x", "Status", "Severity_x", "Severity_y"]:
        if cand in merged.columns:
            main_col = cand
            break
    if main_col is not None:
        merged["Severity"] = merged[main_col]
    # 나머지 전부 제거
    merged = merged.loc[:, ~((merged.columns.str.contains("Status|Severity")) & (merged.columns != "Severity"))]
else:
    # Status만 하나 있을 경우 바로 rename
    if "Status" in merged.columns:
        merged.rename(columns={"Status": "Severity"}, inplace=True)

# ============================================================
# 7️⃣ DATE 처리 (NaT 방지)
# ============================================================
if "DATE" in merged.columns:
    # 문자열이면 변환 시도
    merged["DATE"] = merged["DATE"].astype(str).str.strip()
    merged["DATE"] = merged["DATE"].replace("nan", pd.NA)
    # 여러 포맷 시도 (YYYY-MM-DD / DD-MM-YYYY 등)
    merged["DATE"] = pd.to_datetime(
        merged["DATE"],
        errors="coerce",
        format=None,  # 자동 감지
        infer_datetime_format=True
    )

    # DATE 전부 NaT인 경우 → RID별 TIME 순서 기준으로 가상 날짜 생성
    if merged["DATE"].isna().all():
        print("[WARN] All DATE values are NaT. Generating pseudo-dates by RID/TIME order.")
        pseudo_dates = (
            merged.groupby("RID")["TIME"]
            .transform(lambda x: pd.to_datetime("2020-01-01") + pd.to_timedelta(x, unit="D"))
        )
        merged["DATE"] = pseudo_dates
else:
    merged["DATE"] = pd.NaT

# ============================================================
# 8️⃣ 컬럼 순서 및 정렬
# ============================================================
meta_cols = ["RID", "TIME", "DATE", "Severity"]
feature_cols = [c for c in merged.columns if c not in meta_cols + ["Sample"]]
merged = merged[meta_cols + feature_cols]
merged = merged.sort_values(by=["RID", "DATE", "TIME"]).reset_index(drop=True)

# ============================================================
# 9️⃣ 저장
# ============================================================
merged.to_csv(save_path, index=False)
print(f" Saved processed scRNA pseudobulk → {save_path}")
print(f"[INFO] Final shape: {merged.shape}")
print(f"[INFO] Columns preview:", list(merged.columns[:10]))
print(merged.head(5)[["RID", "TIME", "DATE", "Severity"]])

