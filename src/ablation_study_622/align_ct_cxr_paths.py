from __future__ import annotations
from pathlib import Path
from datetime import datetime
import pandas as pd

# ========= 1) Base directories (fixed to /data) =========
BASE_DIR = Path("/data").resolve()
CSV_PATH = BASE_DIR / "processed" / "rid,date,severity.csv"
CT_ROOT  = BASE_DIR / "embeddings" / "ct_embeddings"
CXR_ROOT = BASE_DIR / "embeddings" / "cxr_embeddings"
OUT_CSV  = BASE_DIR / "processed" / "final_metadata_manual_aligned_3days.csv"

print(f"📂 기준 디렉토리: {BASE_DIR}")
print(f"🧾 입력 CSV:     {CSV_PATH}")
print(f"💾 CT 폴더:      {CT_ROOT}")
print(f"💾 CXR 폴더:     {CXR_ROOT}")

# ========= 2) Date helpers =========
def parse_date(date_str: str) -> datetime:
    return datetime.strptime(str(date_str), "%Y-%m-%d")

def date_folder_to_dt(folder_name: str) -> datetime | None:
    # e.g., '20210112_extra' -> '20210112'
    clean = str(folder_name).split("_", 1)[0]
    try:
        return datetime.strptime(clean, "%Y%m%d")
    except ValueError:
        return None

# ========= 3) Scan embeddings into a per-RID map =========
def collect_available_paths(mod_root: Path) -> dict[str, list[tuple[datetime, Path]]]:
    """
    Expected layout under mod_root:
      mod_root/
        <group>/
          <RID>/
            <date_folder>/.../*.npy
    Returns: { rid_norm: [(dt, npy_path0), ...], ... }
    """
    mapping: dict[str, list[tuple[datetime, Path]]] = {}

    if not mod_root.is_dir():
        return mapping

    for group_dir in mod_root.iterdir():
        if not group_dir.is_dir():
            continue

        for rid_dir in group_dir.iterdir():
            if not rid_dir.is_dir():
                continue

            rid_norm = rid_dir.name.replace("_", "-")
            seen_dates: set[datetime] = set()

            for date_dir in sorted((p for p in rid_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
                dt = date_folder_to_dt(date_dir.name)
                if dt is None or dt in seen_dates:
                    continue
                seen_dates.add(dt)

                # find first .npy under this date_dir (depth-first)
                npy_path: Path | None = None
                for sub in [date_dir] + [p for p in date_dir.rglob("*") if p.is_dir()]:
                    for f in sub.iterdir():
                        if f.is_file() and f.suffix == ".npy":
                            npy_path = f
                            break
                    if npy_path:
                        break

                if npy_path:
                    mapping.setdefault(rid_norm, []).append((dt, npy_path))

    return mapping

print("🔍 CT/CXR 폴더 스캔 중...")
ct_map  = collect_available_paths(CT_ROOT)
cxr_map = collect_available_paths(CXR_ROOT)
print(f"📊 CT 케이스 수: {len(ct_map)}, CXR 케이스 수: {len(cxr_map)}")

# ========= 4) Load metadata CSV =========
df = pd.read_csv(CSV_PATH)

# ========= 5) Matching utilities =========
def find_match(rid_map: dict[str, list[tuple[datetime, Path]]], rid: str, date: datetime) -> Path | None:
    lst = rid_map.get(rid)
    if not lst:
        return None
    # first file within ±3 days (you can tweak policy here)
    for d, fpath in lst:
        if abs((d - date).days) <= 3:
            return fpath
    return None

def to_data_relative(path: Path | None) -> str | None:
    """
    Return an absolute '/data/...' path.
    - If `path` is under BASE_DIR -> normalize to '/data/...'
    - If `path` is relative like 'data/...' -> map to '/data/...'
    - Else return its absolute resolved string.
    """
    if path is None:
        return None

    p = Path(path)

    # If it's a 'data/...' relative path, convert to '/data/...'
    if not p.is_absolute() and str(p).startswith("data/"):
        return str((BASE_DIR / str(p)[len("data/"):]).resolve())

    p = p.resolve()

    try:
        # If p is already under /data, normalize to that absolute
        rel = p.relative_to(BASE_DIR)
        return str((BASE_DIR / rel).resolve())
    except ValueError:
        # Not under /data — return absolute as-is
        return str(p)

# ========= 6) Perform matching per row =========
ct_paths: list[str | None] = []
cxr_paths: list[str | None] = []

for _, row in df.iterrows():
    rid  = str(row["RID"]).replace("_", "-")
    date = parse_date(row["DATE"])

    ct_match  = find_match(ct_map, rid, date)
    cxr_match = find_match(cxr_map, rid, date)

    ct_paths.append(to_data_relative(ct_match))
    cxr_paths.append(to_data_relative(cxr_match))

df["ct_path"]  = ct_paths
df["cxr_path"] = cxr_paths

# ========= 7) Save =========
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)

print(f"\n✅ 저장 완료: {OUT_CSV}")
print(f"총 {len(df)}개의 행 처리 완료")
