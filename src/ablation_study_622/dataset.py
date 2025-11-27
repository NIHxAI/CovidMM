import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_sequence
from . import configs


class MultiModalDataset(Dataset):
    def __init__(self, patient_ids, all_metadata_df, all_ehr_df, scrna_df=None):
        """
        Args:
            patient_ids (list): 환자 단위 RID 목록 (예: COV-CCO-001)
            all_metadata_df (pd.DataFrame): 환자별 메타데이터 (Severity, ct_path 등)
            all_ehr_df (pd.DataFrame): EHR 시계열 데이터
            scrna_df (pd.DataFrame, optional): 전처리된 scRNA pseudobulk 데이터 (RID, TIME, DATE, Severity, features...)
        """
        self.patient_ids = patient_ids
        self.metadata = all_metadata_df[all_metadata_df["RID"].isin(patient_ids)].copy()
        self.ehr_data = all_ehr_df
        self.scrna_data = scrna_df  # ✅ scRNA 데이터프레임 저장

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]

        # ------------------------------------------------------
        # ① Label / Meta / EHR 불러오기
        # ------------------------------------------------------
        patient_meta = self.metadata[self.metadata["RID"] == patient_id].sort_values(by="DATE")
        patient_ehr = self.ehr_data[self.ehr_data["RID"] == patient_id].sort_values(by="DATE")

        label = list(configs.LABEL_MAP.keys()).index(patient_meta["Severity"].iloc[-1])

        feature_cols = [f for f in configs.NUMERICAL_FEATURES if f in patient_ehr.columns]
        ehr_features = patient_ehr[feature_cols]
        ehr_tensor = torch.tensor(ehr_features.values, dtype=torch.float32)

        # ------------------------------------------------------
        # ② CXR / CT 임베딩 경로 처리
        # ------------------------------------------------------
        ct_paths = patient_meta["ct_path"].dropna().unique().tolist()
        cxr_paths = patient_meta["cxr_path"].dropna().unique().tolist()

        ct_embeds = [torch.from_numpy(np.load(p)) for p in ct_paths if os.path.exists(p)]
        cxr_embeds = [torch.from_numpy(np.load(p)) for p in cxr_paths if os.path.exists(p)]

        # ------------------------------------------------------
        # ③ scRNA pseudobulk (💡 시계열 입력 유지)
        # ------------------------------------------------------
        scrna_tensor = None
        if self.scrna_data is not None:
            patient_scrna = self.scrna_data[self.scrna_data["RID"] == patient_id].sort_values(by="TIME")
            if not patient_scrna.empty:
                # 숫자형 변환 + NaN 처리
                scrna_features = (
                    patient_scrna.drop(columns=["RID", "TIME", "DATE", "Severity"])
                    .apply(pd.to_numeric, errors="coerce")
                    .fillna(0)
                    .values.astype(np.float32)
                )
                if len(scrna_features) > 0:
                    # ✅ 시계열 그대로 tensor로 반환 → [T, F]
                    scrna_tensor = torch.tensor(scrna_features, dtype=torch.float32)

        return {
            "rid": patient_id,
            "label": torch.tensor(label, dtype=torch.long),
            "ehr": ehr_tensor if len(ehr_tensor) > 0 else None,
            "ct": torch.stack([e.squeeze(0) for e in ct_embeds]) if ct_embeds else None,
            "cxr": torch.stack([e.squeeze(0) for e in cxr_embeds]) if cxr_embeds else None,
            "scrna": scrna_tensor if scrna_tensor is not None else None,
        }


def collate_fn(batch):
    """
    배치 단위 데이터 결합 (PackedSequence 구성)
    """
    batch_out = {
        "rid": [x["rid"] for x in batch],
        "label": torch.stack([x["label"] for x in batch]),
        "ehr": pack_sequence([x["ehr"] for x in batch if x["ehr"] is not None], enforce_sorted=False),
        "ct": pack_sequence([x["ct"] for x in batch if x["ct"] is not None], enforce_sorted=False),
        "cxr": pack_sequence([x["cxr"] for x in batch if x["cxr"] is not None], enforce_sorted=False),
    }

    # ✅ scRNA도 pack_sequence로 묶기 (시계열 입력 유지)
    scrna_seqs = [x["scrna"] for x in batch if x["scrna"] is not None]
    if len(scrna_seqs) > 0:
        batch_out["scrna"] = pack_sequence(scrna_seqs, enforce_sorted=False)
    else:
        batch_out["scrna"] = None

    return batch_out
