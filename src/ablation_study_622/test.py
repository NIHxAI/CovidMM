# src/test.py

import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import configs
from .dataset import MultiModalDataset, collate_fn
from .model import MultiModalClassifier_v2 as MultiModalClassifier



# ============================================================
# 🔹 Confusion Matrix 시각화
# ============================================================
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[✅] Confusion Matrix 저장 완료 → {save_path}")


# ============================================================
# 🔸 Main Testing Function
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Multimodal Testing (EHR / CT / CXR / scRNA)")
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["ehr", "ct", "cxr"],
        choices=["ehr", "ct", "cxr", "scrna"],
        help="List of modalities to test."
    )
    args = parser.parse_args()
    print(f"선택된 모달리티: {args.modalities}")

    print("=" * 60)
    print("       🎯 최종 모델 성능 평가 및 Confusion Matrix 생성 시작")
    print("=" * 60)

    # -----------------------------
    # 📂 Load Datasets
    # -----------------------------
    metadata_df = pd.read_csv(configs.METADATA_FILE, low_memory=False)
    ehr_df = pd.read_csv(configs.EHR_PROCESSED_FILE, low_memory=False)
    metadata_df["DATE"] = pd.to_datetime(metadata_df["DATE"])
    ehr_df["DATE"] = pd.to_datetime(ehr_df["DATE"])

    scrna_df = None
    if "scrna" in args.modalities:
        scrna_df = pd.read_csv(configs.SCRNA_PROCESSED_FILE, low_memory=False)
        scrna_df["DATE"] = pd.to_datetime(scrna_df["DATE"])

    # -----------------------------
    # 🧩 Split Test Set
    # -----------------------------
    patient_labels = metadata_df.groupby("RID")["Severity"].last()
    unique_patients = patient_labels.index.values
    labels_for_stratify = patient_labels.values
    _, test_ids, _, _ = train_test_split(
        unique_patients, labels_for_stratify,
        test_size=0.2, random_state=42, stratify=labels_for_stratify
    )
    print(f"Test 환자 수: {len(test_ids)}")

    # -----------------------------
    # 🧠 Test Dataset & DataLoader
    # -----------------------------
    test_dataset = MultiModalDataset(
        test_ids, metadata_df, ehr_df, scrna_df=scrna_df
    )
    test_loader = DataLoader(
        test_dataset, batch_size=configs.BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn
    )

    # -----------------------------
    # ⚙️ Model Initialize
    # -----------------------------
    actual_ehr_features = [f for f in configs.NUMERICAL_FEATURES if f in ehr_df.columns]
    num_ehr_features = len(actual_ehr_features)
    print(f"실제 사용될 EHR 특징 개수: {num_ehr_features}")

    model = MultiModalClassifier(
        num_ehr_features=num_ehr_features,
        modalities=args.modalities
    ).to(configs.DEVICE)

    model_name = f"best_model_{'_'.join(sorted(args.modalities))}.pth"
    model_path = os.path.join(configs.MODEL_SAVE_DIR, model_name)

    # -----------------------------
    # 🔍 Load Model Weights
    # -----------------------------
    try:
        model.load_state_dict(torch.load(model_path, map_location=configs.DEVICE))
        print(f"[✅] '{model_path}'에서 모델 가중치 불러오기 성공!")
    except FileNotFoundError:
        print(f"[❌] 저장된 모델 파일을 찾을 수 없습니다: {model_path}")
        return

    # -----------------------------
    # 🚀 Testing Loop
    # -----------------------------
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        test_loop = tqdm(test_loader, desc="Testing")
        for batch in test_loop:
            labels = batch["label"].to(configs.DEVICE)

            # 각 modality별 GPU 전송
            for key in ["ehr", "ct", "cxr", "scrna"]:
                if key in batch and batch[key] is not None:
                    batch[key] = batch[key].to(configs.DEVICE)

            outputs = model(batch)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # -----------------------------
    # 📊 Classification Report
    # -----------------------------
    class_names = list(configs.LABEL_MAP.keys())
    print("\n" + "=" * 60)
    print("                 📈 최종 성능 평가 리포트")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    # -----------------------------
    # 🔹 Confusion Matrix
    # -----------------------------
    cm_name = f"confusion_matrix_{'_'.join(sorted(args.modalities))}.png"
    cm_path = os.path.join(configs.MODEL_SAVE_DIR, cm_name)
    plot_confusion_matrix(all_labels, all_preds, class_names, cm_path)


if __name__ == "__main__":
    main()
