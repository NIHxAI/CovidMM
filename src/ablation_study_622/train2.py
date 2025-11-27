# src/train.py

import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from . import configs
from .dataset import MultiModalDataset, collate_fn
# src/train.py
from .model2 import MultiModalClassifier_v2 as MultiModalClassifier

# ============================================================
# 🔸 Early Stopping Class
# ============================================================
class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0, path='saved_models/best_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_f1_max = -np.Inf
        self.delta = delta
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def __call__(self, val_f1, model):
        score = val_f1
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_f1, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_f1, model)
            self.counter = 0

    def save_checkpoint(self, val_f1, model):
        if self.verbose:
            print(f"Validation F1 increased ({self.val_f1_max:.4f} → {val_f1:.4f}). Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_f1_max = val_f1


# ============================================================
# 🔸 Main Training Function
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Multimodal Ablation Study Training")
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["ehr", "ct", "cxr"],
        choices=["ehr", "ct", "cxr", "scrna"],
        help="List of modalities to use for training."
    )
    args = parser.parse_args()
    print(f"선택된 모달리티: {args.modalities}")

    print("="*60)
    print("                 모델 학습을 시작합니다 ")
    print("="*60)
    print(f"사용 장치: {configs.DEVICE}")
    print(f"라벨 맵: {configs.LABEL_MAP}")

    # -----------------------------
    # 📂 Load Datasets
    # -----------------------------
    try:
        metadata_df = pd.read_csv(configs.METADATA_FILE, low_memory=False)
        ehr_df = pd.read_csv(configs.EHR_PROCESSED_FILE, low_memory=False)
        metadata_df["DATE"] = pd.to_datetime(metadata_df["DATE"])
        ehr_df["DATE"] = pd.to_datetime(ehr_df["DATE"])

        scrna_df = None
        if "scrna" in args.modalities:
            scrna_df = pd.read_csv(configs.SCRNA_PROCESSED_FILE, low_memory=False)
            scrna_df["DATE"] = pd.to_datetime(scrna_df["DATE"])
    except FileNotFoundError as e:
        print(f"[❌ 오류] 파일을 찾을 수 없습니다: {e}")
        return

    # -----------------------------
    # 🧩 Train / Val / Test Split
    # -----------------------------
    patient_labels = metadata_df.groupby("RID")["Severity"].last()
    unique_patients = patient_labels.index.values
    labels_for_stratify = patient_labels.values

    train_val_ids, test_ids, train_val_labels, _ = train_test_split(
        unique_patients, labels_for_stratify, test_size=0.2,
        random_state=42, stratify=labels_for_stratify
    )
    train_ids, val_ids, train_labels, _ = train_test_split(
        train_val_ids, train_val_labels, test_size=0.25,
        random_state=42, stratify=train_val_labels
    )

    print(f"Train 환자 수: {len(train_ids)}, Val 환자 수: {len(val_ids)}, Test 환자 수: {len(test_ids)}")

    # -----------------------------
    # 📊 Dataset & DataLoader
    # -----------------------------
    train_dataset = MultiModalDataset(train_ids, metadata_df, ehr_df, scrna_df=scrna_df)
    val_dataset = MultiModalDataset(val_ids, metadata_df, ehr_df, scrna_df=scrna_df)

    train_loader = DataLoader(
        train_dataset, batch_size=configs.BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=configs.BATCH_SIZE,
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

    # -----------------------------
    # ⚖️ Class Weights
    # -----------------------------
    label_counts = Counter(train_labels)
    sorted_labels = sorted(configs.LABEL_MAP.keys(), key=lambda k: configs.LABEL_MAP[k])
    counts = [label_counts.get(label, 0) for label in sorted_labels]
    num_samples = sum(counts)
    class_weights = [
        num_samples / (len(counts) * count) if count > 0 else 1
        for count in counts
    ]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(configs.DEVICE)
    print(f"계산된 클래스 가중치: {class_weights_tensor.cpu().numpy()}")

    # -----------------------------
    # 🧠 Optimizer & Scheduler
    # -----------------------------
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = AdamW(model.parameters(), lr=configs.LEARNING_RATE, weight_decay=1e-4)
    NUM_EPOCHS = configs.NUM_EPOCHS
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)

    model_name = f"best_model_{'_'.join(sorted(args.modalities))}.pth"
    model_path = os.path.join(configs.MODEL_SAVE_DIR, model_name)
    early_stopping = EarlyStopping(patience=20, verbose=True, path=model_path)

    # ============================================================
    # 🚀 TRAIN LOOP
    # ============================================================
    print("\n" + "="*25 + " 학습 시작 " + "="*26)
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for batch in train_loop:
            optimizer.zero_grad()
            labels = batch["label"].to(configs.DEVICE)

            # 각 모달리티별 device 전송
            for key in ["ehr", "ct", "cxr", "scrna"]:
                if key in batch and batch[key] is not None:
                    batch[key] = batch[key].to(configs.DEVICE)

            outputs = model(batch)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        # -----------------------------
        # 🔍 Validation
        # -----------------------------
        model.eval()
        val_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
            for batch in val_loop:
                labels = batch["label"].to(configs.DEVICE)
                for key in ["ehr", "ct", "cxr", "scrna"]:
                    if key in batch and batch[key] is not None:
                        batch[key] = batch[key].to(configs.DEVICE)
                outputs = model(batch)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                val_loop.set_postfix(loss=loss.item())

        # -----------------------------
        # 📈 Metrics
        # -----------------------------
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(all_labels, all_preds)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.1e}")

        early_stopping(val_f1, model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    print("\n" + "="*25 + " 학습 종료 " + "="*26)


if __name__ == "__main__":
    main()
