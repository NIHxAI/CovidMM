import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence, pack_sequence  # ✅ 여기에 확실히 추가
from . import configs



# ============================================================
# 🧠 Vision Branch (CT / CXR)
# ============================================================
class VisionBranchLSTM(nn.Module):
    def __init__(self, input_size=512, hidden_size=int(configs.HIDDEN_SIZE * 1.5), bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=3, batch_first=True,
            bidirectional=bidirectional, dropout=configs.DROPOUT_RATE
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(configs.DROPOUT_RATE)
        
    def forward(self, packed_input):
        packed_output, _ = self.lstm(packed_input)
        padded_output, lengths = pad_packed_sequence(packed_output, batch_first=True)
        pooled_outputs = [
            self.pooling(seq[:ln].T.unsqueeze(0)).squeeze(-1)
            for seq, ln in zip(padded_output, lengths)
        ]
        return torch.cat(pooled_outputs, dim=0)


# ============================================================
# 💊 EHR Branch
# ============================================================
class EHR_Branch(nn.Module):
    def __init__(self, input_size, embedding_size=384, hidden_size=int(configs.HIDDEN_SIZE * 1.5)):
        super().__init__()
        self.embedding = nn.Linear(input_size, embedding_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            embedding_size, hidden_size, num_layers=3,
            batch_first=True, bidirectional=True, dropout=configs.DROPOUT_RATE
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(configs.DROPOUT_RATE)

    def forward(self, packed_ehr: PackedSequence):
        embedded_data = self.relu(self.embedding(packed_ehr.data))
        packed_embedded = PackedSequence(
            embedded_data, packed_ehr.batch_sizes,
            packed_ehr.sorted_indices, packed_ehr.unsorted_indices
        )
        packed_output, _ = self.lstm(packed_embedded)
        padded_output, lengths = pad_packed_sequence(packed_output, batch_first=True)
        pooled_outputs = [
            self.pooling(seq[:ln].T.unsqueeze(0)).squeeze(-1)
            for seq, ln in zip(padded_output, lengths)
        ]
        return torch.cat(pooled_outputs, dim=0)


# ============================================================
# 🧬 scRNA Branch (MLP 기반)
# ============================================================
class SCRNA_Branch(nn.Module):
    def __init__(self, input_size, embed_size=512, hidden_size=256):  # hidden 256으로 축소
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_size, embed_size),
            nn.ReLU(),
            nn.Dropout(configs.DROPOUT_RATE),
        )
        # GRU로 교체 (단층)
        self.gru = nn.GRU(
            embed_size, hidden_size,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.layernorm = nn.LayerNorm(hidden_size * 2)

    def forward(self, scrna_seq_batch):
        x, lengths = pad_packed_sequence(scrna_seq_batch, batch_first=True)
        x = self.feature_encoder(x)
        packed_out, _ = self.gru(pack_sequence([x[i, :ln] for i, ln in enumerate(lengths)], enforce_sorted=False))
        padded_out, lengths = pad_packed_sequence(packed_out, batch_first=True)
        padded_out = self.layernorm(padded_out)
        pooled_outputs = [self.pooling(seq[:ln].T.unsqueeze(0)).squeeze(-1) for seq, ln in zip(padded_out, lengths)]
        return torch.cat(pooled_outputs, dim=0)

# ============================================================
# 🧩 Multi-Modal Classifier
# ============================================================


class MultiModalClassifier_v2(nn.Module):
    def __init__(self, num_ehr_features, modalities=['ehr', 'ct', 'cxr', 'scrna']):
        super().__init__()
        self.modalities = modalities
        self.branch_output_size = int(configs.HIDDEN_SIZE * 1.5) * 2  # bi-LSTM output
        self.embed_dim = 512  # 통일 차원

        # ---------------- Branches ----------------
        if 'ehr' in self.modalities:
            self.ehr_branch = EHR_Branch(input_size=num_ehr_features)
            self.proj_ehr = nn.Linear(self.branch_output_size, self.embed_dim)
        if 'ct' in self.modalities:
            self.ct_branch = VisionBranchLSTM()
            self.proj_ct = nn.Linear(self.branch_output_size, self.embed_dim)
        if 'cxr' in self.modalities:
            self.cxr_branch = VisionBranchLSTM()
            self.proj_cxr = nn.Linear(self.branch_output_size, self.embed_dim)
        if 'scrna' in self.modalities:
            self.scrna_branch = SCRNA_Branch(input_size=configs.SCRNA_FEATURES)
            self.proj_scrna = nn.Linear(512, self.embed_dim)  # 일관성 유지

        # ---------------- Fusion ----------------
        combined_size = len(self.modalities) * self.embed_dim
        self.layernorm = nn.LayerNorm(combined_size)
        self.modality_dropout = nn.Dropout(p=0.25)  # 모달리티 레벨 dropout
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_size, combined_size // 2),
            nn.ReLU(),
            nn.LayerNorm(combined_size // 2),
            nn.Dropout(configs.DROPOUT_RATE),
            nn.Linear(combined_size // 2, combined_size // 4),
            nn.ReLU(),
            nn.LayerNorm(combined_size // 4),
            nn.Dropout(configs.DROPOUT_RATE),
            nn.Linear(combined_size // 4, configs.NUM_CLASSES)
        )

    def forward(self, batch):
        batch_size = len(batch["rid"])
        device = configs.DEVICE
        outputs = []

        def make_output():
            return torch.zeros(batch_size, self.embed_dim, device=device)

        # ----- 각 branch forward -----
        if 'ehr' in self.modalities:
            ehr_out = make_output()
            if batch["ehr"] is not None and batch["ehr"].batch_sizes.numel() > 0:
                ehr_pred = self.ehr_branch(batch["ehr"])
                ehr_out[:ehr_pred.size(0)] = self.proj_ehr(ehr_pred)
            outputs.append(ehr_out)

        if 'ct' in self.modalities:
            ct_out = make_output()
            if batch["ct"] is not None and batch["ct"].batch_sizes.numel() > 0:
                ct_pred = self.ct_branch(batch["ct"])
                ct_out[:ct_pred.size(0)] = self.proj_ct(ct_pred)
            outputs.append(ct_out)

        if 'cxr' in self.modalities:
            cxr_out = make_output()
            if batch["cxr"] is not None and batch["cxr"].batch_sizes.numel() > 0:
                cxr_pred = self.cxr_branch(batch["cxr"])
                cxr_out[:cxr_pred.size(0)] = self.proj_cxr(cxr_pred)
            outputs.append(cxr_out)

        if 'scrna' in self.modalities:
            scrna_out = make_output()
            if batch["scrna"] is not None:
                scrna_pred = self.scrna_branch(batch["scrna"])
                scrna_out[:scrna_pred.size(0)] = self.proj_scrna(scrna_pred)
            outputs.append(scrna_out)

        # ----- Fusion -----
        combined = torch.cat(outputs, dim=1)
        combined = self.layernorm(combined)
        combined = self.modality_dropout(combined)
        logits = self.fusion_layer(combined)
        return logits

