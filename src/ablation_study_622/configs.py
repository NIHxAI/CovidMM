import torch
import os
import pandas as pd

# ============================================================
# 📁 PATH SETTINGS
# ============================================================
DATA_DIR = '/data'
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
EMBEDDING_DIR = os.path.join(DATA_DIR, 'embeddings')
MODEL_SAVE_DIR = '/saved_models'

# --- Core metadata files ---
METADATA_FILE = os.path.join(PROCESSED_DIR, 'final_metadata_manual_aligned_3days.csv')
EHR_PROCESSED_FILE = os.path.join(PROCESSED_DIR, 'ehr_processed.csv')
SCRNA_PROCESSED_FILE = os.path.join(PROCESSED_DIR, 'scrna_processed.csv')

# ============================================================
# ⚙️ MODEL & TRAINING HYPERPARAMETERS
# ============================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HIDDEN_SIZE = 256
DROPOUT_RATE = 0.3
NUM_CLASSES = 3
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
NUM_EPOCHS = 50

# ============================================================
# 🧬 SCRNA SETTINGS (Auto-detect HVG-based dimension)
# ============================================================
SCRNA_FEATURES = 16000  # fallback default

if os.path.exists(SCRNA_PROCESSED_FILE):
    try:
        scrna_df = pd.read_csv(SCRNA_PROCESSED_FILE, nrows=1)
        SCRNA_FEATURES = len([
            c for c in scrna_df.columns
            if c not in ["RID", "TIME", "DATE", "Severity"]
        ])
    except Exception as e:
        print(f"[WARN] Could not auto-detect scRNA feature dim: {e}")
else:
    print(f"[WARN] scrna_processed.csv not found → using default SCRNA_FEATURES={SCRNA_FEATURES}")

# ============================================================
# 📊 EHR FEATURE LIST
# ============================================================
NUMERICAL_FEATURES = [
    'BUN', 'Creatinine', 'Hemoglobin', 'LDH', 'Neutrophils', 'Lymphocytes',
    'Platelet count', 'WBC Count', 'CRP', 'BDTEMP', 'BREATH', 'DBP', 'PULSE',
    'SBP', 'SPO2', 'Oxygen', 'Albumin', 'ALKP', 'ALT', 'AST', 'rGTP',
    'Calcium', 'Glucose', 'Phosphorus', 'Protein.total', 'Uricacid',
    'Homocysteine', 'Bilirubin.direct', 'Bilirubin.total', 'Iron', 'UIBC',
    'hsCRP', 'Cholesterol', 'Triglyceride', 'HDLC', 'LDLC', 'Ferritin',
    'VitaminB12', 'Folate', 'CPK.total', 'CystatinC', 'APOA1', 'APOB',
    'APOA2', 'HbA1c', 'WBC', 'RBC', 'Hematocrit', 'Platelet', 'A2M', 'B2M',
    'AHSG', 'MBL2', 'PROS1', 'SERPINA4', 'CD14', 'C2', 'PF4', 'LBP', 'LRG1',
    'CFP', 'SERPINC1', 'SERPINA10', 'ALDH1A1', 'TNFRSF17', 'CCL14', 'CCL18',
    'PROC', 'COL1A1', 'CFD', 'DPP4', 'FAP', 'LGALS3', 'LGALS3BP', 'IGFBP2',
    'LUM', 'MPO', 'AOC3', 'MUC1', 'CCL5', 'C9', 'S100A12', 'CSF1R', 'MMP2',
    'MMP9', 'MB', 'LCN2', 'TIMP1', 'CCL19', 'CCL2', 'CCL3', 'CCL4', 'FCER2',
    'PECAM1', 'CXCL9', 'ERBB3', 'FLT3LG', 'GZMB', 'WFDC2', 'IFNG', 'IGFBP4',
    'IL1B', 'IL10', 'IL12A', 'IL6', 'IL8', 'KLK6', 'TNFSF14', 'MMP12',
    'MMP13', 'MMP3', 'MMP7', 'MMP8', 'TGFA', 'TNF', 'TREM1', 'TSLP', 'VEGF',
    'AMBP', 'ANGPT2', 'TNFSF13B', 'BMP10', 'CNTN1', 'CXCL11', 'CXCL13',
    'CXCL5', 'ESM1', 'ERBB2', 'FGF2', 'LGALS9', 'ICAM1', 'IL1RN', 'IL4',
    'IL5', 'LEP', 'MADCAM1', 'KITLG', 'THPO', 'PLAU', 'VCAM1', 'CCL22',
    'CCL23', 'CCL26', 'KIT', 'CD163', 'CEACAM1', 'F3', 'FSTL3', 'FT',
    'IGFBP1', 'IL11', 'IFNL3', 'IL4R', 'IL6R', 'MCAM', 'OSM', 'GPNMB',
    'REG3A', 'RETN', 'S100A9', 'CLEC11A', 'TNFRSF13B', 'THBD', 'THBS2',
    'TNFRSF10B', 'TNFSF11', 'ADAMTS13', 'TNFSF13', 'MUC16', 'CHI3L1',
    'CSCL1', 'CXCL10', 'DKK1', 'EGF', 'ENPP2', 'GDF15', 'GH1', 'HGF',
    'IFNA1', 'IL13', 'IL18', 'IL23A', 'IL33', 'CSF1', 'MICA', 'MIF', 'PTX3',
    'SFTPD', 'IL1RL1', 'PLAUR', 'VWF', 'ANFPTL1', 'ANFPTL3', 'CCL11',
    'CCL24', 'CD40LG', 'C5', 'CXCL2', 'CXCL6', 'FABP4', 'LASLG', 'GZMA',
    'IL15', 'IL25', 'IL3', 'IL36B', 'IL7', 'LIF', 'SELL', 'MIA', 'NECTIN4',
    'NPHS1', 'SPP1', 'PDGFA', 'CD274', 'PRL', 'PCSK9', 'SELP', 'SDC1',
    'TFF3', 'TNFSF10', 'FLT1', 'TGFB1', 'BDNF', 'PDGFD', 'C1Q', 'CSCL12'
]

# ============================================================
# 🎯 LABEL MAP
# ============================================================
LABEL_MAP = {
    "Mild": 0,
    "Moderate": 1,
    "Severe": 2
}
