# src/preprocess_ehr.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import numpy as np

def main():
    print("="*60)
    print("      Merged 데이터 최종 전처리를 시작합니다.")
    print("      (로직: 컬럼 동적 선택 → 결측치 처리 → 상수 제거 → 표준화)")
    print("="*60)

    DATA_DIR = '/data/processed'
    INPUT_FILE = os.path.join(DATA_DIR, 'ehr_merged.csv')
    OUTPUT_FILE = os.path.join(DATA_DIR, 'ehr_processed.csv')

    SEVERE_THRESHOLD = 0.5


    print("✅ 1. 원본 파일 로드")
    df = pd.read_csv(INPUT_FILE, low_memory=False)


    numerical_features = [col for col in df.columns if col not in ['RID', 'DATE']]
    print(f"✅ 2. 처리 대상 숫자형 컬럼 {len(numerical_features)}개 자동 선택")


    print("✅ 3. 공백(whitespace)을 NaN으로 변환")
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)


    for col in numerical_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')


    print("✅ 4. 결측치 처리 시작")
    print("   - 1단계: 환자별 평균으로 채우기")
    df[numerical_features] = df.groupby('RID')[numerical_features].transform(lambda x: x.fillna(x.mean()))
    print("   - 2단계: 전체 평균으로 채우기")
    df[numerical_features] = df[numerical_features].fillna(df[numerical_features].mean())


    print("✅ 5. Target Leakage 방지를 위해 'CovSF_Score' 컬럼 제거")
    if 'CovSF_Score' in df.columns:
        df = df.drop(columns=['CovSF_Score'])
    if 'CovSF_Score' in numerical_features:
        numerical_features.remove('CovSF_Score')


    print("✅ 6. 상수 컬럼(standard deviation = 0) 제거")
    constant_cols = [col for col in numerical_features if df[col].nunique(dropna=False) <= 1]

    if constant_cols:
        print(f"   - 제거된 상수 컬럼: {constant_cols}")
        df = df.drop(columns=constant_cols)
        numerical_features = [f for f in numerical_features if f not in constant_cols]
    else:
        print("   - 상수 컬럼이 없습니다.")

    print("✅ 7. 특징(Features) 표준화")
    if numerical_features:
        scaler = StandardScaler()
        df[numerical_features] = scaler.fit_transform(df[numerical_features])

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "="*60)
    print(f"최종 학습용 데이터 생성 완료! 파일 위치: '{OUTPUT_FILE}'")
    print("="*60)

if __name__ == '__main__':
    main()
