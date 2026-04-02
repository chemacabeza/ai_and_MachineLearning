import numpy as np
import json, hashlib, os
from datetime import datetime
print("=== Data Management for ML: Pipeline Simulation ===\n")
np.random.seed(42)
raw_data = np.random.randn(100, 5)
raw_data[10, 2] = np.nan
raw_data[25, 0] = np.nan
raw_data[50, 4] = 999.0
raw_data[75, 1] = -888.0
print("Step 1: Data Ingestion")
print(f"   Raw samples: {len(raw_data)}, Features: {raw_data.shape[1]}")
missing = np.isnan(raw_data).sum()
print(f"   Missing values: {missing}")
print("\nStep 2: Data Cleaning")
for col in range(raw_data.shape[1]):
    col_data = raw_data[:, col]
    mask = np.isnan(col_data)
    if mask.any():
        col_mean = np.nanmean(col_data)
        col_data[mask] = col_mean
        print(f"   Column {col}: Imputed {mask.sum()} missing values with mean={col_mean:.3f}")
    q1, q3 = np.percentile(col_data, [25, 75])
    iqr = q3 - q1
    outliers = (col_data < q1 - 3*iqr) | (col_data > q3 + 3*iqr)
    if outliers.any():
        col_data[outliers] = np.clip(col_data[outliers], q1 - 3*iqr, q3 + 3*iqr)
        print(f"   Column {col}: Clipped {outliers.sum()} outliers")
raw_data[:, :] = raw_data
print(f"\nStep 3: Feature Engineering")
raw_data_ext = np.column_stack([raw_data, raw_data.mean(axis=1), raw_data.std(axis=1)])
print(f"   Added 2 aggregate features → Total features: {raw_data_ext.shape[1]}")
print(f"\nStep 4: Data Versioning")
data_hash = hashlib.sha256(raw_data_ext.tobytes()).hexdigest()[:12]
version = f"v1.0-{data_hash}"
print(f"   Dataset version: {version}")
print(f"   Timestamp: {datetime.now().isoformat()}")
print(f"\nStep 5: Train/Val/Test Split")
n = len(raw_data_ext)
train, val, test = raw_data_ext[:int(0.7*n)], raw_data_ext[int(0.7*n):int(0.85*n)], raw_data_ext[int(0.85*n):]
print(f"   Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
print(f"\nStep 6: Quality Report")
print(f"   Final missing values: {np.isnan(raw_data_ext).sum()}")
print(f"   Final shape: {raw_data_ext.shape}")
print(f"   Feature ranges: min={raw_data_ext.min():.3f}, max={raw_data_ext.max():.3f}")
print(f"\n✅ Data pipeline complete! Dataset {version} is ready for training.")
