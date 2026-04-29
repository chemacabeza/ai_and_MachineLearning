<div align="center">
  <img src="cover.png" alt="Data Management for ML Cover" width="800"/>
</div>

# Chapter 20: Data Management for ML Systems

**🎯 The Big Goal:** Learn the engineering practices behind production ML data pipelines — data versioning, quality monitoring, feature stores, and reproducible preprocessing — the invisible infrastructure that makes or breaks real AI systems.

## Core Concepts

In production ML, 80% of the work is data engineering. A model is only as good as its data. **Data Management for ML** covers the tools and practices that ensure your data is clean, versioned, and reproducible.

### The ML Data Pipeline

1. **Ingestion:** Collect raw data from databases, APIs, streams.
2. **Cleaning:** Handle missing values, remove duplicates, fix encoding errors.
3. **Validation:** Detect data drift (distribution changes), schema violations, outliers.
4. **Feature Engineering:** Transform raw data into model-ready features.
5. **Versioning:** Track which data was used to train which model.
6. **Splitting:** Create reproducible train/validation/test splits.

### Why Data Versioning Matters

Imagine your model works great in January but fails in February. Was it a code change or a data change? Without data versioning, you can't tell. Tools like DVC (Data Version Control) add git-like version tracking to datasets, so you can always reproduce any past training run.

### Data Drift

The world changes. Customer preferences shift, new products launch, pandemics happen. When the real-world data distribution diverges from training data, model accuracy silently degrades — this is **data drift**, ML's silent killer.

---

## 🤔 Reflection Questions

<details>
<summary>💡 View Answer: What is the difference between data drift and concept drift?</summary>

**Data drift** (covariate shift) is when the input distribution changes — e.g., your model trained on English text now receives Spanish text. **Concept drift** is when the relationship between input and output changes — e.g., what counts as "spam" evolves over time even though emails look similar. Both degrade model performance, but concept drift is harder to detect because the inputs may look normal.
</details>

<details>
<summary>💡 View Answer: Why is reproducibility critical in ML?</summary>

If you can't reproduce a training run, you can't debug failures, prove compliance with regulations, or compare experiments fairly. Reproducibility requires versioning code, data, model weights, hyperparameters, and the random seed. In regulated industries (healthcare, finance), non-reproducible models are often illegal to deploy.
</details>

---

## 🐳 Hands-On Exercise: ML Data Pipeline Simulation

### Step 1: Build
```bash
cd exercise
docker build -t ch20-data-mgmt .
```

### Step 2: Run
```bash
docker run --rm ch20-data-mgmt
```

### Dockerfile
```dockerfile
FROM python:3.9-alpine
WORKDIR /app
RUN pip install numpy
COPY data_pipeline.py /app/
CMD ["python", "data_pipeline.py"]
```

### Source Code

```python
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
```
