<div align="center">
  <img src="cover.png" alt="AI for IoT Cover" width="800"/>
</div>

# Chapter 17: AI for the Internet of Things

**🎯 The Big Goal:** Understand how to deploy lightweight AI models on edge devices to process sensor data in real-time — enabling smart homes, factories, and wearables to make instant decisions without cloud connectivity.

## Core Concepts

The Internet of Things (IoT) connects billions of devices — thermostats, cameras, industrial sensors, wearables — to the internet. Adding AI to these devices enables them to make intelligent decisions locally.

### Edge AI vs. Cloud AI

| | Cloud AI | Edge AI |
|---|---------|---------|
| **Latency** | 100–500ms (network round-trip) | 1–10ms (local processing) |
| **Privacy** | Data leaves the device | Data stays on-device |
| **Bandwidth** | Requires constant internet | Works offline |
| **Power** | Unlimited compute | Battery-constrained |

### Anomaly Detection

The most common AI task for IoT is **anomaly detection**: continuously monitoring sensor readings and flagging unusual patterns. A factory temperature sensor normally reads 20–24°C. If it suddenly reads 35°C, that's an anomaly that could indicate equipment failure.

The simplest approach uses **statistical methods**: compute a rolling mean and standard deviation, then flag any reading that deviates more than N standard deviations from the mean (z-score method).

---

## 🤔 Reflection Questions

<details>
<summary>💡 View Answer: Why can't we just send all IoT data to the cloud for processing?</summary>

Three reasons: (1) **Latency** — a self-driving car cannot wait 500ms for a cloud response when it needs to brake immediately. (2) **Bandwidth** — a factory with 10,000 sensors each producing 100 readings/second generates 1 million data points per second — too much for most internet connections. (3) **Privacy** — health wearables and home cameras contain sensitive data that users may not want leaving their devices.
</details>

<details>
<summary>💡 View Answer: What is model compression and why does it matter for IoT?</summary>

IoT devices have limited memory (often kilobytes, not gigabytes). Model compression techniques — quantization (reducing weight precision from 32-bit to 8-bit), pruning (removing unnecessary connections), and knowledge distillation (training a tiny student model to mimic a large teacher) — shrink models to fit on microcontrollers while preserving most of the accuracy.
</details>

---

## 🐳 Hands-On Exercise: IoT Sensor Anomaly Detection

### Step 1: Build
```bash
cd exercise
docker build -t ch17-iot .
```

### Step 2: Run
```bash
docker run --rm ch17-iot
```

### Dockerfile
```dockerfile
FROM python:3.9-alpine
WORKDIR /app
RUN pip install numpy
COPY iot_anomaly.py /app/
CMD ["python", "iot_anomaly.py"]
```

### Source Code

```python
import numpy as np
print("=== AI for IoT: Anomaly Detection in Sensor Data ===\n")
np.random.seed(42)
hours = 168
normal_temp = 22 + 2 * np.sin(np.linspace(0, 14*np.pi, hours)) + np.random.normal(0, 0.5, hours)
anomaly_idx = [50, 51, 100, 101, 140]
for i in anomaly_idx:
    normal_temp[i] += np.random.choice([-8, 8])
window = 12
rolling_mean = np.convolve(normal_temp, np.ones(window)/window, mode='valid')
rolling_std = np.array([np.std(normal_temp[max(0,i-window):i]) for i in range(window, len(normal_temp))])
threshold = 2.5
anomalies_found = []
print(f"Monitoring {hours} hours of temperature data (window={window}h):\n")
for i in range(len(rolling_mean)):
    actual = normal_temp[i + window]
    z_score = abs(actual - rolling_mean[i]) / (rolling_std[i] + 1e-8)
    if z_score > threshold:
        anomalies_found.append(i + window)
        print(f"  🚨 Hour {i+window:3d}: Temp={actual:5.1f}°C (expected ~{rolling_mean[i]:5.1f}°C, z={z_score:.1f})")
print(f"\n📊 Results:")
print(f"   Total readings: {hours}")
print(f"   Anomalies detected: {len(anomalies_found)}")
print(f"   True anomaly hours: {anomaly_idx}")
print(f"   Detected hours:     {anomalies_found}")
true_pos = len(set(anomalies_found) & set(anomaly_idx))
print(f"   True positives: {true_pos}/{len(anomaly_idx)}")
print(f"\n✅ Edge AI detected sensor anomalies in real-time!")
```
