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
