# Drone Flight Anomaly Detection (One Marker Point)

🚀 **Experimental prototype** of an ML-based anomaly detection system for quadcopter flight data using **one marker point**.  
This project focuses on analyzing simulated drone motion data (velocity & acceleration in X, Y, Z axes) to detect anomalies using a variety of ML models.

---

## 🔍 Project Overview
- Designed an anomaly detection system for quadcopter motion data captured via **MATLAB simulations** and **motion-capture inputs**.
- Implemented data preprocessing, feature engineering, and ML pipeline in **Python**.
- Evaluated anomaly detection models with strong results:
  - **Isolation Forest**
  - **Gaussian Process Regression (GPR)**
  - **Autoencoder**
  - **LSTM (Long Short-Term Memory)**
  - **Ensemble Models**

📊 Achieved **92% detection accuracy** with a **5% false positive rate**.

---

## ⚙️ Features
- **Data Preprocessing**: Cleaning, normalization, feature extraction.  
- **Anomaly Detection Models**: Isolation Forest, Autoencoder, LSTM, GPR.  
- **Visualization Tools**: Interactive plots for anomalies in X, Y, Z axes.  
- **Evaluation**: Confusion matrices, precision/recall metrics.  

---

## 🛠️ Tech Stack
- **Languages**: Python  
- **Libraries**:  
  - `OpenCV`, `NumPy`, `Pandas`, `Seaborn`, `Matplotlib`  
  - `TensorFlow`, `scikit-learn`, `pyod`  
  - `Autoencoder`, `LSTM`, `Ensemble Models`  
  - `Confusion-Matrix`, `Isolation Forest`, `Gaussian Process Regression`

---

## 📈 Results
- Detected anomalies in **1-marker motion data** effectively.  
- Provided clear anomaly visualizations → reducing physical test flights by **100%**.  
- Improved stability assessment speed by **40%**.
