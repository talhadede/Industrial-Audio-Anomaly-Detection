# Industrial-Audio-Anomaly-Detection
Deep Learning-based anomaly detection for industrial machine sounds using PyTorch and DCASE 2020 dataset.

# Industrial Audio Anomaly Detection using Deep Learning 🎧🏗️

This project focuses on detecting anomalies in industrial machine sounds (valves, pumps, fans, etc.) using the **DCASE 2020 Challenge** dataset. The system leverages **PyTorch** to distinguish between normal operational sounds and potential mechanical failures, serving as a critical tool for predictive maintenance.

## 🚀 Technical Highlights

### 1. Advanced Neural Architectures
* **Multi-Layer Perceptron (MLP):** Implemented to establish a baseline for sound classification.
* **Convolutional Neural Networks (CNN):** Leveraged 2D representations of sound for advanced spatial feature extraction from spectrograms.

### 2. Signal Processing & Feature Engineering
* **Mel-Spectrograms:** Raw audio signals were converted into time-frequency representations using `Librosa` to capture complex acoustic patterns.
* **STFT Analysis:** Utilized Short-Time Fourier Transform (STFT) to map sound signals into the frequency domain.

### 3. Deep Learning Challenges & Insights
* **Vanishing Gradient Problem:** Analyzed and addressed gradient stability issues specifically within deeper MLP architectures implemented from scratch.
* **Model Convergence:** Demonstrated identical baseline accuracies (~82.45%) across different models, highlighting data-driven constraints in industrial anomaly detection.

## 🛠️ Tech Stack
* **Framework:** `PyTorch`
* **Signal Processing:** `Librosa`, `PyWavelets`
* **Data Science:** `NumPy`, `Matplotlib`, `Scikit-learn`
* **Dataset:** DCASE 2020 Challenge Task 2 (Industrial Machine Sounds)

## 📂 Dataset Note
Due to GitHub's file size limits and the dataset's large size (~5 GB), the raw audio files are not included in this repository. You can access the official dataset and task details here: [DCASE 2020 Challenge Website](https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sound).
