# 🌌 **Spectre.AI** — AI-Driven Pipeline for Exoplanet Atmospheric Spectra Extraction

> A modular, physics-informed machine learning framework developed for the **ESA Ariel Data Challenge 2024**, enabling accurate reconstruction of exoplanet spectra with robust uncertainty quantification.

> **Challenge**: Extract exoplanet atmospheric spectra from simulated telescope data

> **Host**: European Space Agency ([ESA Ariel Mission](https://www.esa.int/Science_Exploration/Space_Science/Ariel/Ariel_Data_Challenges))

> **Focus**: Machine learning pipeline for infrared spectroscopic data

---

## 🚀 Overview

The **Ariel mission** aims to study exoplanet atmospheres through infrared spectroscopy. This challenge simulates the mission's data to test algorithms for:

* **Jitter correction**
* **Noise reduction**
* **Spectral extraction**
* **Uncertainty quantification**

Our solution is a **modular, physics-informed machine learning pipeline** designed for high accuracy and robust uncertainty estimation.

---

## 🧠 Pipeline Architecture

```
📦 Raw Data
 └── 🔧 Preprocessing
      └── 🎯 Jitter Correction (TCN-JNet)
           └── 🧼 Denoising (PhySNet)
                └── 🌈 Spectral Extraction (Bayesian ResNet-1D)
                     └── 📉 Uncertainty Quantification
                          └── ✅ Final Spectrum + Uncertainty Bands
```

---

## ✨ Key Features

✅ **End-to-end ML Pipeline**
✅ **Temporal Convolutional Networks for jitter correction**
✅ **Physics-guided U-Net (PhySNet) for noise removal**
✅ **Bayesian ResNet-1D for spectral extraction**
✅ **Aleatoric + Epistemic uncertainty modeling**
✅ **Transfer learning with masked autoencoding (Ti-MAE)**
✅ **GPU-accelerated and modular design**

---

## 📁 Project Structure

```
.
├── configs/                # YAML configuration files
├── scripts/                # Training & inference scripts
├── src/                    # Core source code
│   ├── preprocessing/
│   ├── jitter_correction/
│   ├── denoising/
│   ├── spectral_extraction/
│   ├── uncertainty/
│   └── pipeline.py
├── main.py                 # Full pipeline runner
├── requirements.txt
└── README.md
```

---

## 📥 Installation

1. **Clone the repo:**

   ```bash
   git clone <repository-url>
   cd ariel-data-challenge-2024
   ```

2. **Set up environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🗂 Dataset Setup

Download the official dataset from the [Ariel Challenge Website](https://www.ariel-datachallenge.space/) and organize as follows:

```
data/
├── train/
│   ├── <planet_id>/
│   │   ├── AIRS-CH0_calibration/
│   │   ├── AIRS-CH0_signal.parquet
│   │   ├── FGS1_calibration/
│   │   └── FGS1_signal.parquet
├── test/
├── train_labels.csv
├── train_adc_info.csv
├── wavelengths.csv
└── axis_info.parquet
```

---

## ⚙️ Usage

### 🔁 Full Pipeline

```bash
python main.py --data_dir path/to/data --output_dir path/to/output
```

### 🏋️‍♀️ Train Models

```bash
python scripts/train.py --config configs/train_config.yaml --data_dir path/to/data --output_dir path/to/output
```

### 🔍 Run Inference

```bash
python scripts/infer.py --config configs/inference_config.yaml --data_dir path/to/data --models_dir path/to/models --output_dir path/to/submissions
```

---

## 🧪 Methods

### 🧹 Preprocessing

* Dark current subtraction
* Flat-field correction
* Bad pixel interpolation (U-Net or median filtering)
* Temporal normalization with robust Z-scoring

### 🛰 Jitter Correction – `TCN-JNet`

* Temporal Convolutional Networks (TCN)
* PSF-constrained cross-correlation with FGS1 signals

### 🧼 Denoising – `PhySNet`

* ConvNeXtV2-inspired U-Net
* Spectral convolutions in Fourier space
* Physics-based attention using PSF templates
* Ti-MAE pretraining

### 🌈 Spectral Extraction – `Bayesian ResNet-1D`

* Spectral Dropout + Template Attention
* Heteroscedastic noise modeling

### 📉 Uncertainty Quantification

* Aleatoric: Heteroscedastic variance
* Epistemic: Monte Carlo dropout
* Spectral smoothness priors
* Temperature scaling for calibration

---

## 📊 Evaluation Metric

**Gaussian Log-Likelihood (GLL)**

$$
\text{GLL} = -\frac{1}{2} \left( \log(2\pi\sigma^2) + \frac{(y_{\text{true}} - y_{\text{pred}})^2}{\sigma^2} \right)
$$

Higher GLL scores indicate better predictive accuracy and uncertainty calibration.

---

## 🏆 Results

> **Validation GLL Score**: `[score]`
> Our approach demonstrates reliable spectral reconstruction with calibrated uncertainties, placing it among top-tier solutions.

---

## 📜 License

\[Insert license here, e.g., MIT, Apache 2.0]

---

## 🙏 Acknowledgments

* **ESA** – European Space Agency
* The Ariel Science Consortium
