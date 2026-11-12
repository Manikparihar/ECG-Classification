# ECG Classification Using CNN and Keras (DenseNet121 + Grad-CAM)

**Course:** Image and Video Processing Lab (CSET344)  
**Institution:** School of Computer Science Engineering & Technology, Bennett University  
**Academic Session:** Odd Semester, November 2026  
**Guided By:** Dr Kumar Saurabh / Shuvo M  

---

## Team Members

| S.No. | Student Name | Enrollment No. | Role in Project |
| :---: | :--- | :--- | :--- |
| 1 | **Shubh Pratap Singh Parihar** | S24CSEU0941 | (Group Leader) — Model Architecture, Two-Stage Fine-Tuning & Checkpointing |
| 2 | **Harshiv Batra** | S24CSEU1536 | Dataset Pipeline, Contrast Enhancement, Clipping & Augmentation |
| 3 | **Twinkle Sahu** | S24CSEU0090 | Quantitative Evaluation, Confusion Matrix, Specificity & ROC-AUC Analysis |
| 4 | **Bhawani Singh** | S24CSEU2540 | Grad-CAM Visual Explainability & Windows CLI Inference Utility |

---

## 1. Project Overview

This repository contains an end-to-end 2D medical image processing and deep transfer learning pipeline for classifying 12-lead paper electrocardiogram (ECG) printouts into four cardiac diagnostic categories using **TensorFlow** and **Keras**:

1. **Class 0:** `ECG Images of Myocardial Infarction Patients (240x12=2880)`
2. **Class 1:** `ECG Images of Patient that have History of MI (172x12=2064)`
3. **Class 2:** `ECG Images of Patient that have abnormal heartbeat (233x12=2796)`
4. **Class 3:** `Normal Person ECG Images (284x12=3408)`

### Key Technical Features
* **Contrast-Calibrated High-Resolution Preprocessing:** Standardizes 12-lead ECG printouts to $384 \times 384 \times 3$ RGB tensors, applies deterministic linear contrast enhancement ($\text{contrast\_factor} = 1.3$), clips pixel intensities to $[0.0, 255.0]$, and normalizes values to $[0.0, 1.0]$ across training, validation, testing, and CLI inference splits.
* **Imbalance-Aware Transfer Learning (`ECG_DenseNet121`):** Uses an ImageNet-pretrained `DenseNet121` backbone ($7,301,956$ total parameters) with Global Average Pooling, `Dense(256, ReLU)`, `BatchNormalization`, `Dropout(0.40)`, and balanced inverse-frequency class weights.
* **Two-Stage Threshold-Guarded Training:**
  * **Stage 1 (Frozen Backbone Warmup):** Locks all $428$ `DenseNet121` layers and trains the classification head using $\text{Adam}(\text{lr} = 10^{-4})$, reaching $\text{val\_loss} = 1.0751$ ($\text{val\_accuracy} = 60.03\%$).
  * **Stage 2 (Partial Fine-Tuning):** Unfreezes the top $30$ layers while keeping `BatchNormalization` layers frozen ($22$ trainable backbone layers) at $\text{Adam}(\text{lr} = 5 \times 10^{-6})$ with `initial_value_threshold` protection, improving validation loss to $0.9955$.
* **Grad-CAM Explainability & Local CLI Predictor:** Computes spatial heatmaps from the final `Conv2D` feature map of `DenseNet121` and provides a standalone Windows CLI utility (`predict_ecg_2.py`) for single-image screening.

---

## 2. Repository Structure

    ECG-prototype/
    ├── end_2.ipynb                     # Complete training, evaluation, Grad-CAM & ZIP export notebook
    ├── predict_ecg_2.py                # Standalone Windows CLI inference script for single ECG images
    ├── requirements.txt                # Python dependencies for local CLI and notebook execution
    ├── README.md                       # Project documentation and step-by-step execution guide
    └── ECG_Models_Bundle/              # Extracted from ECG_Models_Bundle.zip
        ├── ecg_densenet121.keras       # Trained Keras v3 model (27.85 MB)
        ├── ecg_densenet121.weights.h5  # Trained model weights file
        ├── class_names.json            # Ordered list of the 4 diagnostic class names
        └── ecg_saved_model/            # Exported TensorFlow SavedModel directory

---

## 3. Step-by-Step Execution Guide

### Part A: Running Local CLI Inference on Windows (Live Demo)

**Step 1: Clone the Repository & Download Model Artifacts**  
Clone this repository to your local machine (e.g., `G:\ECG-prototype\`) and extract `ecg_densenet121.keras` and `class_names.json` into the project root directory:

    git clone <your-github-repo-url>
    cd ECG-prototype

**Step 2: Create a Virtual Environment & Install Requirements**  
Ensure Python $3.10+$ is installed, then create and activate a virtual environment and install the dependencies from `requirements.txt`:

    python -m venv venv
    .\venv\Scripts\activate
    pip install -r requirements.txt

**Step 3: Check Model & JSON Paths in `predict_ecg_2.py`**  
Verify that `MODEL_PATH` and `CLASS_NAMES_PATH` at the top of `predict_ecg_2.py` point to your local files:

    MODEL_PATH = r"G:\ECG-prototype\ecg_densenet121.keras"
    CLASS_NAMES_PATH = r"G:\ECG-prototype\class_names.json"
    IMAGE_SIZE = (384, 384)

**Step 4: Run Prediction on a 12-Lead ECG Image**  
Execute `predict_ecg_2.py` by passing the path to any `.jpg` or `.png` ECG printout:

    python predict_ecg_2.py "G:\ECG-prototype\sample_ecg.jpg"

**Expected Console Output:**

    Loading ECG model...
    Model loaded successfully!

    ===========================================================================
                               ECG RESULT
    ===========================================================================
    Predicted class : ECG Images of Patient that have abnormal heartbeat (233x12=2796)
    Confidence      : 89.42%

    Class probabilities:
    ---------------------------------------------------------------------------
    ECG Images of Myocardial Infarction Patients (240x12=2880)        1.15%
    ECG Images of Patient that have History of MI (172x12=2064)       4.81%
    ECG Images of Patient that have abnormal heartbeat (233x12=2796)  89.42%
    Normal Person ECG Images (284x12=3408)                            4.62%
    ===========================================================================

---

### Part B: Reproducing Training & Grad-CAM on Kaggle (`end_2.ipynb`)

1. **Import Notebook:** Upload `end_2.ipynb` to [Kaggle Notebooks](https://www.kaggle.com/code).
2. **Attach Dataset:** Click **+ Add Input** in the right-hand sidebar and add the [Kaggle ECG Image Dataset (`kanishkarathore1604/ecg-image`)](https://www.kaggle.com/datasets/kanishkarathore1604/ecg-image).
3. **Select GPU Accelerator:** Open **Settings -> Accelerator** and select **GPU T4 x2** or **GPU P100**.
4. **Run All Cells:**
   * **Cells 1–5:** Loads packages, checks dataset paths ($2,420$ train, $603$ validation, $928$ test images), initializes $384 \times 384$ contrast-clipped `ImageDataGenerator` pipelines, and computes balanced class weights.
   * **Cells 6–10:** Builds `ECG_DenseNet121`, runs Stage 1 head training and Stage 2 partial fine-tuning, restores `best_ecg_densenet121.keras`, and plots accuracy/loss curves.
   * **Cells 11–15:** Evaluates the model on all $928$ test images and outputs precision, recall, specificity, F1-score, the $4 \times 4$ confusion matrix, and weighted One-vs-Rest ROC-AUC.
   * **Cells 18–20:** Runs `make_gradcam_heatmap()` and `show_gradcam()` to display the Original ECG, Grad-CAM heatmap, and Superimposed Activation Overlay.
   * **Cell 21:** Bundles `.keras`, `.weights.h5`, `SavedModel`, and `class_names.json` into `/kaggle/working/ECG_Models_Bundle.zip`.

---

## 4. Held-Out Test Set Results ($n = 928$ Images)

| Diagnostic Category | Precision | Recall (Sensitivity) | Specificity | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0: Myocardial Infarction (MI)** | $1.0000$ | $0.2678$ | $1.0000$ | $0.4224$ | $239$ |
| **1: History of MI Patients** | $0.3990$ | $0.8837$ | $0.6971$ | $0.5497$ | $172$ |
| **2: Abnormal Heartbeat** | $0.8901$ | $0.6953$ | $0.9712$ | $0.7807$ | $233$ |
| **3: Normal Person ECG** | $0.6246$ | $0.6620$ | $0.8245$ | $0.6427$ | $284$ |
| **Macro Average** | **$0.7284$** | **$0.6272$** | **$0.8732$** | **$0.5989$** | **$928$** |
| **Weighted Average** | **$0.7461$** | **$0.6099$** | **—** | **$0.6034$** | **$928$** |

* **Overall Test Accuracy:** $60.99\%$ ($+9.48\%$ gain over uncalibrated baseline)
* **Weighted One-vs-Rest (OvR) ROC-AUC:** $0.9308$