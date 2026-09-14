# SignalScope: Telling Real From Synthetic in the Age of Generative Media

**Smart India Hackathon 2026 [Internal Hackathon]**  
*L. J. Institute of Engineering and Technology [C-433]*

---

## 1. Modules Built
* **Core Task:** Real-vs-AI-generated image classification with calibrated confidence (Stage 1 contract initialized).
* **Minimal Interface:** Mandatory minimal UI for prediction (`src/ui/app.py` - scheduled for Stage 5).
* **Bonus Modules:**
  * *Module A (Headline Bonus):* Faithful Explanation (Grad-CAM + grounded cues - scheduled for Stage 8).
  * *Module C:* Robustness to degradation (scheduled for Stage 9).
  * *Module D:* EXIF provenance metadata (conditional - scheduled for Stage 10).

---

## 2. Setup & Reproduction Instructions (<10 Minutes)

### Prerequisites
* Windows 11 / Linux / macOS
* Python 3.11 isolated virtual environment

### 1-Click Execution (Windows)
```cmd
run_demo.bat path\to\image.jpg
```

### Manual Setup
```bash
# 1. Create Python 3.11 virtual environment
py -3.11 -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Run prediction via standardized predict interface
python src/model/predict.py --image path/to/image.jpg
```

---

## 3. Datasets & Citations
* **Provided Dataset:** Kickoff real-vs-synthetic dataset (~100k images, CIFAKE-style).
* **Public Datasets Used:** Disclosed and cited upon integration in Stage 2.
* **Held-Out Test Set:** Completely isolated; used solely for final official evaluation.

---

## 4. Reported Metrics

| Metric | Internal Validation (Seen) | Internal Val (Unseen Proxy) | Official Held-Out Test Set |
| :--- | :---: | :---: | :---: |
| **ROC-AUC (Overall)** | *TBD (Stage 4)* | *TBD (Stage 4)* | *TBD (Final Benchmark)* |
| **ROC-AUC (Unseen Split)** | N/A | *TBD (Stage 4)* | *TBD (Final Benchmark)* |
| **Macro-F1** | *TBD (Stage 4)* | *TBD (Stage 4)* | *TBD (Final Benchmark)* |
| **Accuracy (at threshold)** | *TBD (Stage 4)* | *TBD (Stage 4)* | *TBD (Final Benchmark)* |
| **False Positive Rate (FPR)** | *TBD (Stage 4)* | *TBD (Stage 4)* | *TBD (Final Benchmark)* |

---

## 5. Architecture & Approach Overview
* **Candidate Backbones:** ConvNeXt-Tiny / EfficientNet-V2-S.
* **Calibration:** Temperature Scaling fit strictly on validation data.
* **Generalization Strategy:** Frequency/blur/compression augmentations during training.
* **Limitations & Honest Disclosure:** Documented upon evaluation in the Model Report (`report/model_report.pdf`).

---

## 6. Demonstration Video & Deployed App
* **Demo Video (3–5 minutes):** `[Link to Demo Video](<DEMO_VIDEO_URL_PLACEHOLDER>)`  
  *(Hosting platform to be finalized per official SIH submission guidelines).*
* **Deployed App:** *(To be added upon deployment).*

---

## 7. Originality Declaration
This repository adheres to SIH 2026 originality rules:
* All substantive work is developed within the hackathon timeframe (10–15 September).
* Pretrained backbones (`timm`, `torchvision`) and open-source libraries (`torch`, `scikit-learn`, `numpy`) are utilized as disclosed.
* No public real-vs-fake detection notebooks have been copied wholesale.
