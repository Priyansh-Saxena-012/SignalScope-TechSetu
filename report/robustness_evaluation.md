# SignalScope Robustness Evaluation Report (Stage 12)

> [!IMPORTANT]
> **Development Data Evaluation Only**: This robustness evaluation is conducted strictly on a deterministic, stratified development validation partition from Tiny-GenImage. The official 100,000-image held-out benchmark (`C:\Datasets\SignalScope\test`) remains completely untouched, pristine, and isolated.

> [!NOTE]
> **Degradation vs Adversarial Robustness**: This evaluation assesses robustness to organic social-media distribution channels (JPEG compression, bilinear resizing/restoration, optical blur, and light benign edits). It does not measure worst-case norm-bounded gradient adversarial attacks (e.g. FGSM/PGD).

---

## 1. Executive Summary & Experimental Methodology

- **Model Architecture**: Vision Transformer Base (`vit_base_patch16_224`, 85.8M parameters)
- **Checkpoint**: `model/weights/checkpoint_best.pth` (FROZEN Epoch-5 Checkpoint)
- **Decision Threshold**: Strictly fixed at **0.50** across all evaluations (no per-condition threshold tuning)
- **Evaluation Preprocessing**: Unchanged deterministic evaluation transforms ($\text{Resize}(256) \to \text{CenterCrop}(224) \to \text{Normalize}$)
- **Development Cohort**: Exactly **350** stratified samples:
  - **175 Real images** (ImageNet validation partition)
  - **175 AI images** (25 each across 7 generator families: `adm`, `biggan`, `glide`, `midjourney`, `sd15`, `vqdm`, `wukong`)
- **Pairwise Comparative Baseline**: Every transformed version is paired and compared against the identical clean original image.

---

## 2. Robustness Conditions & Defensible Parameters

1. **JPEG Compression**:
   - Quality levels: $Q \in \{95, 75, 50, 30\}$
   - Simulates near-lossless re-save ($Q=95$), standard social web uploads ($Q=75$), messaging app compression ($Q=50$), and heavy bandwidth reduction ($Q=30$).
2. **Bilinear Downscaling & Restoration**:
   - Scale factors: $s \in \{0.75, 0.50, 0.25\}$, restored to original pixel dimensions via bilinear resampling.
   - Evaluates spatial frequency loss and resolution invariance without altering tensor ingestion shapes.
3. **Gaussian Blur (Low-Pass Filtering)**:
   - Kernel standard deviation: $\sigma \in \{0.5, 1.0, 2.0\}$ pixels.
   - Measures detector reliance on high-frequency pixel fingerprints versus macro-structural forensic anomalies.
4. **Screenshot-Like Degradation Proxy**:
   - Controlled multi-stage proxy: Bilinear downscale to $70\% \to$ Bilinear restoration $\to$ JPEG quality $80$.
   - *Scope Note:* This is a controlled proxy simulating display rasterization and re-save; it does not claim to identically model proprietary platform transcoding algorithms.
5. **Light Benign Photometric Edits**:
   - Brightness adjustments: Factor $1.15$ (+15%) and $0.85$ (-15%).
   - Contrast adjustments: Factor $1.15$ (+15%) and $0.85$ (-15%).
   - Assesses false-alarm stability on legitimate consumer camera photos.

---

## 3. Overall Robustness Benchmark Matrix

*Status: Empirical evaluation pending execution. Placeholders below illustrate the final reporting schema.*

| Condition | Degradation Parameter | ROC-AUC | Δ AUC | Macro-F1 (τ=0.5) | Accuracy | Flip Rate | Evasion Rate (AI→Real) | False Alarm (Real→AI) | MAPS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `clean` | *None (Baseline)* | *Pending* | *0.0000* | *Pending* | *Pending* | *0.0%* | *0.0%* | *0.0%* | *0.0000* |
| `jpeg_q95` | Quality = 95 | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| `jpeg_q75` | Quality = 75 | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| `jpeg_q50` | Quality = 50 | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| `jpeg_q30` | Quality = 30 | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| `downscale_0.75` | Scale = 0.75 | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| `downscale_0.50` | Scale = 0.50 | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| `downscale_0.25` | Scale = 0.25 | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| `blur_s0.5` | Sigma = 0.5 | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| `blur_s1.0` | Sigma = 1.0 | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| `blur_s2.0` | Sigma = 2.0 | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| `screenshot_proxy` | Scale 0.70 + Q80 | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| `brightness_1.15` | Factor = 1.15 | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| `brightness_0.85` | Factor = 0.85 | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| `contrast_1.15` | Factor = 1.15 | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| `contrast_0.85` | Factor = 0.85 | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |

---

## 4. Generator-Wise Sensitivity Breakdown (Representative Conditions)

*Status: Empirical evaluation pending execution. Placeholders below illustrate the final reporting schema.*

| Generator Family | Clean Acc | JPEG Q50 Acc | JPEG Q50 Evasion | Blur σ=1.0 Acc | Blur σ=1.0 Evasion | Downscale 50% Acc | Downscale 50% Evasion |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ADM** | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| **BIGGAN** | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| **GLIDE** | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| **MIDJOURNEY** | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| **SD15** | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| **VQDM** | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| **WUKONG** | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |

---

## 5. Scope & Limitations

1. **Development Cohort**: Evaluated on 350 stratified development samples. Full held-out verification on the official 100,000-image test set remains pending GPU access.
2. **Organic vs Adversarial**: The detector is evaluated against natural transformations, not white-box adversarial perturbations.
3. **Static Image Scope**: Video compression artifacts (e.g. H.264 inter-frame motion vector quantization) are out of scope.
