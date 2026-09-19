# PET Frame-Prediction Project — Full Context Handoff

**Purpose of this file:** complete state of the project so work can continue in a
new conversation with no loss of context. Everything below is verified against
the actual result files, not recalled from memory.

---

## 1. The project in one paragraph

Fourteen predictive models are benchmarked on their ability to forecast future
frames of a **simulated dynamic FDG-PET time series** from the first 10 frames
only. Data comes from a multi-scale biomathematical model (COMSOL), not from
patients. Evaluation is leave-one-tumour-out cross-validation over 10 tumours.
The headline finding is not the leaderboard but an ablation: **fixing four
implementation choices in a conditional VAE moved it from 0.303 to 0.821 SSIM —
a larger gap than the entire spread between the ten working architectures.**

Deliverable so far: a 28-page Persian BSc thesis (KNTU template) plus the full
experimental toolchain.

---

## 2. Data

| Property | Value |
|---|---|
| Source | Multi-scale biomathematical model; angiogenesis → interstitial flow → FDG convection-diffusion-reaction → SUV field. Solved in COMSOL Multiphysics (FEM, MUMPS, BDF). |
| Provenance | Method matches Abazari et al., *Cancers* 2022, 14, 2786 (same university, KNTU). Confirm with supervisor whether the student ran COMSOL or received the images. |
| Nature | **100% simulated. No patient data.** Must be stated in title/abstract. |
| Tumours | 10 independent capillary networks, 1–3 cm diameter |
| Frames | 130–150 per tumour (median 150) after preprocessing |
| **Time unit** | **MINUTES — confirmed by the user.** Max timepoint 4470 min. Figures labelled `min` are correct. |
| Image | 64×64×3, **jet colormap** (not scalar activity) |
| Location | `C:\IMAGES\<tumour>\cleaned\resized\` |
| Tumour folder names | `t1-r=0.1`, `T1-R=0.2`, `T1-R=0.3`, `T1-R=0.4`, `T2-R=0.1`, `T2-R=0.2`, `t2-r=0.3`, `T2-R=0.4`, `T3-R=0.2`, `T3-R=0.1` |
| R parameter | **User confirmed the R values are unrelated to each other** — all 10 are independent tumours, not 3 tumours × 4 conditions. This was checked explicitly; it determines that plain LOOCV (not leave-one-group-out) is correct. |

### Preprocessing chain
1. COMSOL exports SUV field **with black vessel outlines and a colourbar**
2. Crop to field of view (removes colourbar/axes)
3. **Remove vasculature by inpainting** — the black lines are a graphical overlay, not physics. Leaving them lets the network memorise vessel geometry instead of learning kinetics, and they distort SSIM as high-contrast edges.
4. Drop the last ~50 frames (constant blue plateau, washout complete — no temporal information, biases the model toward flat outputs)
5. Resize to 64×64 with `INTER_AREA` (correct for downsampling colormapped images; bilinear causes aliasing)
6. Normalise per-frame to [−1, 1] — **per-frame, so no cross-fold statistic leaks**

### Task definition
Given the first 10 frames as context, predict the frame at an arbitrary later
time. Context covers only ~8% of the time axis; prediction runs to 100%.
**Prediction horizon ≈ 16× the width of the observed window.** This is the
single most important fact for interpreting every result.

---

## 3. Evaluation design

- **Leave-one-tumour-out CV, 10 folds.** Per fold: 7 train / 2 validation / 1 test, all splits **at tumour level**.
- **Rotating validation set.** A fixed seeded permutation; validation = next 2 entries cyclically after the test tumour. Verified: every tumour is validation in exactly 2 folds, training in 7.
  - *Why this mattered:* the original code used `remaining[:2]`, which put two specific tumours in validation in **9 of 10 folds** — they almost never contributed to training, and the other eight never influenced early stopping.
- **Unit of analysis = tumour, not frame.** Metrics averaged within tumour; 95% CI by **cluster bootstrap resampling tumours** (10,000 resamples). Frame-level bootstrap would give CIs several times too narrow.
- **Tests:** Wilcoxon signed-rank paired by tumour, Holm–Bonferroni.
- **Two pre-specified families:**
  - *Confirmatory:* 14 model-vs-baseline comparisons → p_holm = 0.0273, significant
  - *Exploratory:* 120 all-pairs comparisons → **0 significant, and this is arithmetic, not a finding.** Minimum attainable two-sided p at n=10 is 2/2¹⁰ = 0.00195; × 120 = 0.234. No pairwise comparison *can* reach significance regardless of effect size.
- Hyperparameters fixed a priori, identical across architectures. No tuning against test folds.
- Epoch budget: 400 (diffusion 1200), early stopping patience 60, min epochs 80.

---

## 4. FINAL RESULTS (RGB metric space, all 14 models, one consistent sweep)

| Model | SSIM | 95% CI | PSNR (dB) | MAE | Beats baseline |
|---|---|---|---|---|---|
| **cVAE-2** | **0.8213** | 0.779–0.858 | 24.82 | 0.041 | ✔ |
| SimVP | 0.7886 | 0.747–0.823 | 23.30 | 0.047 | ✔ |
| Swin-UNet | 0.7798 | 0.733–0.816 | 23.50 | 0.048 | ✔ |
| cGAN | 0.7738 | 0.722–0.819 | 22.92 | 0.051 | ✔ |
| Diffusion | 0.7696 | 0.724–0.812 | 24.06 | 0.046 | ✔ |
| Attention U-Net | 0.7476 | 0.700–0.790 | 21.05 | 0.056 | ✔ |
| ConvLSTM | 0.7429 | 0.699–0.784 | 21.61 | 0.058 | ✔ |
| U-Net | 0.7422 | 0.701–0.779 | 21.33 | 0.057 | ✔ |
| Temporal U-Net | 0.7283 | 0.687–0.766 | 20.97 | 0.058 | ✔ |
| ViT | 0.7090 | 0.661–0.752 | 21.72 | 0.054 | ✔ |
| Pix2Pix | 0.5757 | 0.555–0.598 | 15.63 | 0.121 | ✔ |
| *Baseline: last-frame* | *0.3667* | *0.341–0.396* | *10.34* | *0.250* | — |
| Attention-ResNet | 0.3073 | 0.267–0.346 | 11.91 | 0.167 | ✘ (n.s., p=0.193) |
| cVAE-1 | 0.3033 | 0.282–0.326 | 11.87 | 0.184 | ✘ (worse) |
| *Baseline: mean-frame* | *0.2697* | *0.246–0.295* | *9.15* | *0.287* | ✘ |
| Kinetic fit | 0.2541 | 0.236–0.276 | 9.72 | 0.261 | ✘ (worse) |

**11 models beat baseline, every one in 10/10 folds, Wilcoxon stat = 0.0, p_holm = 0.0273.**

### The three controlled ablations (the paper's real contribution)

| Comparison | Isolates | Δ SSIM | Folds |
|---|---|---|---|
| cVAE-2 − cVAE-1 | Implementation only (same architecture family) | **+0.518** | 10/10 |
| cGAN − Pix2Pix | Explicit time conditioning + SSIM loss, same GAN backbone | **+0.198** | 10/10 |
| Swin-UNet − ViT | Windowed local vs global attention | **+0.071** | 10/10 |

The +0.518 exceeds the 0.11 spread across all ten working architectures.
**Implementation dominated architecture at this data scale.**

### Overfitting diagnostics

| Model | val_degradation | % early stopped | mean stop epoch | val↔test r |
|---|---|---|---|---|
| ViT | 0.017 | 80 | 269 | 0.083 |
| cVAE-2 | 0.020 | 100 | 170 | 0.164 |
| Temporal U-Net | 0.020 | 100 | 216 | −0.178 |
| Attention-ResNet | 0.023 | 80 | 206 | **0.898** |
| ConvLSTM | 0.030 | 60 | 333 | 0.146 |
| Swin | 0.058 | 100 | 115 | −0.003 |
| cGAN | 0.069 | 100 | 214 | 0.165 |
| SimVP | 0.072 | 100 | 126 | 0.093 |
| U-Net | 0.153 | 100 | 182 | 0.223 |
| Diffusion | 0.156 | 100 | 407 | 0.491 |
| Attention U-Net | 0.200 | 100 | 149 | −0.132 |
| cVAE-1 | 0.228 | 100 | **80** (= min_epochs floor) | −0.066 |

**Interpretation:**
- Overfitting is **not** a problem. Max 22.8% validation degradation; median ~6%. Early stopping fired in 60–100% of folds. Budget was adequate.
- `val_over_train` is **not comparable across models** (each has a different loss on a different scale — cGAN's carries a 100× L1 term, hence its meaningless 0.04). Use `val_degradation` only.
- **Model-selection limitation:** val↔test correlation is weak for most models (−0.18 to +0.22; |r|>0.63 needed for significance at n=10). With 2 validation tumours, the selection signal reflects *which* tumours landed in validation, not generalisation. Checkpoint choice was approximately arbitrary within the converged region. **This belongs in Limitations.**
- ViT has the *lowest* degradation and mediocre accuracy → **underfitting, not virtue.**
- cVAE-1 stopped at exactly the min_epochs floor (80) in every fold with the highest degradation → **signature of posterior collapse.**

---

## 5. The colormap / metric-space issue (important, unresolved)

Frames are **jet-colormapped**, not scalar activity. Jet is a nonlinear,
**non-monotonic** function of activity: as activity falls, blue rises while red
falls. Consequences:

1. Per-channel SSIM/PSNR/MAE on RGB are not physically meaningful.
2. Per-voxel kinetic fitting in RGB space is **invalid outright** — washout is monotonic in activity but not in R, G or B.

`pet_common.py` can invert the colormap (nearest-neighbour LUT match; verified
round-trip error 0.13%, 0.42% with 2% noise) and computes **both** metric spaces,
saving predictions to `fold_outputs.npz`.

**Measured impact** (activity − RGB): kinetic **+0.204**, diffusion +0.113,
swin +0.090, cvae2 +0.077; but ~0.00 for models whose errors are structural
rather than chromatic.

**The reversal worth reporting:** in RGB the kinetic baseline scores 0.254 and
*loses* to last-frame-repeat; in activity space it scores 0.458 and *wins*.
How you measure colormapped PET predictions changes the conclusion about
whether classical kinetic modelling works at all.

**Current state:** the final table is **all-RGB**, because only the 6
`pet_common` models save predictions. The 9 original scripts compute their own
RGB metrics and ignore `PET_METRIC_SPACE`. A previous run mixed the two spaces
and had to be rebuilt with `06_rebuild_metrics.py --space rgb`.

**To get activity metrics for all 14:** patch the 9 originals to call
`pet_common.finalize_fold`, then one more full sweep (~16–24 h).

---

## 6. Open issues

1. **Attention-ResNet at 0.307** — the only model statistically indistinguishable from the trivial baseline, while siblings reach 0.74–0.75. Diagnostics are odd: lowest val/train ratio, low degradation, *highest* val↔test correlation (0.898). Converges consistently to something bad → suggests a bug. **Inspect `RESULTS_final\attention_resnet\loocv\*\comparison.png` before submitting.** Do not publish "attention-ResNet fails" on the back of a bug.
2. **Only 10 tumours.** Since data is simulated, the obvious referee question is "why not generate 100?" At n=30 the power ceiling disappears and architectures could actually be ranked. **Highest-value next step if the simulator is available.**
3. **Metric space** — decide RGB-only, or re-run for activity across all models.
4. **Fairness note:** cVAE-2 received targeted debugging the others did not. Mitigated by the fact that all 9 originals already use L1+SSIM (verified — no MSE handicap), so the comparison is defensible. Disclose λ_SSIM varies 0.3–0.5 per model in a supplementary hyperparameter table.

---

## 7. Toolchain (all on `C:\Pet_Code`, Windows, GTX 1660 6 GB, torch 2.11.0+cu128)

| File | Role |
|---|---|
| `START_HERE.bat` | one-click: env setup → paths → patch → smoke test |
| `RUN_ALL_FINAL.bat` | full sweep → stats → overfitting report |
| `01_setup_env.bat` | venv + CUDA PyTorch (**must use `--index-url .../cu128`; plain pip gives CPU-only on Windows**) |
| `00_set_paths.py` | sets DATA_ROOT in all 9 scripts, verifies data, diagnoses failures |
| `02_patch_scripts.py` | applies rotating-validation fix + env overrides + cudnn.benchmark; `--revert` restores |
| `03_run_all.py` | orchestrator: subprocess per model, resumable, OOM retry, auto-detects venv |
| `_pet_worker.py` | per-model worker; monkeypatches `train_one_fold` to capture per-frame metrics |
| `04_compare_models.py` | cluster bootstrap, Wilcoxon + Holm, power-ceiling warning, paper text |
| `05_overfitting_report.py` | parses logs for degradation / early stop / val↔test r |
| `06_rebuild_metrics.py` | rewrites per-frame CSVs in one consistent metric space from npz |
| `pet_common.py` | shared framework for the 6 new models: data, colormap inversion, metrics, LOOCV driver |
| New models | `pet_cvae2_paper.py`, `pet_swin_paper.py`, `pet_diffusion_paper.py`, `pet_cgan_paper.py`, `pet_simvp_paper.py`, `pet_kinetic_baseline_paper.py` |
| Originals (9) | `pet_unet_paper.py`, `pet_attention_unet_paper.py`, `pet_attention_resnet_paper.py`, `pet_temporal_unet_paper.py`, `pet_convlstm_paper.py`, `pet_cvae_paper.py`, `pet_vit_paper.py`, `pet_pix2pix_paper.py`, `pet_naive_baseline_paper.py` |

### Model design notes worth preserving
- **cVAE-2 fixes** (the +0.518): KL warm-up + free bits (posterior collapse); deterministic inference at prior mean (random z per frame caused the global colour tint); L1+SSIM instead of MSE (MSE → conditional mean → washed out); U-Net skip conditioning so the latent carries only the unexplained part.
- **Diffusion rebuild** (0.355 → 0.770): predicts the **residual from the last context frame**, not the whole image from noise; **v-parameterisation** instead of ε; **EMA weights** (decay 0.999); DDIM 100 steps; 3× epoch budget.
- **cGAN vs Pix2Pix:** cGAN = pix2pix + sinusoidal time conditioning + SSIM term. Early stopping on **reconstruction** loss, never adversarial loss (a moving target, not comparable across epochs).
- **Swin:** windowed attention + shifted windows + U-Net skips. ViT kept as ablation showing what the locality bias is worth.
- **Kinetic baseline:** amplitude-weighted log-linear fit of `y = a·exp(−k·t)` per voxel, k clamped, James-Stein shrinkage toward the image-wide median decay for poorly-fitted voxels. Fails because of the 16× extrapolation.

### Environment gotchas already solved
- Windows `pip install torch` → CPU-only. Must use the CUDA index URL.
- Console code page cp1256 → `UnicodeEncodeError` on box-drawing chars. Fixed by forcing UTF-8 with `errors=replace`.
- OneDrive-synced Desktop → `OSError [Errno 22]` writing PNGs, which killed a whole trained fold. All figure/npz saves are now best-effort. **Keep the project off the Desktop.**
- `make_run_loocv` must resolve `train_one_fold` from module globals at call time, or the worker's monkeypatch is bypassed and per-frame CSVs come out empty.
- Inline images in docx inherit 1.5 line spacing and get **clipped to a thin strip** — image paragraphs need explicit `lineRule: AUTO`.

---

## 8. Thesis status

`PET_Thesis_FA.docx` — 28 pages, Persian, KNTU template style. Names left as
dotted lines. Structure: title → abstract → TOC → 6 chapters → 20 references.
9 figures, 4 tables.

**Outstanding requests from the user (not yet done):**
1. **Architecture diagrams** for cGAN / diffusion / VAE / Swin — block diagrams showing data flow. None exist yet.
2. **More plots and tables** generally; "jazzed up" visual quality.
3. **Keep model names in English.** Do NOT translate to Persian —
   `خودرمزگذار وردشی شرطی` for "conditional VAE" reads badly. Use
   **cVAE, cGAN, Diffusion, Swin-UNet, SimVP, ViT, U-Net** as Latin text
   inline in the Persian prose.
4. Time unit is **minutes** — already correct in the document.

**Persian writing constraints:** formal academic register, strict grammar
(referees are strict). Font B Nazanin. Use `bidirectional: true` on paragraphs,
`rightToLeft: true` on runs, `visuallyRightToLeft: true` on tables.

---

## 9. Suggested framing for paper/defense

**Title:** Benchmarking Deep Learning Architectures for Temporal Frame
Prediction in **Simulated** Dynamic PET: A Leave-One-Tumour-Out Cross-Validation
Study.

**Lead with the ablation, not the leaderboard.** "Implementation choices
outweighed architectural choice" is a more useful contribution than ranking 14
models, and it is the only claim the statistics fully support.

**Do not claim a winner** among the top five (0.77–0.82). Differences are small
and, by the pre-specified plan, exploratory. Correct wording: *cVAE-2 achieved
the highest mean SSIM and outperformed every other architecture in all 10 folds,
though pairwise differences among the leading group did not survive correction
for multiple comparisons.*

**Realistic venues:** Physics in Medicine & Biology, Medical Physics, EJNMMI
Physics, Computers in Biology and Medicine — as proof-of-concept. Not a clinical
journal: n=10, simulated data, no external cohort.
