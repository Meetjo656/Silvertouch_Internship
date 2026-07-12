# Task 1 — Perceptual Text-Color Semantic Comparison

> **Domain:** Computer Vision / Color Science
> **Type:** Standalone Python pipeline
> **Status:** Complete & runnable (single-shot CLI)
> **Folder:** `task1/`

---

## 1. Overview

Task 1 is a perceptual color-comparison tool that decides whether the **text color** of two cropped packaging images (a `reference` and a `target`) is an acceptable visual match. Rather than comparing raw RGB vectors — which are not perceptually uniform and drift under shadow — the pipeline converts colors to **CIE L\*a\*b\*** space and scores the difference with the professional **CIEDE2000 ΔE** metric. A `ΔE₀₀ ≤ 8.0` is treated as an acceptable human-perceived match.

It isolates the text foreground, finds the dominant text color via density-based clustering, maps that color to a named color family (the 949-color XKCD palette), and renders a side-by-side dashboard.

## 2. Problem Statement

Standard RGB math fails for print-vs-digital comparison: shadows push RGB vectors around wildly even when the perceived hue is unchanged. Colorimetry-grade ΔE in LAB space lets the system mimic the human eye — analyzing only the chromaticity (`a*`, `b*`) axes while treating `L*` (lightness) separately — so lighting variation does not invalidate the comparison.

## 3. Architecture & Modules

```
task1/
├── main.py                     # single entry point: orchestrates the full flow
├── config.py                   # paths + tunable thresholds
├── preprocessing/              # image load + background/foreground separation
├── clustering/                 # dominant-color discovery (MeanShift in LAB)
├── color_processing/           # color families, ΔE comparison, hex utils
├── utils/                      # result logging helpers
└── visualization/              # matplotlib comparison dashboard
```

| Module | Key file(s) | Responsibility |
|---|---|---|
| `preprocessing` | `load_image.py` | `load_image()` — BGR→RGB via OpenCV |
| | `estimate_background.py` | `estimate_background()` — median of border pixels |
| | `foreground_extraction.py` | `extract_foreground()` — foreground mask via CIE 1976 ΔE vs background; returns `lab`, `rgb`, `mask`, `delta_from_bg_values` |
| `clustering` | `meanshift_cluster.py` | `MeanShiftExtractor.extract_colors()` — density-based MeanShift in LAB, `estimate_bandwidth`, scores clusters by `count * mean_delta_e`, LAB→RGB via `skimage.color.lab2rgb` |
| `color_processing` | `dominant_color.py` | `get_dominant_color()` — returns highest-importance-cluster color |
| | `color_family.py` | `get_color_family()` — nearest XKCD name by vectorized CIEDE2000; `get_meta_family()` — keyword grouping into meta-families |
| | `rgb_hex.py` | `rgb_to_hex()` |
| | `deltae_compare.py` | `CompareColors` — `rgb_to_lab`, `delta_e`, `is_match` |
| `utils` | `helpers.py` | `print_comparison_results()` — aligned ASCII log |
| `visualization` | `dashboard.py` | `generate_comparison_dashboard()` — 3×4 matplotlib figure; `normalize_cluster_map()`; debug saves gated by `SAVE_DEBUG_IMAGES` |

## 4. Key Files & Entry Points

- **`main.py` → `main()`** is the only entry point. Flow:
  1. Load `data/reference.png` + `data/target.png` (paths from `config.py`: `REFERENCE_IMAGE`, `TARGET_IMAGE`, `OUTPUT_DASHBOARD`).
  2. `extract_foreground()` on each.
  3. `MeanShiftExtractor.extract_colors()`.
  4. `get_dominant_color()`.
  5. `get_color_family()` / `get_meta_family()`.
  6. `CompareColors.is_match()` (CIEDE2000, threshold from `config.py`).
  7. `print_comparison_results()` → map cluster labels spatially → `generate_comparison_dashboard()` → save `outputs/comparison_dashboard.png`.

## 5. Core Techniques

- **MeanShift clustering in LAB** — `sklearn.cluster.MeanShift` with `estimate_bandwidth(quantile=0.2)` from `config.MEANSHIFT_QUANTILE`; finds true color peaks without preset cluster counts.
- **CIE LAB conversion** — `colour.sRGB_to_XYZ` → `XYZ_to_Lab`.
- **CIEDE2000 ΔE** — `colour.delta_E(method='CIE 2000')` used in `get_color_family` and `CompareColors.is_match`; foreground masking uses the simpler **CIE 1976** (`config.FOREGROUND_DELTA_THRESHOLD = 10`).
- **Dominant color** — `get_dominant_color()` picks the cluster with the max `importance_score` = `pixel_count * mean_Δ_from_background`. Black `[0,0,0]` fallback if no clusters.
- **Color families** — nearest-neighbor over 949 matplotlib `XKCD_COLORS` (computed once at import), then meta-family keyword grouping.

## 6. Tech Stack / Dependencies

Inferred from imports: `opencv-python` (`cv2`), `numpy`, `scikit-learn` (`sklearn.cluster`), `scikit-image` (`skimage.color.lab2rgb`), **colour-science** (`colour`), `matplotlib`.

> ⚠️ **No `requirements.txt`, `setup.py`, or `pyproject.toml`** — dependencies are undocumented and must be reproduced manually. `colour` and `scikit-image` are non-obvious installs worth pinning.

## 7. Data & Artifacts

- Inputs: `data/reference.png`, `data/target.png`.
- Output: `outputs/comparison_dashboard.png` (already present from a prior run).

## 8. Maturity / Status

**Complete and runnable.** No TODOs, stubs, or `NotImplementedError`. Comment markers ("Refinement 2/3/4/8") show iterative polish. Robustness notes: empty foreground raises `ValueError` (wrapped in a top-level `try/except` in `main()`); `get_dominant_color` has a black fallback. The only inconsistency: foreground thresholding uses CIE 1976 while the match decision uses CIEDE2000 — intentional but worth documenting.

## 9. How to Run

From the `task1/` root, with `data/reference.png` and `data/target.png` present:

```bash
pip install opencv-python numpy scikit-learn scikit-image colour matplotlib
python main.py
```

No install command is documented in-repo; the snippet above reconstructs the dependency set.

---

*Report generated from source inspection of `task1/`. See parent `README.md` for the portfolio-level summary of this task.*
