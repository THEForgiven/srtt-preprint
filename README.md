# A Structure–Randomness Transfer Theorem for Sparse Data — Preprint & Repro

This repository hosts the **preprint** and a **minimal, reproducible code+data** bundle that validates the core functions of the theorem.

## Quick reproduce
```bash
python3 -m venv srtt-env
source srtt-env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r env/requirements.txt
bash repro/run_repro.sh
# -> outputs/ with recovered.npy, residual.npy, coefficients.csv, srtt_report.html
```

Expected outcomes (tolerances in `repro/params.json`):
- Residual **U²** ≪ Recovered **U²**
- Residual spectral entropy ≫ recovered spectral entropy
- SNR-like improves when using `custom_D.npy` vs Fourier

## Contents
- `paper/preprint.pdf` — The unreviewed preprint (frozen artifact).
- `docs/` — GitHub Pages landing with MathJax + **Download PDF** button.
- `src/` — Minimal code to reproduce (`repro_batch.py`) and (optionally) run the app code.
- `data-min/` — Minimal non-sensitive data to reproduce primary claims.
- `repro/` — One-command runner + default params.
- `env/` — Python requirements (pip/conda) and an optional Dockerfile.
- `.github/workflows/` — CI for reproducibility and Pages deployment.
- `CITATION.cff` — How to cite this work.
- `LICENSE` — Code license (MIT). See `paper/LICENSE` for preprint (CC BY 4.0).

## How to cite
See [`CITATION.cff`](CITATION.cff). A DOI will be minted upon first GitHub Release via Zenodo (see `RELEASE_CHECKLIST.md`).

## Security & data notes
- The sample datasets are small and non-sensitive. Do not add production logs directly to this repo.
- For larger/sensitive datasets, publish separately as **Zenodo datasets** and link from this README.

---

**Author:** Kyle E. Litz, PhD  
**Version:** v0.1-preprint (2025-09-20)
