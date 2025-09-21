# A Structure–Randomness Transfer Theorem for Sparse Data — Preprint & Repro
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://THEForgiven.github.io/srtt-preprint/)
![Python 3.10](https://img.shields.io/badge/python-3.10-3776AB?logo=python&logoColor=white)
[![License](https://img.shields.io/github/license/THEForgiven/srtt-preprint)](./LICENSE)
[![Reproduce CI](https://github.com/THEForgiven/srtt-preprint/actions/workflows/reproduce.yml/badge.svg?branch=main)](https://github.com/THEForgiven/srtt-preprint/actions/workflows/reproduce.yml)


This repository hosts the **preprint** and a **minimal, reproducible code+data** bundle that validates the core functions of the theorem.
The structure-randomness transfer theorem unifies concepts from higher-order Fourier analysis, relative Szemerédi-type transference, and compressed sensing. The core idea of the theorem is that a single "pseudorandom majorant" can be used for two purposes: to count structured patterns within sparse data and to stably recover structured signals from a small number of random measurements.
The theorem relies on two main components:
·	A pseudorandom majorant (v), which is a weighting function that acts like a uniform measure on low-complexity linear patterns.
·	An arithmetic regularity decomposition, which splits the data's indicator function (1A​) into a structured component (gstr​) and a uniform, or pseudorandom, component (gunf​).
The theorem has two key parts:
·	Part (i): Polynomial Pattern Counts — It guarantees that the number of polynomial configurations of a certain degree within a set A matches what you would expect in a random model, with a small error. This extends the transference paradigm, which shows that theorems about dense sets can be applied to sparse, pseudorandom ones.
·	Part (ii): Stable Structured Recovery — It demonstrates that the same pseudorandom majorant can generate a small number of random measurements that approximately satisfy the Restricted Isometry Property (RIP). This property allows for the stable recovery of a structured signal (one that lives in the span of the structured component) using a convex program.


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

---

## Build status
[![reproduce](https://github.com/THEForgiven/srtt-preprint/actions/workflows/ci.yml/badge.svg)](https://github.com/THEForgiven/srtt-preprint/actions/workflows/ci.yml)
[![github-pages](https://github.com/THEForgiven/srtt-preprint/actions/workflows/pages.yml/badge.svg)](https://github.com/THEForgiven/srtt-preprint/actions/workflows/pages.yml)

**Site:** https://THEForgiven.github.io/srtt-preprint/  
**Repo:** https://github.com/THEForgiven/srtt-preprint
