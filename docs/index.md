---
title: A Structure–Randomness Transfer Theorem for Sparse Data
---

# A Structure–Randomness Transfer Theorem for Sparse Data

**Author:** Kyle E. Litz, PhD

> **Abstract (short):** This site accompanies the preprint and provides a minimal,
> reproducible code+data bundle that validates the structure–randomness recovery
> pipeline under compressed non-uniform measurements.

## Key equations (rendered)
Inline: $\hat{x} = \arg\min_x\ \tfrac12\|Ax - y\|_2^2 + \lambda\|x\|_1$.

Block:
$$
\hat{g}_{\mathrm{str}}(t) = \sum_{k=1}^{K} \alpha_k D_k(t),\qquad
A = R\,\mathrm{diag}(\sqrt{\nu})\,D.
$$

## Reproduce the main figure
```bash
python3 -m venv srtt-env
source srtt-env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r env/requirements.txt
bash repro/run_repro.sh
```

<div style="margin: 1rem 0;">
  <a class="btn" href="../paper/preprint.pdf" download style="
     display:inline-block;padding:10px 16px;background:#0b5ed7;color:#fff;
     border-radius:6px;text-decoration:none;">Download PDF</a>
</div>

<details>
<summary>Inline viewer (optional)</summary>
<object data="../paper/preprint.pdf" type="application/pdf" width="100%" height="800px">
  <a href="../paper/preprint.pdf">Open PDF</a>
</object>
</details>
