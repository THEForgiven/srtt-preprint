#!/usr/bin/env python3
"""
SRTT headless runner: sparse recovery + 3-plot HTML report (Observed/Recovered/Residual) + metrics.

Inputs:
  --x <path.npy>             observed 1D vector
  --dict {custom,fourier}    dictionary type
  --D <path.npy>             required if --dict custom (shape N x K with N==len(x))
  --m <int>                  book-keeping (e.g., target samples); not required by solver
  --lam <float>              L1 strength (lambda)
  --nu <str>                 placeholder for 'ν' mode (kept for CLI compatibility)
  --seed <int>               RNG seed for reproducibility
  --out <dir>                output directory

Outputs (in --out):
  recovered.npy, residual.npy, coefficients.csv, metrics.json,
  srtt_report.html  (3 plots only), observed.png / recovered.png / residual.png
"""
import argparse, os, io, json, base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Utils ----------
def soft(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)

def power_spectral_norm_squared(D, niter=10, seed=0):
    rng = np.random.RandomState(seed)
    v = rng.randn(D.shape[1])
    for _ in range(niter):
        v = D.T @ (D @ v)
        n = np.linalg.norm(v) + 1e-12
        v /= n
    L = float(np.linalg.norm(D @ v) + 1e-12) ** 2
    return L

def ista(D, x, lam, iters=300, seed=0):
    L = power_spectral_norm_squared(D, seed=seed)
    step = 1.0 / (L + 1e-12)
    z = np.zeros(D.shape[1])
    for _ in range(iters):
        grad = D.T @ (D @ z - x)
        z = soft(z - step * grad, lam * step)
    return z

def spectral_entropy(x):
    x = np.asarray(x)
    N = x.size
    # real-sided spectrum
    X = np.fft.rfft(x - x.mean())
    P = (np.abs(X) ** 2).astype(np.float64)
    s = P.sum() + 1e-12
    p = P / s
    H = -(p * (np.log(p + 1e-12))).sum()  # nats
    Hmax = np.log(p.size + 1e-12)
    return float(H / (Hmax + 1e-12))

def u2_proxy(x):
    """A light 'structure' proxy akin to low-order Gowers signal:
       squared normalized adjacent autocorrelation (unitless in [0,1)-ish)."""
    x = np.asarray(x, float)
    if x.size < 2:
        return 0.0
    num = np.mean(x[:-1] * x[1:])
    den = np.mean(x * x) + 1e-12
    val = (num / den) ** 2
    return float(max(0.0, val))

def snr_like(rec, res):
    num = float(np.dot(rec, rec))
    den = float(np.dot(res, res)) + 1e-12
    return num / den

def plot_b64(y, title):
    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(111)
    ax.plot(y)
    ax.set_title(title)
    ax.set_xlabel("sample")
    ax.set_ylabel("value")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def write_png(path, y, title):
    fig = plt.figure(figsize=(10,3))
    plt.plot(y)
    plt.title(title); plt.xlabel("sample"); plt.ylabel("value")
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)

def write_three_plot_report(outdir, obs, rec, res, metrics):
    # Save standalone PNGs
    write_png(os.path.join(outdir, "observed.png"),  obs, "Observed")
    write_png(os.path.join(outdir, "recovered.png"), rec, "Recovered")
    write_png(os.path.join(outdir, "residual.png"),  res, "Residual")

    # Embed
    b64_obs = plot_b64(obs, "Observed")
    b64_rec = plot_b64(rec, "Recovered")
    b64_res = plot_b64(res, "Residual")

    metrics_block = json.dumps(metrics, indent=2)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>SRTT Report — Observed / Recovered / Residual</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
 body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Helvetica,Arial,sans-serif;line-height:1.45;margin:24px;color:#111827}}
 h1,h2{{margin:.2em 0}}
 .card{{background:#fff;border:1px solid #e5e7eb;border-radius:12px;box-shadow:0 1px 2px rgba(0,0,0,.04);padding:16px;margin:16px 0}}
 img{{max-width:100%;height:auto;border-radius:8px;border:1px solid #e5e7eb}}
 .small{{font-size:.9em;color:#374151}}
 pre{{padding:12px;background:#f6f8fa;border:1px solid #e5e7eb;border-radius:8px;overflow-x:auto}}
</style>
</head>
<body>
  <h1>SRTT Report</h1>
  <p class="small">This report shows the Observed signal and the recovered structure with its residual, plus summary metrics.</p>

  <div class="card">
    <h2>Observed</h2>
    <img alt="Observed" src="data:image/png;base64,{b64_obs}"/>
  </div>

  <div class="card">
    <h2>Recovered</h2>
    <img alt="Recovered" src="data:image/png;base64,{b64_rec}"/>
  </div>

  <div class="card">
    <h2>Residual</h2>
    <img alt="Residual" src="data:image/png;base64,{b64_res}"/>
  </div>

  <div class="card">
    <h2>Key Metrics</h2>
    <pre>{metrics_block}</pre>
  </div>
</body>
</html>
"""
    with open(os.path.join(outdir, "srtt_report.html"), "w", encoding="utf-8") as f:
        f.write(html)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True, help="path to .npy observed vector")
    ap.add_argument("--dict", dest="dict_type", choices=["custom","fourier"], default="custom")
    ap.add_argument("--D", help="path to dictionary .npy if custom")
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--nu", type=str, default="phase")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output dir")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load observed
    x = np.load(args.x).astype(float).ravel()
    N = x.size

    # Build/load dictionary
    if args.dict_type == "custom":
        if not args.D:
            raise SystemExit("ERROR: --dict custom requires --D <path.npy>")
        D = np.load(args.D).astype(float)
        if D.shape[0] != N:
            raise SystemExit(f"ERROR: D.shape[0]={D.shape[0]} must equal len(x)={N}")
    else:
        # real Fourier dictionary (cos/sin pairs), K chosen moderately
        K = 256
        t = np.arange(N) / N
        atoms = []
        for k in range(1, K//2 + 1):
            atoms.append(np.cos(2*np.pi*k*t))
            atoms.append(np.sin(2*np.pi*k*t))
        D = np.vstack(atoms).T
        D = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)

    # Solve (ISTA)
    z = ista(D, x, args.lam, iters=300, seed=args.seed)
    rec = D @ z
    res = x - rec

    # Save core artifacts
    np.save(os.path.join(args.out, "recovered.npy"), rec)
    np.save(os.path.join(args.out, "residual.npy"),  res)
    # coefficients
    np.savetxt(os.path.join(args.out, "coefficients.csv"), z.reshape(1,-1), delimiter=",", fmt="%.6g")

    # Metrics
    metrics = {
        "dict": f"{args.dict_type}",
        "m": args.m,
        "lambda": args.lam,
        "l2": 0,
        "U2": {
            "obs": u2_proxy(x),
            "rec": u2_proxy(rec),
            "res": u2_proxy(res),
        },
        "spectral_entropy": {
            "obs": spectral_entropy(x),
            "rec": spectral_entropy(rec),
            "res": spectral_entropy(res),
        },
        "snr_like": snr_like(rec, res),
    }
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Write the 3-plot report (this is now the ONLY HTML produced)
    write_three_plot_report(args.out, x, rec, res, metrics)

if __name__ == "__main__":
    main()
