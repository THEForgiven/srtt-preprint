#!/usr/bin/env python3
import argparse, os, json, io, base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def soft(x, t): return np.sign(x)*np.maximum(np.abs(x)-t, 0.0)

def ista(D, x, lam, iters=250):
    # D:(N,K), x:(N,)
    # Lipschitz ≈ ||D||_2^2 via power iteration
    rs = np.random.RandomState(0)
    v = rs.randn(D.shape[1])
    for _ in range(12):
        v = D.T @ (D @ v); v /= (np.linalg.norm(v)+1e-12)
    L = float(np.linalg.norm(D @ v) + 1e-12)**2
    step = 1.0/(L+1e-9)
    z = np.zeros(D.shape[1])
    for _ in range(iters):
        z = soft(z - step * (D.T @ (D@z - x)), lam*step)
    return z

def gowers_u2(a):
    a = np.asarray(a, float)
    n = len(a)
    if n < 2:
        return 0.0
    s = 0.0
    for h in range(1, min(64, n-1)):  # light probe
        v = a[:n-h]*a[h:]
        s += np.mean(v)**2
    return float(np.sqrt(s/(min(64, n-1)+1e-12)))

def gowers_u3(a):
    # cheap proxy: U3 >= U2^1.5 on many structured cases; keep it light
    return float(gowers_u2(a)**1.5)

def spectral_entropy(a):
    a = np.asarray(a, float)
    A = np.fft.rfft(a - a.mean())
    p = np.abs(A)**2
    p = p/(p.sum()+1e-12)
    H = -(p * np.log(p + 1e-12)).sum()
    return float(H)

def fig_to_png_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def save_png(path, y, title):
    fig = plt.figure(figsize=(9,2.2), dpi=120)
    plt.plot(y, linewidth=1); plt.title(title); plt.xlabel("t"); plt.ylabel("amp")
    plt.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True, help="path to .npy observed vector")
    ap.add_argument("--dict", choices=["custom","fourier"], default="custom")
    ap.add_argument("--D", help="path to dictionary .npy (if custom)")
    ap.add_argument("--m", type=int, default=None)      # optional window for consistency
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--nu", default="phase")            # not used here; kept for API compat
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", required=True, help="output dir")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    x = np.load(args.x).astype(float).ravel()
    N = len(x)

    if args.dict == "custom":
        if not args.D: raise SystemExit("custom dict selected but --D not provided")
        D = np.load(args.D).astype(float)
        if D.shape[0] != N: raise SystemExit(f"D has {D.shape[0]} rows, expected {N}")
    else:
        # real Fourier atoms (cos/sin pairs), K≈min(256, N-1)*2
        K = min(256, max(2, N//8))
        t = np.arange(N)/N
        atoms = []
        for k in range(1, K//2+1):
            atoms.append(np.cos(2*np.pi*k*t))
            atoms.append(np.sin(2*np.pi*k*t))
        D = np.vstack(atoms).T
        D = D/(np.linalg.norm(D, axis=0, keepdims=True)+1e-12)

    # Recover via L1 (ISTA)
    z = ista(D, x, args.lam)
    xhat = D @ z
    r = x - xhat

    # Metrics (observed / recovered / residual)
    metrics = {
        "U2": {"obs": gowers_u2(x), "rec": gowers_u2(xhat), "res": gowers_u2(r)},
        "spectral_entropy": {"obs": spectral_entropy(x), "rec": spectral_entropy(xhat), "res": spectral_entropy(r)},
        "snr_like": float(np.var(xhat)/(np.var(r)+1e-12)),
        "dict": f"{args.dict}" + (f"({os.path.basename(args.D)})" if args.dict=="custom" else ""),
        "m": args.m if args.m else N, "lambda": args.lam, "l2": 0
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save arrays + coefficients
    np.save(os.path.join(args.out, "recovered.npy"), xhat)
    np.save(os.path.join(args.out, "residual.npy"), r)
    np.savetxt(os.path.join(args.out, "coefficients.csv"), z[None,:], delimiter=",")

    # Three PNGs
    save_png(os.path.join(args.out,"observed.png"),  x,    "Observed")
    save_png(os.path.join(args.out,"recovered.png"), xhat, "Recovered")
    save_png(os.path.join(args.out,"residual.png"),  r,    "Residual")

    # Inline-only 3-plot HTML
    with open(os.path.join(args.out, "observed.png"), "rb") as f:
        obs_b64 = base64.b64encode(f.read()).decode("ascii")
    with open(os.path.join(args.out, "recovered.png"), "rb") as f:
        rec_b64 = base64.b64encode(f.read()).decode("ascii")
    with open(os.path.join(args.out, "residual.png"), "rb") as f:
        res_b64 = base64.b64encode(f.read()).decode("ascii")

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SRTT Report</title>
<style>body{{font-family:system-ui,Arial,sans-serif;margin:20px}}
h1{{margin:0 0 10px}} .row{{display:flex;gap:12px;flex-wrap:wrap}}
.card{{flex:1 1 320px;border:1px solid #ddd;border-radius:10px;padding:10px}}
.card img{{width:100%;height:auto;border-radius:6px}}</style></head>
<body>
<h1>SRTT Report (3-plot)</h1>
<div class="row">
  <div class="card"><h3>Observed</h3><img src="data:image/png;base64,{obs_b64}"></div>
  <div class="card"><h3>Recovered</h3><img src="data:image/png;base64,{rec_b64}"></div>
  <div class="card"><h3>Residual</h3><img src="data:image/png;base64,{res_b64}"></div>
</div>
<h3>Key Metrics</h3>
<pre>{json.dumps(metrics, indent=2)}</pre>
</body></html>"""
    with open(os.path.join(args.out, "srtt_report.html"), "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    main()
