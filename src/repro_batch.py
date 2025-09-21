#!/usr/bin/env python3
import argparse, os, json, base64, io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def soft(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)

def ista(D, x, lam, iters=200):
    # D: (N,K), x: (N,)
    # step 1/L with L ~ ||D||_2^2 via power iteration
    v = np.random.RandomState(0).randn(D.shape[1])
    for _ in range(10):
        v = D.T @ (D @ v)
        n = np.linalg.norm(v) + 1e-12
        v /= n
    L = float(np.linalg.norm(D @ v) + 1e-12)**2
    step = 1.0 / (L + 1e-9)
    z = np.zeros(D.shape[1])
    for _ in range(iters):
        grad = D.T @ (D @ z - x)
        z = soft(z - step * grad, lam * step)
    return z

def U2(x):
    X = np.fft.fft(x) / len(x)
    return float(np.sum(np.abs(X)**4)**0.25)

def spectral_entropy(x, eps=1e-12):
    X = np.fft.rfft(x)
    P = np.abs(X)**2
    s = float(P.sum()) + eps
    p = P/s + eps
    return float(-np.sum(p*np.log(p)))

def plot_series(x, rec, resid):
    fig = plt.figure(figsize=(10,4))
    plt.plot(x, label="observed", linewidth=1.0)
    plt.plot(rec, label="recovered", linewidth=1.0)
    plt.plot(resid, label="residual", linewidth=1.0)
    plt.legend(loc="upper right")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True, help="path to .npy observed vector")
    ap.add_argument("--dict", dest="dict_name", default="custom", choices=["custom","fourier"])
    ap.add_argument("--D", default=None, help="path to dictionary .npy if custom")
    ap.add_argument("--m", type=int, default=1400)   # placeholder
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--nu", default="phase")         # placeholder
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", required=True, help="output dir")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    x = np.load(args.x).astype(float).reshape(-1)
    N = x.shape[0]

    if args.dict_name == "custom":
        if args.D is None:
            raise SystemExit("custom dictionary selected but --D not provided")
        D = np.load(args.D).astype(float)
        if D.shape[0] != N:
            raise SystemExit(f"Dictionary rows {D.shape[0]} != len(x) {N}")
    else:
        # simple real Fourier dictionary
        K = max(64, N//4)
        t = np.arange(N)/N
        atoms = []
        for k in range(1, K//2 + 1):
            atoms.append(np.cos(2*np.pi*k*t))
            atoms.append(np.sin(2*np.pi*k*t))
        D = np.vstack(atoms).T
        D = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)

    xc = x - x.mean()

    z = ista(D, xc, lam=args.lam, iters=300)
    rec = D @ z
    resid = xc - rec

    metrics = {
        "dict": f"{args.dict_name}(K={D.shape[1]})",
        "m": args.m,
        "lambda": args.lam,
        "l2": 0,
        "U2": {"obs": U2(xc), "rec": U2(rec), "res": U2(resid)},
        "spectral_entropy": {"obs": spectral_entropy(xc), "rec": spectral_entropy(rec), "res": spectral_entropy(resid)},
        "snr_like": float((np.linalg.norm(rec)**2) / (np.linalg.norm(resid)**2 + 1e-12))
    }

    np.save(os.path.join(args.out, "recovered.npy"), rec)
    np.save(os.path.join(args.out, "residual.npy"), resid)
    np.savetxt(os.path.join(args.out, "coefficients.csv"), z[None, :], delimiter=",")
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    img64 = plot_series(xc, rec, resid)
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>SRTT Report</title></head><body>"
        "<h2>SRTT Headless Report</h2>"
        f"<p><b>Dictionary:</b> {metrics['dict']} &nbsp; <b>lambda:</b> {args.lam}</p>"
        f"<img src='data:image/png;base64,{img64}' style='max-width:100%;'/>"
        f"<pre>{json.dumps(metrics, indent=2)}</pre>"
        "</body></html>"
    )
    with open(os.path.join(args.out, "srtt_report.html"), "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    main()
