#!/usr/bin/env bash
set -euo pipefail
mkdir -p outputs
python src/repro_batch.py   --x data-min/interesting_stream.npy   --dict custom --D data-min/custom_D.npy   --m 1400 --lam 0.02 --nu phase --seed 12345   --out outputs
echo "Done. See outputs/srtt_report.html"
