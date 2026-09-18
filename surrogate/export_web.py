"""
Export the POD surrogate and the verification runs for the browser demo (web/).

  web/model/pod.bin        float32 LE: mean (2·N²) then k modes (k · 2·N²)
  web/model/meta.json      k, N, the spline (x, y, M), the feature scaling, the training range,
                           and the list of exported solver runs with the POD error at each
  web/model/runs/Re<re>.bin   float32 LE (u, v) at the cell centres for each held-out / outside run

    .venv\\Scripts\\python surrogate\\export_web.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import data                       # noqa: E402
from pod import POD               # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "web", "model")


def main():
    m = POD.load(os.path.join(os.path.dirname(__file__), "models", "pod.npz"))
    man = data.manifest()
    os.makedirs(os.path.join(OUT, "runs"), exist_ok=True)

    blob = np.concatenate([m.mean, m.modes.ravel()]).astype("<f4")
    blob.tofile(os.path.join(OUT, "pod.bin"))

    runs = []
    for split in ("holdout", "extrap"):
        for re in man[split]:
            f, _ = data.load(re)
            f.astype("<f4").tofile(os.path.join(OUT, "runs", f"Re{int(re)}.bin"))
            err = data.rel_l2(m.predict([re])[0], f)
            runs.append(dict(re=int(re), split=split, err=err))
    runs.sort(key=lambda r: r["re"])

    lo, hi = np.log10(50), np.log10(1500)
    meta = dict(
        k=int(m.modes.shape[0]), N=int(m.shape[1]),
        feature=dict(centre=(lo + hi) / 2, half=(hi - lo) / 2),
        train_range=[min(man["train"]), max(man["train"])],
        n_train=len(man["train"]),
        spline=dict(x=m.spline.x.tolist(), y=m.spline.y.tolist(), M=m.spline.M.tolist()),
        runs=runs,
    )
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"pod.bin {blob.nbytes / 1e6:.2f} MB, {len(runs)} runs exported, k = {meta['k']}")


if __name__ == "__main__":
    main()
