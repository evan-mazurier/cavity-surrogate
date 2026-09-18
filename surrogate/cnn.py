"""
Surrogate 2 — a small convolutional decoder: scalar log Re → dense → 4×4 feature map → five
upsampling stages → the (2, 128, 128) velocity field. ~500k parameters, trained with plain MSE on the
64 training runs, full batch, Adam. The held-out Re are never seen.

The POD model is a LINEAR subspace with a smooth 1-D interpolant; this one is nonlinear and free to
invent structure. That freedom is exactly what the held-out and extrapolation checks are for.

    .venv\\Scripts\\python surrogate\\cnn.py            # trains and saves surrogate/models/cnn.pt
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
import data                       # noqa: E402

MODELS = os.path.join(os.path.dirname(__file__), "models")
SEED = 0


class Decoder(nn.Module):
    def __init__(self, ch=(64, 64, 32, 32, 16), base=4, width=64):
        super().__init__()
        self.base, self.width = base, width
        self.fc = nn.Sequential(nn.Linear(1, 128), nn.GELU(), nn.Linear(128, 256), nn.GELU(),
                                nn.Linear(256, width * base * base), nn.GELU())
        blocks, c_in = [], width
        for c_out in ch:
            blocks += [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                       nn.Conv2d(c_in, c_out, 3, padding=1), nn.GELU(),
                       nn.Conv2d(c_out, c_out, 3, padding=1), nn.GELU()]
            c_in = c_out
        self.up = nn.Sequential(*blocks)
        self.head = nn.Conv2d(c_in, 2, 3, padding=1)

    def forward(self, s):                     # s: (n, 1) scaled log Re
        x = self.fc(s).view(-1, self.width, self.base, self.base)
        return self.head(self.up(x))


class CNN:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Decoder().to(self.device)

    def fit(self, re, fields, epochs=4000, lr=1e-3, log_every=500, holdout=None):
        torch.manual_seed(SEED)
        s = torch.tensor(data.feature(re), dtype=torch.float32, device=self.device)[:, None]
        y = torch.tensor(fields, dtype=torch.float32, device=self.device)
        self.scale = float(y.abs().max())
        y = y / self.scale
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        t0 = time.perf_counter()
        for ep in range(1, epochs + 1):
            self.net.train()
            opt.zero_grad()
            loss = ((self.net(s) - y) ** 2).mean()
            loss.backward()
            opt.step(); sched.step()
            if ep % log_every == 0 or ep == 1:
                msg = f"  epoch {ep:5d}  train mse {loss.item():.3e}"
                if holdout is not None:
                    errs = [data.rel_l2(p, t) for p, t in zip(self.predict(holdout[0]), holdout[1])]
                    msg += f"  holdout rel-L2 mean {np.mean(errs):.2e} max {np.max(errs):.2e}"
                print(msg + f"  ({time.perf_counter() - t0:.0f}s)", flush=True)
        return self

    @torch.no_grad()
    def predict(self, re):
        self.net.eval()
        s = torch.tensor(data.feature(re), dtype=torch.float32, device=self.device).reshape(-1, 1)
        return (self.net(s) * self.scale).cpu().numpy()

    def save(self, path):
        torch.save({"state": self.net.state_dict(), "scale": self.scale}, path)

    @classmethod
    def load(cls, path, device=None):
        m = cls(device)
        d = torch.load(path, map_location=m.device)
        m.net.load_state_dict(d["state"]); m.scale = d["scale"]
        return m


def main():
    re_tr, f_tr = data.load_split("train")
    re_ho, f_ho = data.load_split("holdout")
    m = CNN()
    n_par = sum(p.numel() for p in m.net.parameters())
    print(f"device {m.device}, {n_par:,} parameters, train {len(re_tr)} runs, holdout {len(re_ho)}")
    m.fit(re_tr, f_tr, holdout=(re_ho, f_ho))
    os.makedirs(MODELS, exist_ok=True)
    m.save(os.path.join(MODELS, "cnn.pt"))
    print("saved surrogate/models/cnn.pt")


if __name__ == "__main__":
    main()
