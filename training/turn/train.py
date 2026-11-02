"""Trains a head on Gemma hidden states: the end-of-turn head, or with
--label addressed the addressee head.

Features come from gemmalive-turn-features: per clip, the output of every
layer at the token that closes the committed audio. Labels come from the TSVs
next to them: Smart Turn's (prepare.py, make_pauses.py) or the addressee
data's (training/addressee).

    python train.py --features out/ --sweep          # linear probe per layer
    python train.py --features out/ --layers 16,20   # train and evaluate a head
    python train.py --features addressee/ --layers 8,14,17,21 --hidden 256 \
        --label addressed --test-prefix val --group-by kind
"""

import argparse
import collections
import glob
import json
import os
import struct

import numpy as np
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load_split(features_dir, prefix):
    """Memory-maps every <prefix>-*.f16 with its .tsv. Returns [(array, rows)]."""
    parts = []
    for path in sorted(glob.glob(os.path.join(features_dir, f"{prefix}-*.f16"))):
        with open(path, "rb") as f:
            n_layer, n_embd = struct.unpack("<ii", f.read(8))
        x = np.memmap(path, dtype=np.float16, mode="r", offset=8)
        x = x.reshape(-1, n_layer, n_embd)
        with open(path[:-4] + ".tsv") as f:
            header = f.readline().rstrip("\n").split("\t")
            rows = [dict(zip(header, line.rstrip("\n").split("\t"))) for line in f]
        if len(rows) != len(x):
            raise SystemExit(f"{path}: {len(x)} feature rows but {len(rows)} labels (incomplete?)")
        parts.append((x, rows))
    if not parts:
        raise SystemExit(f"no {prefix}-*.f16 in {features_dir}")
    return parts


LABEL = "endpoint_bool"


def gather(parts, layers, limit=None, seed=0):
    """Stacks the selected layers of all parts into float32 [N, len(layers), E]."""
    xs, rows = [], []
    for x, r in parts:
        xs.append(np.asarray(x[:, layers, :], dtype=np.float32))
        rows.extend(r)
    x = np.concatenate(xs)
    if limit and limit < len(x):
        idx = np.random.default_rng(seed).choice(len(x), limit, replace=False)
        idx.sort()
        x = x[idx]
        rows = [rows[i] for i in idx]
    y = np.array([r[LABEL] == "True" for r in rows], dtype=np.float32)
    return x, y, rows


class Head(nn.Module):
    """Learned mix of the selected layers, normalised, then linear or MLP."""

    def __init__(self, n_layers, n_embd, hidden=0):
        super().__init__()
        self.mix = nn.Parameter(torch.zeros(n_layers))
        self.register_buffer("mean", torch.zeros(n_embd))
        self.register_buffer("std", torch.ones(n_embd))
        if hidden:
            self.net = nn.Sequential(nn.Linear(n_embd, hidden), nn.GELU(), nn.Linear(hidden, 1))
        else:
            self.net = nn.Linear(n_embd, 1)

    def mixed(self, x):
        w = torch.softmax(self.mix, dim=0)
        return torch.einsum("l,nle->ne", w, x)

    def forward(self, x):
        h = (self.mixed(x) - self.mean) / self.std
        return self.net(h).squeeze(-1)


def fit(x, y, x_val, y_val, hidden=0, epochs=12, lr=1e-3, wd=1e-4, batch=2048, device=DEVICE, quiet=False):
    torch.manual_seed(0)
    xt = torch.from_numpy(x).to(device)
    yt = torch.from_numpy(y).to(device)
    xv = torch.from_numpy(x_val).to(device)
    yv = torch.from_numpy(y_val).to(device)

    head = Head(x.shape[1], x.shape[2], hidden).to(device)
    with torch.no_grad():
        m = head.mixed(xt)
        head.mean.copy_(m.mean(0))
        head.std.copy_(m.std(0) + 1e-4)

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(x) + batch - 1) // batch))
    loss_fn = nn.BCEWithLogitsLoss()
    best = (-1.0, None)
    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(len(xt), device=device)
        for i in range(0, len(xt), batch):
            idx = perm[i : i + batch]
            loss = loss_fn(head(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        acc = accuracy(head, xv, yv)
        if acc > best[0]:
            best = (acc, {k: v.detach().clone() for k, v in head.state_dict().items()})
        if not quiet:
            print(f"  epoch {epoch + 1:2d}  loss {loss.item():.4f}  val acc {acc:.4f}")
    head.load_state_dict(best[1])
    return head, best[0]


@torch.no_grad()
def predict(head, x, device=DEVICE, batch=8192):
    head.eval()
    out = []
    for i in range(0, len(x), batch):
        xb = x[i : i + batch] if torch.is_tensor(x) else torch.from_numpy(x[i : i + batch]).to(device)
        out.append(torch.sigmoid(head(xb)).float().cpu())
    return torch.cat(out).numpy()


def accuracy(head, x, y):
    p = predict(head, x)
    yy = y.cpu().numpy() if torch.is_tensor(y) else y
    return float(((p > 0.5) == (yy > 0.5)).mean())


GROUPS = ("language", "dataset")


def report(p, y, rows, threshold=0.5):
    pred = p > threshold
    truth = y > 0.5
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    res = {
        "n": len(y),
        "accuracy": float((pred == truth).mean()),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
    }
    res["f1"] = 2 * res["precision"] * res["recall"] / max(res["precision"] + res["recall"], 1e-9)
    # As in the Smart Turn benchmark: share of all samples.
    res["fpr"] = fp / len(y)
    res["fnr"] = fn / len(y)
    for field in GROUPS:
        key = field + "s"
        if not any(field in r for r in rows):
            continue
        groups = collections.defaultdict(list)
        for i, r in enumerate(rows):
            groups[r.get(field, "-")].append(i)
        res[key] = {
            name: float((pred[idx] == truth[idx]).mean())
            for name, idx in sorted(groups.items(), key=lambda kv: -len(kv[1]))
        }
    if any("synthetic" in r for r in rows):
        synth = np.array([r.get("synthetic", "True") == "True" for r in rows])
        res["synthetic_accuracy"] = float((pred[synth] == truth[synth]).mean()) if synth.any() else None
        res["human_accuracy"] = float((pred[~synth] == truth[~synth]).mean()) if (~synth).any() else None
    return res


def split_val(n, frac=0.05, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    k = int(n * frac)
    return idx[k:], idx[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--sweep", action="store_true", help="linear probe on each layer")
    ap.add_argument("--sweep-limit", type=int, default=60000, help="train clips used by the sweep")
    ap.add_argument("--layers", default="", help="comma-separated layers to mix")
    ap.add_argument("--hidden", type=int, default=0, help="MLP hidden size, 0 = linear")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--out", default="", help="write the head (.pt) and test report (.json)")
    ap.add_argument("--label", default="endpoint_bool", help="label column, True/False")
    ap.add_argument("--test-prefix", default="test", help="feature files of the test split")
    ap.add_argument("--group-by", default="language,dataset", help="report accuracy per value of these columns")
    ap.add_argument("--fn-cost", type=float, default=1.0,
                    help="cost of a false negative relative to a false positive, for picking the threshold")
    args = ap.parse_args()
    global LABEL, GROUPS
    LABEL = args.label
    GROUPS = tuple(g for g in args.group_by.split(",") if g)

    train_parts = load_split(args.features, "train")
    test_parts = load_split(args.features, args.test_prefix)
    n_layer = train_parts[0][0].shape[1]

    if args.sweep:
        print(f"layer sweep, linear probe, {args.sweep_limit} train clips")
        results = []
        for layer in range(n_layer):
            x, y, _ = gather(train_parts, [layer], limit=args.sweep_limit)
            tr, va = split_val(len(x))
            _, acc = fit(x[tr], y[tr], x[va], y[va], epochs=6, quiet=True)
            results.append((layer, acc))
            print(f"layer {layer:2d}  val acc {acc:.4f}", flush=True)
        best = sorted(results, key=lambda r: -r[1])[:5]
        print("best:", ", ".join(f"{l} ({a:.4f})" for l, a in best))
        return

    layers = [int(v) for v in args.layers.split(",")]
    x, y, _ = gather(train_parts, layers)
    tr, va = split_val(len(x))
    print(f"train {len(tr)}  val {len(va)}  layers {layers}  hidden {args.hidden}")
    head, val_acc = fit(x[tr], y[tr], x[va], y[va], hidden=args.hidden, epochs=args.epochs)
    del x

    xt, yt, rows = gather(test_parts, layers)
    p_test = predict(head, xt)
    res = report(p_test, yt, rows)
    # How the error types trade off: a higher threshold cuts the user off less
    # often (fpr) but waits for the pause limit more often (fnr).
    res["thresholds"] = {}
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        r = report(p_test, yt, rows, threshold=t)
        res["thresholds"][t] = {k: r[k] for k in ("accuracy", "fpr", "fnr")}
        print(f"threshold {t:.1f}  acc {r['accuracy']:.4f}  fpr {r['fpr']:.4f}  fnr {r['fnr']:.4f}")
    # Threshold with the lowest weighted error on the test split.
    grid = [round(0.05 * i, 2) for i in range(1, 20)]
    costs = {t: (lambda r: r["fpr"] + args.fn_cost * r["fnr"])(report(p_test, yt, rows, threshold=t)) for t in grid}
    res["chosen_threshold"] = min(costs, key=costs.get)
    print(f"chosen threshold {res['chosen_threshold']} (false negatives cost {args.fn_cost}x)")
    res.update({"layers": layers, "hidden": args.hidden, "val_accuracy": val_acc,
                "mix": torch.softmax(head.mix, 0).tolist()})
    groups = [g + "s" for g in GROUPS if g + "s" in res]
    print(json.dumps({k: v for k, v in res.items() if k not in groups}, indent=2))
    for g in groups:
        print(f"per {g[:-1]}:", ", ".join(f"{k} {v:.3f}" for k, v in res[g].items()))

    if args.out:
        torch.save({"state": head.state_dict(), "layers": layers, "hidden": args.hidden}, args.out + ".pt")
        with open(args.out + ".json", "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
