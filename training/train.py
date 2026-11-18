"""Train Ridgeline's sky / land network on the synthetic views in training/data."""
import glob
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, 'ridgeline')
import skymodel as sm   # noqa: E402

DATA = 'training/data'
VIS_FACTOR = 1.3                                              # ln(50) / 3
WLOG = {(v, k): d for f in ('log.json', 'log_clouds.json', 'log_valley.json', 'log_summits.json') if os.path.exists(os.path.join(DATA, f))
        for v, k, d in json.load(open(os.path.join(DATA, f)))}


def load_split(split, n_near=2500, n_rand=2500, seed=0):
    rng = np.random.default_rng(seed)
    Xs, ys, groups = [], [], []
    views = sorted(glob.glob(f'{DATA}/{split}_*.npz'))
    for vi, vp in enumerate(views):
        z = np.load(vp)
        ground, dist, sky = z['ground'], z['dist'], z['skyline']
        H, W = ground.shape
        rows = np.arange(H)[:, None]
        vname = os.path.basename(vp)[:-4]
        for jp in sorted(glob.glob(vp[:-4] + '_w*.jpg')):
            vis = WLOG.get((vname, int(jp[-5])), {}).get('vis_km', 1e9)
            dmax = VIS_FACTOR * vis                           # beyond this land keeps < 2 % contrast
            valid = ~(ground & (dist < 0.15)) & ~(ground & (dist > dmax))
            gv = ground & (dist <= dmax)
            vsky = np.where(gv.any(0), gv.argmax(0), H)       # visible skyline
            near = valid & (np.abs(rows - vsky[None, :]) <= 30) & (vsky[None, :] < H)
            rgb = np.asarray(Image.open(jp).convert('RGB'), np.float32) / 255.0
            F = sm.pixel_features(rgb)
            idx = []
            for m, n in ((near, n_near), (valid, n_rand)):
                cand = np.flatnonzero(m.ravel())
                if cand.size:
                    idx.append(rng.choice(cand, min(n, cand.size), replace=False))
            idx = np.concatenate(idx)
            Xs.append(F.reshape(-1, F.shape[-1])[idx])
            ys.append((~ground).ravel()[idx].astype(np.int8))
            groups.append(np.full(idx.size, vi))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(groups)


def train(out_path='ridgeline/assets/sky_model.npz', hidden=(48, 24), seed=0):
    from sklearn.neural_network import MLPClassifier
    t0 = time.time()
    Xtr, ytr, _ = load_split('train', seed=seed)
    Xva, yva, _ = load_split('val', seed=seed + 1)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    clf = MLPClassifier(hidden_layer_sizes=hidden, activation='relu', alpha=1e-4, batch_size=4096,
                        learning_rate_init=2e-3, max_iter=90, early_stopping=True, validation_fraction=0.1,
                        n_iter_no_change=6, random_state=seed)
    clf.fit((Xtr - mu) / sd, ytr)
    acc_tr = clf.score((Xtr - mu) / sd, ytr)
    acc_va = clf.score((Xva - mu) / sd, yva)
    W = clf.coefs_
    B = clf.intercepts_
    np.savez(out_path, mu=mu.astype(np.float32), sd=sd.astype(np.float32),
             W1=W[0].astype(np.float32), b1=B[0].astype(np.float32), W2=W[1].astype(np.float32),
             b2=B[1].astype(np.float32), W3=W[2].astype(np.float32), b3=B[2].astype(np.float32),
             version=np.array('skynet-4 ' + time.strftime('%Y-%m-%d')), features=np.array(sm.FEATURE_NAMES))
    info = dict(n_train=int(len(ytr)), n_val=int(len(yva)), acc_train=float(acc_tr), acc_val=float(acc_va),
                iters=int(clf.n_iter_), secs=round(time.time() - t0, 1))
    json.dump(info, open('training/train_info.json', 'w'), indent=1)
    return info
