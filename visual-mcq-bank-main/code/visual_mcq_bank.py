"""Visual MCQ question bank (v102): every X-ray of the 4 original datasets becomes one question.

Setting: strict LODO with the 4-domain system of 29 Oct (Dr Batra's data is not used anywhere).
The networks for dataset n were trained once, on the other 3 datasets only, and are NOT retrained
here; so any image of dataset n can be a query without leakage.

Query sets. Per dataset, the original test split is set "test". The original train split is cut into
5 class-stratified chunks "train1".."train5", grouped by patient (OrthoNet hip/knee, physionet) so one
patient's films never sit on both sides. Pacemakers has no patient id (its IMP prefix is a batch), so
its chunks are stratified only. For each query set the album is 5 random images per class drawn from
the rest of that dataset (for "test": from the train split, the same draw 0 as the reported numbers;
for "train k": from test + the other 4 chunks). A class whose every image is in the query set has no
album images, so its queries can only be answered by NOTA.

Each question: the query, the system's top-5 classes (4 on knee, which has 4 classes), shown as the
album image of each class nearest to the query in the system's projected space, shuffled with a fixed
seed, then "None of the above" fixed as option 6. Correct = the slot of the true class, or 6 if the
true class is not among the five (natural rate; nothing is forced).

System (val-chosen arms of the 29 Oct strict-LODO table, mean top-1 86.60):
  A  = Swin + ConvNeXt s1-s4 + frozen DINOv2, projection fitted on the album + 8 augmented views each
  hip and physionet: A + 5th ConvNeXt seed (plain blocks); knee and pacemakers: A
Byte-identical copies: a file whose exact bytes occur more than once keeps one copy (the test copy if
there is one, so no train chunk can find its own copy in an album; else the one in the reported test
album); a copy with two different labels
is dropped entirely. The test album stays the reported draw 0 from the full train split (asserted
free of dropped copies). Dropped rows go to dropped.csv with the reason.
Test images' augmented views come from the query-view cache (same transform, different seed).
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.fewshot_gallery import project
from experiments.semi_aug import scores
from experiments.support_aug_ensemble import ARMS
from experiments.taps_retrieval import nrm
from experiments.uniform5_tables import draw_idx

FOLDS = ["ortho_hip", "ortho_knee", "physionet", "pacemakers"]
EXTRA = {"ortho_hip": "cnx_hip_s5.npz", "physionet": "cnx_physionet_s5.npz"}
GROUPED = {"ortho_hip", "ortho_knee", "physionet"}


def unit(x):
    x = x.astype(np.float64)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def features(aug_dir, extra_dir, fold, members, n_tr, n_te):
    """All images of the fold in index order [train rows; test rows]: plain X (N, D), views V (N, 8, D)."""
    Xb, Vb = [], []
    for m, taps in members:
        z = np.load(f"{aug_dir}/aug_{m}_{fold}.npz")
        zq = np.load(f"{aug_dir}/aug_q_{m}_{fold}.npz")
        assert np.array_equal(z["q_label"], zq["q_label"])
        for t in taps:
            Xb.append(nrm(np.vstack([z[f"g_{t}"], z[f"q_{t}"]]).astype(np.float64)))
            Vb.append(unit(np.concatenate([z[f"a_{t}"], zq[f"qa_{t}"]])))
        lab = np.concatenate([z["g_label"], z["q_label"]]).astype(str)
    if fold in EXTRA:   # plain blocks only; the views repeat the plain vector (as semi_aug.add_extra)
        z = np.load(f"{extra_dir}/{EXTRA[fold]}", allow_pickle=True)
        assert np.array_equal(np.concatenate([z["g_label"], z["q_label"]]).astype(str), lab)
        for t in ("bb", "s2"):
            x = nrm(np.vstack([z[f"g_{t}"], z[f"q_{t}"]]).astype(np.float64))
            Xb.append(x)
            Vb.append(np.repeat(x[:, None], Vb[0].shape[1], 1))
    X, V = np.hstack(Xb), np.concatenate(Vb, -1)
    assert len(X) == n_tr + n_te
    return X, V, lab


def chunks(tr, fold, k, seed):
    """Chunk id 1..k for every train row: class-stratified, patient-grouped where patients exist."""
    out = np.zeros(len(tr), int)
    if fold in GROUPED:
        sp = StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(tr, tr.model, tr.pat)
    else:
        sp = StratifiedKFold(k, shuffle=True, random_state=seed).split(tr, tr.model)
    for c, (_, te) in enumerate(sp, 1):
        out[te] = c
    return out


def dedupe(rows, prefer=()):
    """Row positions to drop, with reasons (see module doc). Among same-label copies the test copy is
    kept, else one in `prefer` (the reported test album), else the first."""
    md5 = rows.path.map(lambda p: hashlib.md5(open(p, "rb").read()).hexdigest())
    drop = {}
    for h, g in rows.groupby(md5):
        if len(g) == 1:
            continue
        if g.model.nunique() > 1:
            assert (g.split == "train").all(), "label conflict inside the test split"
            for i in g.index:
                drop[i] = "same file with different labels: " + " / ".join(sorted(g.model.unique()))
            continue
        te = g[g.split == "test"]
        pref = [i for i in g.index if i in set(prefer)]
        keep = te.index[0] if len(te) else (pref[0] if pref else g.index[0])
        for i in g.index:
            if i != keep:
                drop[i] = f"byte copy of {rows.path[keep]}"
    return drop


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, help="4-domain lodo_index.csv")
    ap.add_argument("--aug-dir", required=True, help="v96/support_aug_v2")
    ap.add_argument("--extra-dir", required=True, help="dir with the 5th ConvNeXt seed caches (v99/netst)")
    ap.add_argument("--chunks", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="chunking, album draw and option-order seed")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.index)
    members = ARMS["+cnx_s2s3s4+dinov2"]
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    allq, albums, summary, dropped = [], [], [], []
    for fold in FOLDS:
        tr = df[(df.domain == fold) & (df.split == "train")].reset_index(drop=True)
        te = df[(df.domain == fold) & (df.split == "test")].reset_index(drop=True)
        rows = pd.concat([tr, te], ignore_index=True)
        X, V, lab = features(args.aug_dir, args.extra_dir, fold, members, len(tr), len(te))
        assert np.array_equal(lab, rows.model.to_numpy().astype(str))
        drop = dedupe(rows, draw_idx(tr, 5, args.seed))
        for i, why in drop.items():
            dropped.append({"dataset": fold, "image": rows.path[i], "model": lab[i], "split": rows.split[i],
                            "reason": why})
        ktr = np.array([i for i in range(len(tr)) if i not in drop])
        rows["set"] = "dropped"
        rows.loc[ktr, "set"] = [f"train{c}" for c in chunks(tr.iloc[ktr].reset_index(drop=True), fold,
                                                            args.chunks, args.seed)]
        kte = np.array([i for i in range(len(tr), len(rows)) if i not in drop])
        rows.loc[kte, "set"] = "test"
        if fold in GROUPED:   # no patient on both sides of any query set
            ps = rows[rows.set != "dropped"].groupby("pat")["set"].nunique()
            assert ps.max() == 1, f"{fold}: patient split across sets"
        for qs in ["test"] + [f"train{c}" for c in range(1, args.chunks + 1)]:
            qpos = np.flatnonzero(rows.set == qs)
            pool = np.flatnonzero((rows.set != qs) & (rows.set != "dropped"))
            if qs == "test":   # album = draw 0 from the train split, exactly as the reported numbers
                pool = np.arange(len(tr))
            sel = pool[draw_idx(rows.iloc[pool], 5, args.seed)]
            assert not set(sel) & set(drop), f"{fold} {qs}: album holds a dropped copy"
            alab = lab[sel]
            M, cl = scores(X, V, sel, alab, X[qpos], True)
            nv = V.shape[1]
            Xf = np.vstack([X[sel], V[sel].reshape(-1, V.shape[-1])])
            _, Z, _ = project(Xf, np.concatenate([alab, np.repeat(alab, nv)]), np.vstack([X[sel], X[qpos]]), 0.1, 256)
            S = Z[len(sel):] @ Z[:len(sel)].T
            for j in sel:
                albums.append({"dataset": fold, "query_set": qs, "image": rows.path[j], "model": lab[j]})
            n_nota = 0
            for i, p in enumerate(qpos):
                top = np.argsort(-M[i])[:min(5, len(cl))]
                opts = []
                for r, c in enumerate(cl[top]):
                    cand = np.flatnonzero(alab == c)
                    j = cand[np.argmax(S[i, cand])]
                    opts.append({"model": str(c), "image": rows.path[sel[j]], "rank": r + 1,
                                 "score": float(M[i, top[r]]), "sim": float(S[i, j])})
                perm = rng.permutation(len(opts))
                shown = [opts[r] for r in perm]
                gt = lab[p]
                slot = next((k + 1 for k, o in enumerate(shown) if o["model"] == gt), 6)
                n_nota += slot == 6
                allq.append({"dataset": fold, "query_set": qs, "query_image": rows.path[p], "query_model": gt,
                             "query_patient": str(rows.pat[p]) if fold in GROUPED else "",
                             "options": shown, "correct_option": int(slot),
                             "gt_in_album": bool(gt in set(alab)), "retrieval_top1": str(cl[top[0]])})
            nq = len(qpos)
            top1 = np.mean([q["retrieval_top1"] == q["query_model"] for q in allq[-nq:]])
            summary.append({"dataset": fold, "query_set": qs, "questions": nq, "classes_in_album": len(cl),
                            "classes_total": rows.model.nunique(), "album_images": len(sel),
                            "nota_correct": int(n_nota), "retrieval_top1": round(100 * top1, 2),
                            "retrieval_top5": round(100 * (1 - n_nota / nq), 2)})
            print(json.dumps(summary[-1]), flush=True)
    for n, q in enumerate(allq, 1):
        q["qid"] = f"Q{n:04d}"
    json.dump({"seed": args.seed, "chunks": args.chunks, "n": len(allq), "summary": summary,
               "questions": allq}, open(f"{args.out}/bank_raw.json", "w"), indent=1)
    pd.DataFrame(albums).to_csv(f"{args.out}/albums_raw.csv", index=False)
    pd.DataFrame(dropped).to_csv(f"{args.out}/dropped.csv", index=False)
    pd.DataFrame(summary).to_csv(f"{args.out}/summary.csv", index=False)
    print(f"{len(allq)} questions, NOTA correct in {sum(q['correct_option'] == 6 for q in allq)}")


if __name__ == "__main__":
    main()
