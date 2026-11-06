"""Export the v102 visual-MCQ bank as a self-contained folder for a GitHub repo.

Layout written to --out:
  images/<dataset>/<id>.<ext>   every image used (query, option or album), original bytes, neutral names
  questions.csv                 one row per question: query, 5 options (+ "None of the above" as 6),
                                model names, correct option
  questions.jsonl               the same with every field (system rank/score of each option)
  albums.csv                    the 5-per-class album behind each (dataset, query set)
  summary.csv                   counts and retrieval accuracy per (dataset, query set)
  dropped.csv                   duplicate files removed before building, with the reason
  image_index.csv               repo image path -> model, original split, query set, source path
  preview/                      browsable pages: every question with its images and model names
"""
import argparse
import json
import os
import shutil

import pandas as pd

NOTA = "None of the above"
NAMES = {"ortho_hip": "OrthoNet hip", "ortho_knee": "OrthoNet knee", "physionet": "Physionet (cardiac devices)",
         "pacemakers": "Pacemakers (Howard et al.)"}
PER_PAGE = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-dir", required=True)
    ap.add_argument("--index", required=True, help="the 4-domain lodo_index.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    B = json.load(open(f"{args.bank_dir}/bank_raw.json"))
    Q = B["questions"]
    albums = pd.read_csv(f"{args.bank_dir}/albums_raw.csv")
    df = pd.read_csv(args.index)
    meta = df.set_index("path")

    # neutral names, numbered per dataset in index order
    used = set(q["query_image"] for q in Q) | {o["image"] for q in Q for o in q["options"]} | set(albums.image)
    rel, rows = {}, []
    for ds in NAMES:
        d = df[(df.domain == ds) & df.path.isin(used)]
        for n, p in enumerate(d.path, 1):
            ext = os.path.splitext(p)[1].lower()
            rel[p] = f"images/{ds}/{ds}_{n:04d}{ext}"
    qset = {q["query_image"]: q["query_set"] for q in Q}
    for p, r in rel.items():
        dst = os.path.join(args.out, r)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            shutil.copyfile(p, dst)
        rows.append({"image": r, "model": meta.model[p], "dataset": meta.domain[p], "original_split": meta.split[p],
                     "query_set": qset.get(p, ""), "source_path": p})
    pd.DataFrame(rows).to_csv(f"{args.out}/image_index.csv", index=False)

    flat, jl = [], open(f"{args.out}/questions.jsonl", "w")
    for q in Q:
        c = q["correct_option"]
        r = {"qid": q["qid"], "dataset": q["dataset"], "query_set": q["query_set"],
             "query_image": rel[q["query_image"]], "query_model": q["query_model"]}
        for k in range(5):
            o = q["options"][k] if k < len(q["options"]) else None
            r[f"option_{k + 1}_image"] = rel[o["image"]] if o else ""
            r[f"option_{k + 1}_model"] = o["model"] if o else ""
        r["option_6"] = NOTA
        r["correct_option"] = c
        r["correct_answer"] = NOTA if c == 6 else q["query_model"]
        r["retrieval_top1_model"] = q["retrieval_top1"]
        r["n_image_options"] = len(q["options"])
        flat.append(r)
        jq = dict(q, query_image=rel[q["query_image"]],
                  options=[dict(o, image=rel[o["image"]]) for o in q["options"]] + [{"model": NOTA}])
        jl.write(json.dumps(jq) + "\n")
    jl.close()
    pd.DataFrame(flat).to_csv(f"{args.out}/questions.csv", index=False)
    albums.assign(image=albums.image.map(rel)).to_csv(f"{args.out}/albums.csv", index=False)
    pd.read_csv(f"{args.bank_dir}/dropped.csv").to_csv(f"{args.out}/dropped.csv", index=False)
    pd.DataFrame(B["summary"]).to_csv(f"{args.out}/summary.csv", index=False)

    # preview pages (GitHub renders the HTML tables inside markdown; image paths are relative)
    pv = os.path.join(args.out, "preview")
    os.makedirs(pv, exist_ok=True)
    F = pd.DataFrame(flat)
    idx = ["# Question bank preview", "",
           "Each page shows up to 50 questions. Green = correct option. Option 6 is always "
           "\"None of the above\" (correct only when the true model is not among the 5 images).", ""]
    for ds in NAMES:
        idx += [f"## {NAMES[ds]}", ""]
        for qs in ["test"] + [f"train{i}" for i in range(1, B["chunks"] + 1)]:
            sub = F[(F.dataset == ds) & (F.query_set == qs)]
            pages = [sub.iloc[i:i + PER_PAGE] for i in range(0, len(sub), PER_PAGE)]
            links = []
            for pn, pg in enumerate(pages, 1):
                name = f"{ds}_{qs}_p{pn:02d}.md"
                links.append(f"[{pg.qid.iloc[0]}–{pg.qid.iloc[-1]}]({name})")
                L = [f"# {NAMES[ds]} · query set `{qs}` · page {pn}/{len(pages)}", "",
                     "[← all pages](README.md)", ""]
                for _, r in pg.iterrows():
                    head = (f"### {r.qid} · correct: option {r.correct_option}"
                            + (" (None of the above)" if r.correct_option == 6 else f" · `{r.query_model}`"))
                    cells = [("Query", r.query_image, r.query_model, False)]
                    for k in range(1, r.n_image_options + 1):
                        cells.append((f"Option {k}", r[f"option_{k}_image"], r[f"option_{k}_model"],
                                      r.correct_option == k))
                    L += [head, "", "<table><tr>"]
                    for t, im, m, ok in cells:
                        mark = "✅ " if ok else ""
                        L.append(f'<td align="center" valign="top"><b>{mark}{t}</b><br>'
                                 f'<img src="../{im}" width="120"><br><sub>{m}</sub></td>')
                    mark = "✅ " if r.correct_option == 6 else ""
                    L += [f'<td align="center" valign="middle"><b>{mark}Option 6</b><br><br>'
                          f"<i>None of the above</i></td>", "</tr></table>", ""]
                open(os.path.join(pv, name), "w").write("\n".join(L) + "\n")
            idx.append(f"- **{qs}** ({len(sub)} questions): " + " · ".join(links))
        idx.append("")
    open(os.path.join(pv, "README.md"), "w").write("\n".join(idx) + "\n")
    print(f"{len(flat)} questions, {len(rel)} images -> {args.out}")


if __name__ == "__main__":
    main()
