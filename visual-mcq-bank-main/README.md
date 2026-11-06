# Visual MCQ question bank: implant identification from X-rays

**Status: draft for review (5 Nov 2026).** Once approved, this bank is frozen and used unchanged for
every experiment until the paper is submitted.

This is a multiple-choice test made only of images. Each question shows a **query X-ray**, the
**5 candidate X-rays** that our retrieval system shortlists, and a 6th option, **"None of the
above"**. A model must say which candidate shows the same implant model as the query. No implant
name, manufacturer or body part is ever shown to the model being tested. The names in this
repository are only there for the people reviewing the bank.

**Browse the questions:** [preview/README.md](preview/README.md) shows every question with its
images. The correct option is marked ✅ and every image has its implant model name under it.

## Size

4,752 questions over 4 public datasets and 84 implant models.

| Dataset | classes | test | train1 | train2 | train3 | train4 | train5 | total | NOTA correct | system top-1 | system top-5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OrthoNet hip | 8 | 119 | 147 | 157 | 167 | 156 | 164 | 910 | 8 | 78.2 | 99.1 |
| OrthoNet knee | 4 | 60 | 83 | 75 | 68 | 71 | 66 | 423 | 0 | 98.6 | 100.0 |
| Physionet | 27 | 354 | 268 | 310 | 242 | 303 | 268 | 1745 | 39 | 85.0 | 97.8 |
| Pacemakers | 45 | 225 | 290 | 290 | 290 | 290 | 289 | 1674 | 15 | 88.0 | 99.1 |
| **All** | 84 | 758 | 788 | 832 | 767 | 820 | 787 | **4752** | 62 | 86.0 | 98.7 |

*System top-1* is how often our retrieval system's first choice is correct. *System top-5* is how
often the true model is among the 5 image options. Those are the questions where "None of the
above" is wrong.

## How each question is built

1. **Retrieval system: strict leave-one-dataset-out (LODO).** For each dataset, the networks
   (Swin-B, 4 or 5 ConvNeXt seeds, frozen DINOv2) were trained once, only on the other 3 datasets.
   They have never seen any image of the dataset being tested, and nothing was retrained for this
   bank. This is the same system as our reported strict-LODO table (mean top-1 86.60 over 10 album
   draws). Dr Batra's private hip data is not used anywhere.
2. **Query sets.** Each dataset's original test split is the query set `test`. Its original train
   split is cut into 5 chunks, `train1` to `train5`. Every chunk has about the same mix of implant
   models, and the chunks are about the size of the test split.
   - Hip, knee and Physionet are cut by patient, so one patient's films never appear on both the
     query side and the album side.
   - Pacemakers has no patient ID, so its chunks are cut by image.
   - Every image of the 4 datasets is the query of exactly one question.
3. **Album.** For each query set, 5 random X-rays of every implant model are drawn from the
   *rest* of that dataset.
   - For `test`, the album comes from the train split. It is the same draw as in our reported
     numbers.
   - For `train k`, the album comes from the test split plus the other 4 chunks.
   - [albums.csv](albums.csv) lists every album.
4. **Shortlist.** The system ranks the implant models for the query and keeps the top 5. Knee has
   only 4 models, so its questions have 4 image options.
5. **Options.** Each shortlisted model is shown as its album X-ray that is closest to the query in
   the system's feature space.
   - The image options are shuffled with a fixed seed. The correct answer is spread evenly over
     positions 1-5.
   - Option 6 is always the text "None of the above".
6. **Correct answer.** The position of the true model among the image options. If the true model
   is not in the shortlist (62 questions, 1.3%), the correct answer is option 6. This is the natural
   rate of the system's misses; no question was altered to force "None of the above".

## Prompt the model will see (phase 2)

```
Query image: <image>
Candidate images:
Image 1: <image>
Image 2: <image>
Image 3: <image>
Image 4: <image>
Image 5: <image>
Option 6: None of the above
Which candidate image shows the same implant model as the query image? If none of them does,
answer 6. Answer with the number only.
```

The answer counts as correct if it is the right image, or 6 when the true model is not among the
images.

## Files

| File | What it holds |
|---|---|
| `questions.csv` | One row per question. The query image and its true model, option 1-5 images and their models, `option_6` = None of the above, `correct_option` (1-6), `correct_answer`, the system's top-1 model, and `n_image_options` (5, or 4 on knee). |
| `questions.jsonl` | The same questions with every field, plus each option's rank and score in the system. |
| `albums.csv` | The album (5 per model) behind each dataset and query set. |
| `summary.csv` | Counts and system accuracy for each dataset and query set. |
| `image_index.csv` | Each image file and its model, original split and query set. |
| `dropped.csv` | 24 files removed before building (see below). |
| `images/` | Every image, with its original bytes and neutral file names (no label in the name). |
| `preview/` | Browsable pages, 50 questions per page. |
| `code/` | The scripts that built this bank (they need our internal feature caches to run). |

## Clean-up before building

- **22 files were exact byte copies** of another image. Only one copy is kept, so no question can
  show the query's own copy as an option.
  - One OrthoNet hip test image also had a copy in the train split. One test image was itself
    duplicated, which is why hip test has 119 questions, not 120.
- **One Physionet file appears under two different models** (Ellipse VR and Rivacor 7 HF-T QP).
  Both copies are removed.
- Every removed file is listed in `dropped.csv` with the reason.

## Reproduce

```
python code/visual_mcq_bank.py --index lodo_index.csv --aug-dir <v96/support_aug_v2> \
       --extra-dir <v99/netst> --out <bank_dir>
python code/visual_mcq_bank_export.py --bank-dir <bank_dir> --index lodo_index.csv --out <repo_dir>
```

Seed 0 is used for the chunks, the album draws and the option order.

## Data

The images come from OrthoNet, the Physionet cardiac-device dataset and the Pacemaker dataset of
Howard et al. They are shared here privately for review only. Do not redistribute them; each
dataset's own licence applies.
