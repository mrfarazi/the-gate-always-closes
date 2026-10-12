# Data

All datasets live under `data/` (override with `HRA_DATA`). Nothing under `data/` is tracked in git
except `compositional_pairs.json`. `python scripts/data/check_data.py` prints which groups are
present and which files are missing.

## Expected tree

```
data/
  gqa/
    train_balanced_questions.json      training questions (943,000 over 72,140 images)
    testdev_balanced_questions.json    evaluation questions (12,578)
    train_sceneGraphs.json             scene graphs: training boxes
    val_sceneGraphs.json               scene graphs: training boxes
    images/                            <imageId>.jpg
  sugarcrepe/
    SugarCrepe_add_att/                datasets.save_to_disk layout, images embedded
    SugarCrepe_add_obj/
    SugarCrepe_replace_att/
    SugarCrepe_replace_obj/
    SugarCrepe_replace_rel/
    SugarCrepe_swap_att/
    SugarCrepe_swap_obj/
  winoground/
    test/                              datasets.save_to_disk layout, images embedded
  vsr/
    test/                              datasets.save_to_disk layout (captions + COCO file names)
  coco/
    train2017/                         images for VSR
    val2014/                           images for POPE
    val2017/                           images for captioning (5,000)
    annotations/captions_val2017.json  captioning references
  pope/
    coco_pope_random.json
    coco_pope_popular.json
    coco_pope_adversarial.json
  compositional_pairs.json             shipped in git; S3 stage-2 training pairs
```

Which parts you need:

| Task | Files |
|---|---|
| Any evaluation on GQA (testdev, GQA-rel, GQA-attr, knock-outs) | `gqa/testdev_balanced_questions.json`, `gqa/images/` (testdev images only) |
| SugarCrepe knock-outs and scoring | `sugarcrepe/*` |
| Winoground scoring (`eval_compositionality.py`) | `winoground/test` |
| VSR (matched mode) | `vsr/test`, `coco/train2017` |
| POPE | `pope/*.json`, `coco/val2014` |
| COCO captioning | `coco/val2017`, `coco/annotations/captions_val2017.json` |
| Training on LLaVA-1.5 or InternVL3 | `gqa/train_balanced_questions.json`, `gqa/train_sceneGraphs.json`, `gqa/val_sceneGraphs.json`, all `gqa/images/` |
| S3 stage 2 | `compositional_pairs.json` (shipped), `gqa/images/` |

## Sources

### GQA

Download from the GQA site (<https://cs.stanford.edu/people/dorarad/gqa/download.html>,
gqadataset.org): the balanced questions (`train_balanced_questions.json`,
`testdev_balanced_questions.json`), the scene graphs (`train_sceneGraphs.json`,
`val_sceneGraphs.json`) and the images (about 21 GB, one `.jpg` per image id). Place them as in
the tree above. Scene graphs exist for the train and val splits only; **GQA testdev has no scene
graphs**, so evaluation uses a 6x6 grid of region proposals (`hra/boxes.py`,
[PROTOCOL.md](PROTOCOL.md)). Training always used the real scene-graph boxes: every
train_balanced question has a scene graph. For evaluation only the testdev images are needed.
Licence and terms: see the GQA download page.

### SugarCrepe

Seven sub-tasks from the Hub as `HuggingFaceM4/SugarCrepe_<subtask>`, saved with
`datasets.save_to_disk` (images are embedded, so no COCO images are needed for SugarCrepe):

```bash
python scripts/data/download_hf_datasets.py --only sugarcrepe
```

7,514 pairs in total; the knock-outs take the first n items round-robin over the seven sub-tasks.
Licence: see the dataset cards on the Hub.

### Winoground

`facebook/winoground` (gated). Accept the terms on the Hub, run `hf auth login`, then

```bash
python scripts/data/download_hf_datasets.py --only winoground
```

Used only by `scripts/eval_compositionality.py`; group scores sit at chance for every
configuration and are reported in the supplement.

### VSR

`cambridgeltl/vsr_random`, test split, saved to `data/vsr/test`:

```bash
python scripts/data/download_hf_datasets.py --only vsr
```

`scripts/eval_vsr.py` opens each statement's image by file name under `data/coco/train2017/` and
skips statements whose image is not there; the 2,113 statements of Table 5 are those found in
train2017. COCO images: <https://cocodataset.org/#download> (2017 train images). Licence: VSR
dataset card on the Hub; COCO terms of use for the images.

### POPE

The three question files `coco_pope_random.json`, `coco_pope_popular.json` and
`coco_pope_adversarial.json` come from the POPE GitHub repository
(<https://github.com/RUCAIBox/POPE>, `output/coco/`); 3,000 questions each. Images are COCO
val2014 (<https://cocodataset.org/#download>). Licence: POPE repository and COCO terms of use.

### COCO captioning

COCO val2017 images (5,000) and `annotations/captions_val2017.json` from
<https://cocodataset.org/#download>. Metrics are computed with `pycocoevalcap`
(`requirements.txt`); METEOR and SPICE need a Java runtime (`environment.yml` installs
`openjdk`). Licence: COCO terms of use.

### compositional_pairs.json (shipped)

`data/compositional_pairs.json` is in git (11 MB): 50,000 entries, each a Visual Genome image id,
a positive relation phrase and four hard negatives (`replace_rel`, `replace_obj`, `replace_subj`,
`swap`). It is the training set of S3 stage 2 (`hra/train_stage2.py`), which resolves the image
ids under `data/gqa/images/` (GQA images are Visual Genome images). The file was produced by
`hra/generate_hard_negatives.py` from `data/vg/relationships.json`; that Visual Genome file is only
needed if you want to regenerate the pairs.

## Base models

The base models are not part of this repository and are loaded from the Hub with their own
licences:

| Model | Repository | Used by |
|---|---|---|
| LLaVA-1.5-7B | `llava-hf/llava-1.5-7b-hf` (Llama 2 community licence) | every LLaVA configuration |
| InternVL3-8B | `OpenGVLab/InternVL3-8B-hf` (`HRA_INTERNVL_ID`) | `scripts/train_internvl.py`, InternVL knock-outs |
| Qwen2-VL-7B | `Qwen/Qwen2-VL-7B-Instruct` (`HRA_QWEN2VL_ID`) | the Qwen2-VL checkpoints' base model |

A local copy of LLaVA at `models/llava-1.5-7b` (or `HRA_LLAVA_PATH`) is used when present;
otherwise the Hub id is used. `HF_HUB_OFFLINE=1` makes the loaders pass `local_files_only`.

## Checking

```bash
python scripts/data/check_data.py
```

prints one line per group (`ok`, `partial`, `MISSING`) with the missing paths, and whether the
LLaVA weights are local or will be downloaded.
