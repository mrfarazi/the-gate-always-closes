---
license: mit
base_model: {base_model}
library_name: peft
tags:
  - vision-language
  - lora
  - accv-2026
  - gate-closure
  - {backbone_tag}
---

# {title}

Checkpoint **`{name}`** from *The Gate Always Closes: On Injecting Auxiliary Signals into Frozen
Vision-Language Models* (ACCV 2026, [arXiv:2607.23335](https://arxiv.org/abs/2607.23335)).
Paper label: **{paper_label}**. Code and the reproduction scripts:
<https://github.com/mrfarazi/the-gate-always-closes>.

## What this is

{description}

| field | value |
|---|---|
| kind | `{kind}` |
| variant (`VARIANT_CONFIGS` key) | `{variant}` |
| backbone | `{base_model}` (not redistributed; downloaded from its own repo) |
| learned gate | {gate_line} |
| trainer | `{trainer}` |
| files | {files} |

`variant.json` records the configuration the graph module must be rebuilt with; the release code
reads it (`hra.checkpoints`). The `lora/` adapter is a standard PEFT adapter for the base model.

## Load

```python
from hra import hub
from hra.checkpoints import load_for_eval
model, processor, wrapper = load_for_eval("{load_kind}", "{name}", "cuda")   # downloads this repo
```

## Behavioural knock-out (from the paper's result files)

{knockout_block}

## Licence and citation

Code and these adapter weights: MIT. The base model's own licence applies to the base weights
({base_model}). Please cite the paper:

```bibtex
@InProceedings{{Farazi_2026_ACCV,
    author    = {{Farazi, Moshiur and Ramasinghe, Sameera and Ciftler, Bekir Sait and Turza, Mahbub Ahmed and Rahman, Shafin}},
    title     = {{The Gate Always Closes: On Injecting Auxiliary Signals into Frozen Vision-Language Models}},
    booktitle = {{Proceedings of the Asian Conference on Computer Vision (ACCV)}},
    month     = {{December}},
    year      = {{2026}}
}}
```
