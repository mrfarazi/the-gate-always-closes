# Convenience targets. GPU targets assume the LLaVA stack (requirements.txt) is active.
PY ?= python
GPU ?= 0

.PHONY: install install-backbones test tables check-tables smoke-train smoke-knockout scrub-check download

install:            ## editable install of the hra package (+ pinned LLaVA-1.5 stack)
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

install-backbones:  ## InternVL3 / Qwen2-VL stack (separate Python 3.11 env recommended)
	$(PY) -m pip install -r requirements-backbones.txt && $(PY) -m pip install -e .

test:               ## CPU unit tests (< 1 min)
	$(PY) -m pytest -q tests

tables:             ## regenerate results/tables/*.tex, results/NUMBERS.md and RESULTS.md (no GPU)
	$(PY) scripts/make_tables.py

check-tables:       ## regenerate into a temp dir and compare with tests/golden/tables
	$(PY) scripts/make_tables.py --out /tmp/tgac_tables --check tests/golden/tables

smoke-train:        ## ~2 min per config on one GPU; writes runs/*/ with variant.json, no checkpoint
	scripts/train_llava.sh attention none --smoke --gpus $(GPU)
	scripts/train_llava.sh prefix raw_gated --smoke --gpus $(GPU)

smoke-knockout:     ## 16-item knock-outs (needs data + checkpoints)
	bash scripts/knockout/smoke_all.sh $(GPU)

scrub-check:        ## no internal identifiers / machine paths / credentials in tracked files
	bash scripts/release/scrub_check.sh && $(PY) scripts/release/scrub_result_paths.py --check

download:           ## make download NAME=raw_gated
	$(PY) -c "from hra import hub; print(hub.download_checkpoint('$(NAME)'))"

