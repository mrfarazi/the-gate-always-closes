"""Gated caption-invariant hyperbolic prefix on Qwen2-VL-7B (paper Sec. 3.3, Table 1).

The S5 recipe (raw, gated prefix: FixedPrefixGraph with no_rms=True and a tanh-gated scalar
initialised at 0) ported from the authors' Qwen2-VL adapter. Only the graph pathway is kept.

Injection. Qwen2-VL pre-expands the image tokens in `input_ids` and derives its 3-D M-RoPE
positions from `input_ids`, so the K prefix embeddings cannot be spliced into the sequence the
way the LLaVA and InternVL wrappers do. Instead the collator RESERVES K pad-token slots right
after the image span (their indices are returned as `prefix_pos`; labels and attention already
account for them; `get_rope_index` sees them as ordinary text positions) and the wrapper
overwrites their embeddings with `prefix_tokens * tanh(gate)` before the language-model forward.
The sequence length never changes.

RoI features are pooled over the post-merge vision grid (3584-d), so FixedPrefixGraph is built
with vis_dim=3584 and its projector widened to K x 3584. The projector's (unused) RMSNorm keeps
its default width, exactly as in the released checkpoints, so state dicts load with strict=True.

Needs the backbone stack of requirements-backbones.txt (transformers >= 4.56). Importing this
module loads no model.
"""
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision.ops import roi_align

from hra.boxes import boxes_or_grid
from hra.checkpoints import read_variant, write_variant_json
from hra.paths import LOCAL_FILES_ONLY, QWEN2VL_ID
from hra.train_prefix_fixed import FixedPrefixGraph

K_PREFIX = 4
MAX_PROPOSALS = 36
MAX_PIXELS = 576 * 28 * 28      # bounds the merged image tokens at ~576 (LLaVA parity)
MIN_PIXELS = 64 * 28 * 28
PROMPT_SUFFIX = "\nAnswer the question using a single word or phrase."
VARIANT_KIND = "prefix_qwen2vl"
VARIANT_NAME = "raw_gated"


# ------------------------------------------------------------------ prefix graph
def build_prefix_graph(hidden_size=3584, k=K_PREFIX):
    """FixedPrefixGraph in its S5 form with the projector widened to k x hidden_size (CPU-safe)."""
    pg = FixedPrefixGraph(vis_dim=hidden_size, k=k, prefix_dropout=0.0,
                          no_rms=True, ungated=False)
    pg.prefix_proj.llm_dim = hidden_size
    pg.prefix_proj.proj = nn.Linear(pg.embed_dim, k * hidden_size)
    return pg


def graph_tokens(prefix_graph, feats, boxes, mask, gamma_override=None):
    """(B, K, H) fp32 prefix tokens = pre-gate graph tokens x tanh(gate) [or x gamma_override],
    plus the entailment and angle losses. FixedPrefixGraph multiplies by tanh(gate) inside its
    projector; the projector's `ungated` flag is lifted for the call so the gate (or the forced
    gamma of a knock-out) is applied exactly once, here."""
    proj = prefix_graph.prefix_proj
    was_ungated = proj.ungated
    proj.ungated = True
    try:
        tokens, _, ent, ang = prefix_graph(feats.float(), boxes.float(), mask.float(),
                                           ref_embeds=None)
    finally:
        proj.ungated = was_ungated
    gate_val = torch.tanh(proj.gate) if gamma_override is None else gamma_override
    if not torch.is_tensor(gate_val):
        gate_val = torch.tensor(float(gate_val), device=tokens.device)
    return tokens * gate_val.to(tokens.device, tokens.dtype), ent, ang


# ------------------------------------------------------------------ data
class Collator:
    """Qwen chat template, answer-token labels, K reserved prefix slots after the image span.

    Items are dicts with `image` (path), `image_id` (for scene-graph boxes; missing -> 6x6 grid),
    `question` and `answer`."""

    def __init__(self, processor, scene_graphs=None, image_token_id=None, k_prefix=K_PREFIX):
        self.proc = processor
        self.tok = processor.tokenizer
        self.sgs = scene_graphs or {}
        self.k = k_prefix
        self.pad_id = self.tok.pad_token_id or self.tok.eos_token_id
        self.image_token_id = (image_token_id if image_token_id is not None
                               else self.tok.convert_tokens_to_ids("<|image_pad|>"))

    def encode_one(self, item, with_answer=True):
        img = Image.open(item["image"]).convert("RGB")
        msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": item["question"] + PROMPT_SUFFIX}]}]
        prompt = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        text, n_answer = prompt, 0
        if with_answer:
            text = prompt + str(item["answer"]) + self.tok.eos_token
            n_answer = (len(self.tok(text, add_special_tokens=False).input_ids)
                        - len(self.tok(prompt, add_special_tokens=False).input_ids))
        enc = self.proc(text=[text], images=[img], return_tensors="pt",
                        min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        ids = enc["input_ids"][0].tolist()
        e = max(i for i, t in enumerate(ids) if t == self.image_token_id)
        ids = ids[:e + 1] + [self.pad_id] * self.k + ids[e + 1:]
        prefix_pos = list(range(e + 1, e + 1 + self.k))
        labels = [-100] * len(ids)
        for j in range(len(ids) - n_answer, len(ids)):
            labels[j] = ids[j]
        bt, bm, _ = boxes_or_grid(self.sgs.get(item.get("image_id")), *img.size,
                                  max_n=MAX_PROPOSALS)
        return {"ids": ids, "labels": labels, "prefix_pos": prefix_pos,
                "pixel_values": enc["pixel_values"], "grid": enc["image_grid_thw"][0],
                "boxes": bt, "box_mask": bm}

    def __call__(self, items):
        encs = [self.encode_one(it) for it in items]
        B, L = len(encs), max(len(e["ids"]) for e in encs)
        input_ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        labels = torch.full((B, L), -100, dtype=torch.long)
        attn = torch.zeros(B, L, dtype=torch.long)
        prefix_pos = torch.zeros(B, self.k, dtype=torch.long)
        for i, e in enumerate(encs):
            n = len(e["ids"])
            input_ids[i, :n] = torch.tensor(e["ids"])
            labels[i, :n] = torch.tensor(e["labels"])
            attn[i, :n] = 1
            prefix_pos[i] = torch.tensor(e["prefix_pos"])
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn,
                "prefix_pos": prefix_pos,
                "pixel_values": torch.cat([e["pixel_values"] for e in encs], 0),
                "image_grid_thw": torch.stack([e["grid"] for e in encs]),
                "boxes": torch.stack([e["boxes"] for e in encs]),
                "box_mask": torch.stack([e["box_mask"] for e in encs])}

    @staticmethod
    def prompt_batch(enc, device):
        """Single prompt-only encoding (encode_one(..., with_answer=False)) -> generate() kwargs."""
        ids = torch.tensor([enc["ids"]], device=device)
        return {"pixel_values": enc["pixel_values"].to(device), "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "boxes": enc["boxes"].unsqueeze(0).to(device),
                "box_mask": enc["box_mask"].unsqueeze(0).to(device),
                "image_grid_thw": enc["grid"].unsqueeze(0).to(device),
                "prefix_pos": torch.tensor([enc["prefix_pos"]], device=device)}


# ------------------------------------------------------------------ model
class GatedPrefixQwen2VL(nn.Module):
    """Qwen2-VL + LoRA + gated caption-invariant hyperbolic prefix (Mechanism B, S5 form).

    forward(...) -> (outputs, aux) with aux = {ent_loss, ang_loss, kappa, gate_raw, gate_tanh}.
    """

    def __init__(self, vlm, processor, k_prefix=K_PREFIX):
        super().__init__()
        self.vlm = vlm
        self.processor = processor
        cfg = self._cfg()
        self.hidden_size = int(cfg.hidden_size)
        self.image_token_id = int(cfg.image_token_id)
        self.spatial_merge = int(cfg.vision_config.spatial_merge_size)
        self.k_prefix = k_prefix
        self.prefix_graph = build_prefix_graph(self.hidden_size, k_prefix)

    # -- module resolution (transformers 4.56 layout, PEFT tolerant)
    def _cfg(self):
        cfg = getattr(self.vlm, "config", None)
        if cfg is None and hasattr(self.vlm, "base_model"):
            cfg = self.vlm.base_model.model.config
        return cfg

    def _root(self):
        m = self.vlm
        if hasattr(m, "peft_config") and hasattr(m, "base_model"):   # PEFT -> HF model
            m = m.base_model.model
        return m

    def _find(self, attr):
        root = self._root()
        for cand in (root, getattr(root, "model", None)):
            if cand is not None and hasattr(cand, attr):
                return cand
        raise RuntimeError(f"no `{attr}` found on the Qwen2-VL model")

    def _embed_tokens(self, ids):
        return self.vlm.get_input_embeddings()(ids)

    @property
    def gate_tanh(self):
        return float(torch.tanh(self.prefix_graph.prefix_proj.gate.detach()))

    def prefix_parameters(self):
        """The graph optimiser group (includes kappa, eta and the gate scalar)."""
        return list(self.prefix_graph.parameters())

    # -- vision / RoI
    def _run_visual(self, pixel_values, image_grid_thw):
        visual = self._find("visual").visual
        dt = visual.get_dtype() if hasattr(visual, "get_dtype") else next(visual.parameters()).dtype
        return visual(pixel_values.type(dt), grid_thw=image_grid_thw)

    def _roi(self, merged, image_grid_thw, boxes, mask):
        """RoI-align the normalised boxes over each image's post-merge token grid."""
        B, N, _ = boxes.shape
        per_image, grids, offset = [], [], 0
        for i in range(image_grid_thw.shape[0]):
            t, h, w = (int(image_grid_thw[i, j].item()) for j in range(3))
            n_i = (t * h * w) // (self.spatial_merge ** 2)
            per_image.append(merged[offset:offset + n_i])
            offset += n_i
            grids.append(((t * h) // self.spatial_merge, w // self.spatial_merge))
        if offset != merged.shape[0] or len(per_image) != B:
            raise RuntimeError("image_grid_thw / merged token count mismatch")
        C = merged.shape[-1]
        out = torch.zeros(B, N, C, device=merged.device, dtype=torch.float32)
        for i in range(B):
            valid = int(mask[i].sum().item())
            hm, wm = grids[i]
            if valid == 0 or hm <= 0 or wm <= 0:
                continue
            grid_feat = per_image[i].reshape(hm, wm, C).permute(2, 0, 1)
            grid_feat = grid_feat.unsqueeze(0).float().contiguous()
            bxs = boxes[i, :valid].clone().float()
            bxs[:, [0, 2]] *= wm
            bxs[:, [1, 3]] *= hm
            rois = torch.cat([torch.zeros(valid, 1, device=bxs.device), bxs], dim=1)
            pooled = roi_align(grid_feat, rois, output_size=(1, 1), spatial_scale=1.0, aligned=True)
            out[i, :valid] = pooled.squeeze(-1).squeeze(-1)
        return out.to(merged.dtype)

    # -- embedding assembly shared by forward / generate
    def _assemble(self, pixel_values, input_ids, attention_mask, boxes, box_mask,
                  image_grid_thw, prefix_pos, gamma_override):
        image_embeds = self._run_visual(pixel_values, image_grid_thw)
        proposal = self._roi(image_embeds.detach(), image_grid_thw, boxes, box_mask)
        embeds = self._embed_tokens(input_ids)
        prefix, ent, ang = graph_tokens(self.prefix_graph, proposal, boxes, box_mask, gamma_override)
        prefix = prefix.to(embeds.dtype)
        pos_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i in range(input_ids.shape[0]):
            pos_mask[i, prefix_pos[i]] = True
        embeds = embeds.masked_scatter(pos_mask.unsqueeze(-1).expand_as(embeds), prefix.reshape(-1))
        image_mask = (input_ids == self.image_token_id).unsqueeze(-1).expand_as(embeds)
        embeds = embeds.masked_scatter(image_mask, image_embeds.to(embeds.device, embeds.dtype))
        owner = self._find("get_rope_index")
        owner.rope_deltas = None
        position_ids, _ = owner.get_rope_index(input_ids, image_grid_thw, None, attention_mask)
        return embeds, position_ids, ent, ang

    def forward(self, pixel_values, input_ids, attention_mask, boxes, box_mask, labels=None,
                image_grid_thw=None, prefix_pos=None, gamma_override=None):
        embeds, position_ids, ent, ang = self._assemble(
            pixel_values, input_ids, attention_mask, boxes, box_mask, image_grid_thw,
            prefix_pos, gamma_override)
        out = self.vlm(input_ids=None, inputs_embeds=embeds, attention_mask=attention_mask,
                       position_ids=position_ids, labels=labels, use_cache=False,
                       return_dict=True)
        aux = {"ent_loss": ent, "ang_loss": ang, "kappa": self.prefix_graph.kappa,
               "gate_raw": float(self.prefix_graph.prefix_proj.gate.detach()),
               "gate_tanh": self.gate_tanh}
        return out, aux

    @torch.no_grad()
    def generate(self, pixel_values, input_ids, attention_mask, boxes, box_mask,
                 image_grid_thw=None, prefix_pos=None, gamma_override=None,
                 max_new_tokens=16, temperature=0.0):
        """Manual cached decode loop (HF generate() rejects explicit 3-D position ids together
        with inputs_embeds in 4.56): prefill once with get_rope_index positions, then advance all
        three M-RoPE dims by one per new token. temperature < 1e-4 -> greedy, otherwise plain
        softmax sampling. Returns list[list[int]] (EOS included if produced)."""
        embeds, position_ids, _, _ = self._assemble(
            pixel_values, input_ids, attention_mask, boxes, box_mask, image_grid_thw,
            prefix_pos, gamma_override)
        B = input_ids.shape[0]
        eos = self.processor.tokenizer.eos_token_id
        out = self.vlm(input_ids=None, inputs_embeds=embeds, attention_mask=attention_mask,
                       position_ids=position_ids, use_cache=True, return_dict=True)
        past = out.past_key_values
        next_pos = position_ids.max(dim=-1).values.max(dim=0).values + 1     # (B,)
        attn = attention_mask
        done = torch.zeros(B, dtype=torch.bool, device=embeds.device)
        seqs = [[] for _ in range(B)]
        logits = out.logits[:, -1, :]
        for _ in range(max_new_tokens):
            if temperature < 1e-4:
                tok = logits.argmax(dim=-1, keepdim=True)
            else:
                tok = torch.multinomial(torch.softmax(logits.float() / temperature, dim=-1), 1)
            for b in range(B):
                if not done[b]:
                    seqs[b].append(int(tok[b, 0]))
                    if int(tok[b, 0]) == eos:
                        done[b] = True
            if bool(done.all()):
                break
            step_pos = next_pos.view(1, -1, 1).expand(3, B, 1).contiguous()
            next_pos = next_pos + 1
            attn = torch.cat([attn, (~done).long().unsqueeze(1)], dim=1)
            out = self.vlm(input_ids=None, inputs_embeds=self._embed_tokens(tok),
                           attention_mask=attn, position_ids=step_pos,
                           past_key_values=past, use_cache=True, return_dict=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
        return seqs

    # -- checkpoints
    def save_prefix_modules(self, path):
        torch.save({"prefix_graph": self.prefix_graph.state_dict(),
                    "kappa": self.prefix_graph.kappa.data,
                    "gate": self.prefix_graph.prefix_proj.gate.data}, path)

    def save_checkpoint(self, directory, **meta):
        """<directory>/{lora, prefix_modules.pt, variant.json}."""
        os.makedirs(directory, exist_ok=True)
        self.vlm.save_pretrained(os.path.join(directory, "lora"))
        self.save_prefix_modules(os.path.join(directory, "prefix_modules.pt"))
        write_variant_json(directory, kind=VARIANT_KIND, variant=VARIANT_NAME,
                           backbone=QWEN2VL_ID, k_prefix=self.k_prefix, **meta)


# ------------------------------------------------------------------ builders
def _load_base():
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(QWEN2VL_ID, local_files_only=LOCAL_FILES_ONLY)
    vlm = Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN2VL_ID, dtype=torch.bfloat16, local_files_only=LOCAL_FILES_ONLY)
    vlm.requires_grad_(False)
    return vlm, processor


def build_gated_qwen(lora_rank=16, lora_alpha=32, lora_targets=("q_proj", "v_proj"),
                     device="cuda", k_prefix=K_PREFIX):
    """Fresh model for training: frozen Qwen2-VL-7B + LoRA(q_proj, v_proj) + S5 prefix graph."""
    vlm, processor = _load_base()
    if lora_rank > 0:
        from peft import LoraConfig, get_peft_model
        vlm = get_peft_model(vlm, LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, target_modules=list(lora_targets),
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    wrapper = GatedPrefixQwen2VL(vlm, processor, k_prefix=k_prefix)
    return wrapper.to(device), processor


def load_gated_qwen(ckpt_dir, device="cuda", k_prefix=None, trainable=False):
    """Checkpoint directory (<dir>/lora + <dir>/prefix_modules.pt [+ variant.json]) -> wrapper."""
    from peft import PeftModel
    meta = read_variant(ckpt_dir) or {}
    if meta.get("kind", VARIANT_KIND) != VARIANT_KIND or meta.get("variant", VARIANT_NAME) != VARIANT_NAME:
        raise ValueError(f"{ckpt_dir}: variant.json says kind={meta.get('kind')!r} "
                         f"variant={meta.get('variant')!r}; this loader expects "
                         f"{VARIANT_KIND}/{VARIANT_NAME}")
    k = int(meta.get("k_prefix", K_PREFIX)) if k_prefix is None else k_prefix
    state = torch.load(os.path.join(ckpt_dir, "prefix_modules.pt"), map_location="cpu")
    vlm, processor = _load_base()
    lora = os.path.join(ckpt_dir, "lora")
    if os.path.isdir(lora):
        vlm = PeftModel.from_pretrained(vlm, lora, is_trainable=trainable)
    wrapper = GatedPrefixQwen2VL(vlm, processor, k_prefix=k)
    wrapper.prefix_graph.load_state_dict(state["prefix_graph"], strict=True)
    wrapper.checkpoint_dir = ckpt_dir
    return wrapper.to(device), processor
