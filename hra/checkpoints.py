"""variant.json bookkeeping and the unified checkpoint loaders.

The prefix module's `no_rms` / `ungated` switches are constructor flags that are not part of the
saved state dict, so a checkpoint directory must say which variant it is. Trainers write
`variant.json`; `resolve_variant` reads it (falling back to an explicit `--variant`) and refuses
to guess from the path.
"""
import json
import os

VARIANT_FILE = "variant.json"


def write_variant_json(directory, **meta):
    os.makedirs(directory, exist_ok=True)
    payload = {"schema": 1}
    payload.update(meta)
    with open(os.path.join(directory, VARIANT_FILE), "w") as f:
        json.dump(payload, f, indent=1, default=str)
    return payload


def read_variant(directory):
    p = os.path.join(directory, VARIANT_FILE)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def resolve_variant(directory, variant=None):
    meta = read_variant(directory)
    if meta and meta.get("variant"):
        if variant and variant != meta["variant"]:
            raise ValueError(f"{directory}: variant.json says '{meta['variant']}' but "
                             f"'{variant}' was requested")
        return meta["variant"]
    if variant:
        return variant
    raise ValueError(f"no {VARIANT_FILE} in {directory}; pass variant= explicitly (inferring "
                     "the graph class from the path was removed because it cannot recover the "
                     "no_rms / ungated flags)")


def registry_variant_for_dir(directory):
    """Variant of a directory that is a registry download or a local-manifest override, else None."""
    from hra import hub
    real = os.path.realpath(directory)
    for name, spec in hub.REGISTRY.items():
        for cand in (hub.local_dir(name), hub._local_override(name)):
            if cand and os.path.realpath(cand) == real:
                return spec["variant"]
    return None


def build_prefix_graph_from_dir(directory, variant=None, prefix_dropout=0.0):
    """CPU-only: the prefix graph with its trained weights and the resolved variant name."""
    import torch
    from hra.train_prefix_fixed import build_prefix_graph_for_eval
    if variant is None and read_variant(directory) is None:
        variant = registry_variant_for_dir(directory)
    v = resolve_variant(directory, variant)
    pg = build_prefix_graph_for_eval(v, prefix_dropout=prefix_dropout)
    state = torch.load(os.path.join(directory, "prefix_modules.pt"), map_location="cpu")
    pg.load_state_dict(state["prefix_graph"], strict=True)
    return pg, v


def load_llava(dtype=None):
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from hra.paths import LLAVA_PATH
    processor = AutoProcessor.from_pretrained(LLAVA_PATH)
    vlm = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_PATH, torch_dtype=dtype or torch.bfloat16, low_cpu_mem_usage=True)
    vlm.requires_grad_(False)
    return vlm, processor


def attach_lora(vlm, directory):
    lora = os.path.join(directory, "lora")
    if os.path.exists(lora):
        from peft import PeftModel
        vlm.language_model = PeftModel.from_pretrained(vlm.language_model, lora)
    return vlm


def load_vanilla(device):
    vlm, processor = load_llava()
    return vlm.to(device).eval(), processor


def load_lora_only(ckpt, device):
    """LoRA weights of a checkpoint with the auxiliary pathway absent."""
    from hra import hub
    d = hub.resolve(ckpt)
    vlm, processor = load_llava()
    attach_lora(vlm, d)
    return vlm.to(device).eval(), processor


def load_prefix(ckpt, device, variant=None):
    """FixedPrefixWrapper for a gated/ungated prefix checkpoint (registry name or directory)."""
    from hra import hub
    from hra.train_prefix_fixed import FixedPrefixWrapper
    d = hub.resolve(ckpt)
    if variant is None and ckpt in hub.REGISTRY and read_variant(d) is None:
        variant = hub.REGISTRY[ckpt]["variant"]
    vlm, processor = load_llava()
    attach_lora(vlm, d)
    pg, v = build_prefix_graph_from_dir(d, variant)
    wrapper = FixedPrefixWrapper(vlm, processor, pg).to(device).eval()
    wrapper.variant = v
    wrapper.checkpoint_dir = d
    return wrapper, processor


def load_hra(ckpt, device):
    """HRAWrapper for an attention-bias checkpoint (registry name or directory)."""
    from hra import hub
    from hra.model import HRAWrapper
    d = hub.resolve(ckpt)
    wrapper, processor = HRAWrapper.load_for_eval(d, device)
    wrapper.checkpoint_dir = d
    return wrapper, processor


def detect_kind(ckpt):
    """'hra' if the directory has hra_modules.pt, 'prefix' if prefix_modules.pt, else 'lora_only'."""
    from hra import hub
    if ckpt is None:
        return "vanilla"
    if ckpt in hub.REGISTRY:
        k = hub.REGISTRY[ckpt]["kind"]
        return "hra" if k.startswith("hra") else "prefix" if k.startswith("prefix") else k
    d = hub.resolve(ckpt)
    if os.path.exists(os.path.join(d, "hra_modules.pt")):
        return "hra"
    if os.path.exists(os.path.join(d, "prefix_modules.pt")):
        return "prefix"
    return "lora_only"


def load_for_eval(model, ckpt, device, variant=None):
    """Uniform entry point for the evaluators.

    Returns (llava_model, processor, wrapper): `wrapper` is the HRA / prefix wrapper when the
    auxiliary pathway is on (its `.vlm` is the LLaVA model), else None. `model` in
    {auto, vanilla, lora_only, hra, prefix}; `auto` picks from the checkpoint contents.
    """
    if model == "auto":
        model = detect_kind(ckpt)
    if model == "vanilla":
        vlm, processor = load_vanilla(device)
        return vlm, processor, None
    if model == "lora_only":
        vlm, processor = load_lora_only(ckpt, device)
        return vlm, processor, None
    if model == "hra":
        wrapper, processor = load_hra(ckpt, device)
        return wrapper.vlm, processor, wrapper
    if model == "prefix":
        wrapper, processor = load_prefix(ckpt, device, variant=variant)
        return wrapper.vlm, processor, wrapper
    raise ValueError(f"unknown model type {model!r}")


def set_prefix_scale(wrapper, scale):
    """Multiply the prefix tokens produced by ``wrapper.prefix_graph`` by ``scale`` (slots kept).

    ``scale=0`` is the *zeroed* inference mode of the paper (prefix x 0 with its slots kept);
    ``scale=1`` leaves the graph unchanged. Works for every graph class: the first element of the
    graph's forward output is the prefix-token tensor.
    """
    scale = float(scale)
    if scale == 1.0:
        return wrapper
    graph = wrapper.prefix_graph
    orig = graph.forward

    def forward(*args, **kwargs):
        out = orig(*args, **kwargs)
        return (out[0] * scale,) + tuple(out[1:])

    graph.forward = forward
    wrapper.prefix_scale = scale
    return wrapper
