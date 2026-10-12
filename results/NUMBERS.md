# NUMBERS.md -- every number in results/tables/*.tex, with its source

Generated 2026-10-12T14:38:37 by `scripts/make_tables.py` (re-run it; this file is overwritten). Paths are relative to the repo root unless absolute.

Conventions: accuracies 1 decimal for n <= 2,000 items and 2 decimals above; deltas signed with the same precision; CIs as [lo, hi]. j5/j7 deltas and CIs are the paired bootstraps stored in the JSON (5,000 resamples, numpy default_rng(0), vs the value-0 arm). E1 matched-mode contrasts are recomputed with `scripts/analyze_iclr_e1.paired_bootstrap` (10,000 resamples, RandomState(42) reset per contrast; two-sided bootstrap p). Welch / one-sample t via scipy (available).

## Sources

Found (84):

- `results/external/gate-objective/g1b/seed0/FINAL.json`
- `results/external/gate-objective/g1b/seed1/FINAL.json`
- `results/external/gate-objective/g1b/seed2/FINAL.json`
- `results/external/gate-objective/g1b_attribute/seed0/FINAL.json`
- `results/external/gate-objective/g1b_attribute/seed1/FINAL.json`
- `results/external/gate-objective/g1b_attribute/seed2/FINAL.json`
- `results/external/gate-objective/g1b_count/seed0/FINAL.json`
- `results/external/gate-objective/g1b_count/seed1/FINAL.json`
- `results/external/gate-objective/g1b_count/seed2/FINAL.json`
- `results/external/gate-objective/g1b_detrap/seed0/FINAL.json`
- `results/external/gate-objective/g1b_detrap/seed1/FINAL.json`
- `results/external/gate-objective/g1b_detrap/seed2/FINAL.json`
- `results/external/gate-objective/g1b_presence/seed0/FINAL.json`
- `results/external/gate-objective/g1b_presence/seed1/FINAL.json`
- `results/external/gate-objective/g1b_presence/seed2/FINAL.json`
- `results/external/gate-objective/gsweep/w4aud_g1b0_gqa.json`
- `results/generation/captioning_metrics_final.json`
- `results/generation/pope_captioning_summary.json`
- `results/generation/rms_dropout_r1_pope_captioning.json`
- `results/generation/rms_ungated_pope_prefix_on.json`
- `results/knockouts/j5_accv_rms_ungated_gqa.json`
- `results/knockouts/j5_accv_rms_ungated_sugarcrepe.json`
- `results/knockouts/j5_hypervis_gqa.json`
- `results/knockouts/j5_hypervis_gqa_gridboxes.json`
- `results/knockouts/j5_hypervis_sugarcrepe.json`
- `results/knockouts/j7_base_r1_gqa_rel.json`
- `results/knockouts/j7_base_r1_sugarcrepe.json`
- `results/knockouts/j7_euclidean_gqa_rel.json`
- `results/knockouts/j7_euclidean_sugarcrepe.json`
- `results/knockouts/j7_hra_v1_gqa.json`
- `results/knockouts/j7_hra_v1_gqa_rel.json`
- `results/knockouts/j7_hra_v1_sugarcrepe.json`
- `results/knockouts/j7_lora_all_linear_r1_gqa_rel.json`
- `results/knockouts/j7_lora_all_linear_r1_sugarcrepe.json`
- `results/knockouts/j7_lora_all_linear_s2_gqa_rel.json`
- `results/knockouts/j7_lora_all_linear_s2_sugarcrepe.json`
- `results/knockouts/j7_no_dropout_gqa.json`
- `results/knockouts/j7_no_dropout_gqa_rel.json`
- `results/knockouts/j7_no_dropout_sugarcrepe.json`
- `results/knockouts/j7_raw_gated_gqa.json`
- `results/knockouts/j7_raw_gated_gqa_rel.json`
- `results/knockouts/j7_raw_gated_sugarcrepe.json`
- `results/knockouts/j7_small_alpha_gqa.json`
- `results/knockouts/j7_small_alpha_gqa_rel.json`
- `results/knockouts/j7_small_alpha_sugarcrepe.json`
- `results/knockouts/j7_soft_prompt_gqa_rel.json`
- `results/knockouts/j7_soft_prompt_sugarcrepe.json`
- `results/knockouts/j7_stage2_gqa_rel.json`
- `results/knockouts/j7_stage2_sugarcrepe.json`
- `results/knockouts/j7_synthesis_gqa_rel.json`
- `results/knockouts/j7_synthesis_sugarcrepe.json`
- `results/knockouts/verify_zero_boxes/verify_s4_box_source_n100.json`
- `results/knockouts/verify_zero_boxes/verify_zero_boxes_n150.json`
- `results/matched_mode/vsr/eucGeoS2_DROP/vsr_predictions.json`
- `results/matched_mode/vsr/eucGeoS3_DROP/vsr_predictions.json`
- `results/matched_mode/vsr/eucGeo_DROP/vsr_predictions.json`
- `results/matched_mode/vsr/rmsPub_DROP/vsr_predictions.json`
- `results/matched_mode/vsr/rmsR2_DROP/vsr_predictions.json`
- `results/matched_mode/vsr/rmsR3_DROP/vsr_predictions.json`
- `results/matched_mode/vsr/vanilla/vsr_predictions.json`
- `results/probes/euclid_s1_1ep_rel.json`
- `results/probes/euclid_s2_1ep_rel.json`
- `results/probes/euclid_s3_1ep_rel.json`
- `results/probes/gate_gradient.json`
- `results/probes/hyp_r2_1ep_rel.json`
- `results/probes/hyp_r3_1ep_rel.json`
- `results/probes/internvl_knockout.json`
- `results/probes/internvl_knockout_seed1.json`
- `results/probes/internvl_knockout_seed2.json`
- `results/probes/internvl_seed0.json`
- `results/probes/internvl_seed1.json`
- `results/probes/internvl_seed2.json`
- `results/probes/j6/j6_llava_base_r1.json`
- `results/probes/j6/j6_llava_euclidean.json`
- `results/probes/j6/j6_llava_lora_all_linear_r1.json`
- `results/probes/j6/j6_llava_lora_all_linear_s2.json`
- `results/probes/j6/j6_llava_no_dropout.json`
- `results/probes/j6/j6_llava_raw_gated.json`
- `results/probes/j6/j6_llava_rms_ungated.json`
- `results/probes/j6/j6_llava_soft_prompt.json`
- `results/probes/j6/j6_llava_synthesis.json`
- `results/probes/layerwise_attention.json`
- `results/published/no_dropout_lora_only_gqa_testdev_predictions.json`
- `results/published/s4_no_dropout_with_prefix.json`

Missing / skipped (1):

- `results/knockouts/j7_stage2_gqa.json  (optional)`

## tab_knockout_main

> off = bias x0 (S1-S3) / gamma=0 with the 4 zero prefix slots kept (S4-S5); on = bias x1 (learned beta) / gamma = learned tanh(gate); absent = prefix slots removed (published LoRA-only condition; not applicable to the attention-bias designs). All arms: 6x6 grid proposals, one wrapper (journal/scripts/j7_llava_knockout.py). GQA-rel = first 1,000 relational testdev questions; SugarCrepe = first 1,000 pairs (round-robin over 7 subtasks); GQA testdev (n=2,000) = first 2,000 testdev questions in file order (item_ids checked); n=12,578 = full testdev.

| quantity | value | source file | key path / derivation |
|---|---|---|---|
| hra_v1 / GQA-rel (n=1000): learned gate | -0.6958 | `results/knockouts/j7_hra_v1_gqa_rel.json` | learned_gate |
| hra_v1 / GQA-rel (n=1000): pathway off acc | 51.7 | `results/knockouts/j7_hra_v1_gqa_rel.json` | rows[gamma=0].acc |
| hra_v1 / GQA-rel (n=1000): pathway on acc | 52.2 | `results/knockouts/j7_hra_v1_gqa_rel.json` | rows[gamma=1].acc |
| hra_v1 / GQA-rel (n=1000): delta on-off | +0.5 | `results/knockouts/j7_hra_v1_gqa_rel.json` | rows[..=1].delta_vs_0_pp |
| hra_v1 / GQA-rel (n=1000): 95% CI | [-0.4, +1.3] | `results/knockouts/j7_hra_v1_gqa_rel.json` | rows[..].ci95 (paired bootstrap 5,000, seed 0) |
| hra_v1 / GQA-rel (n=1000): flips % (items whose prediction differs off vs on) | 4.4 (44/1000) | `results/knockouts/j7_hra_v1_gqa_rel.json` | per_item.pred[i][idx_off]!=per_item.pred[i][idx_on] |
| hra_v1 / GQA-rel (n=1000): bias_abs_max_grid_boxes | 0.7773 | `results/knockouts/j7_hra_v1_gqa_rel.json` | unit_check.bias_abs_max_grid_boxes |
| hra_v1 / GQA-rel (n=1000): bias_abs_mean_grid_boxes | 0.4179 | `results/knockouts/j7_hra_v1_gqa_rel.json` | unit_check.bias_abs_mean_grid_boxes |
| hra_v1 / GQA-rel (n=1000): bias_abs_max_zero_boxes_protocol | 0 | `results/knockouts/j7_hra_v1_gqa_rel.json` | unit_check.bias_abs_max_zero_boxes_protocol |
| hra_v1 / SugarCrepe (n=1000): learned gate | -0.6958 | `results/knockouts/j7_hra_v1_sugarcrepe.json` | learned_gate |
| hra_v1 / SugarCrepe (n=1000): pathway off acc | 77.4 | `results/knockouts/j7_hra_v1_sugarcrepe.json` | rows[gamma=0].acc |
| hra_v1 / SugarCrepe (n=1000): pathway on acc | 77.2 | `results/knockouts/j7_hra_v1_sugarcrepe.json` | rows[gamma=1].acc |
| hra_v1 / SugarCrepe (n=1000): delta on-off | -0.2 | `results/knockouts/j7_hra_v1_sugarcrepe.json` | rows[..=1].delta_vs_0_pp |
| hra_v1 / SugarCrepe (n=1000): 95% CI | [-0.7, +0.3] | `results/knockouts/j7_hra_v1_sugarcrepe.json` | rows[..].ci95 (paired bootstrap 5,000, seed 0) |
| hra_v1 / SugarCrepe (n=1000): flips % (items whose prediction differs off vs on) | 0.6 (6/1000) | `results/knockouts/j7_hra_v1_sugarcrepe.json` | per_item: sign(s_pos-s_neg) differs between arms |
| hra_v1 / SugarCrepe (n=1000): bias_abs_max_grid_boxes | 0.7656 | `results/knockouts/j7_hra_v1_sugarcrepe.json` | unit_check.bias_abs_max_grid_boxes |
| hra_v1 / SugarCrepe (n=1000): bias_abs_mean_grid_boxes | 0.4019 | `results/knockouts/j7_hra_v1_sugarcrepe.json` | unit_check.bias_abs_mean_grid_boxes |
| hra_v1 / SugarCrepe (n=1000): bias_abs_max_zero_boxes_protocol | 0 | `results/knockouts/j7_hra_v1_sugarcrepe.json` | unit_check.bias_abs_max_zero_boxes_protocol |
| hra_v1 / GQA testdev (n=12578): learned gate | -0.6958 | `results/knockouts/j7_hra_v1_gqa.json` | learned_gate |
| hra_v1 / GQA testdev (n=12578): pathway off acc | 57.93 | `results/knockouts/j7_hra_v1_gqa.json` | rows[gamma=0].acc |
| hra_v1 / GQA testdev (n=12578): pathway on acc | 58.55 | `results/knockouts/j7_hra_v1_gqa.json` | rows[gamma=1].acc |
| hra_v1 / GQA testdev (n=12578): delta on-off | +0.62 | `results/knockouts/j7_hra_v1_gqa.json` | rows[..=1].delta_vs_0_pp |
| hra_v1 / GQA testdev (n=12578): 95% CI | [+0.39, +0.86] | `results/knockouts/j7_hra_v1_gqa.json` | rows[..].ci95 (paired bootstrap 5,000, seed 0) |
| hra_v1 / GQA testdev (n=12578): flips % (items whose prediction differs off vs on) | 4.0 (508/12578) | `results/knockouts/j7_hra_v1_gqa.json` | per_item.pred[i][idx_off]!=per_item.pred[i][idx_on] |
| hra_v1 / GQA testdev (n=12578): bias_abs_max_grid_boxes | 0.7539 | `results/knockouts/j7_hra_v1_gqa.json` | unit_check.bias_abs_max_grid_boxes |
| hra_v1 / GQA testdev (n=12578): bias_abs_mean_grid_boxes | 0.3562 | `results/knockouts/j7_hra_v1_gqa.json` | unit_check.bias_abs_mean_grid_boxes |
| hra_v1 / GQA testdev (n=12578): bias_abs_max_zero_boxes_protocol | 0 | `results/knockouts/j7_hra_v1_gqa.json` | unit_check.bias_abs_max_zero_boxes_protocol |
| hra_v1 / GQA testdev (n=12578): item set | testdev dict order, semantic==rel filter, first n | `results/knockouts/j7_hra_v1_gqa.json` | gqa_items / item_ids |
| small_alpha / GQA-rel (n=1000): learned gate | -0.6005 | `results/knockouts/j7_small_alpha_gqa_rel.json` | learned_gate |
| small_alpha / GQA-rel (n=1000): pathway off acc | 53.2 | `results/knockouts/j7_small_alpha_gqa_rel.json` | rows[gamma=0].acc |
| small_alpha / GQA-rel (n=1000): pathway on acc | 53.2 | `results/knockouts/j7_small_alpha_gqa_rel.json` | rows[gamma=1].acc |
| small_alpha / GQA-rel (n=1000): delta on-off | +0.0 | `results/knockouts/j7_small_alpha_gqa_rel.json` | rows[..=1].delta_vs_0_pp |
| small_alpha / GQA-rel (n=1000): 95% CI | [-0.6, +0.5] | `results/knockouts/j7_small_alpha_gqa_rel.json` | rows[..].ci95 (paired bootstrap 5,000, seed 0) |
| small_alpha / GQA-rel (n=1000): flips % (items whose prediction differs off vs on) | 2.3 (23/1000) | `results/knockouts/j7_small_alpha_gqa_rel.json` | per_item.pred[i][idx_off]!=per_item.pred[i][idx_on] |
| small_alpha / GQA-rel (n=1000): bias_abs_max_grid_boxes | 0.8242 | `results/knockouts/j7_small_alpha_gqa_rel.json` | unit_check.bias_abs_max_grid_boxes |
| small_alpha / GQA-rel (n=1000): bias_abs_mean_grid_boxes | 0.4113 | `results/knockouts/j7_small_alpha_gqa_rel.json` | unit_check.bias_abs_mean_grid_boxes |
| small_alpha / GQA-rel (n=1000): bias_abs_max_zero_boxes_protocol | 0 | `results/knockouts/j7_small_alpha_gqa_rel.json` | unit_check.bias_abs_max_zero_boxes_protocol |
| small_alpha / SugarCrepe (n=1000): learned gate | -0.6005 | `results/knockouts/j7_small_alpha_sugarcrepe.json` | learned_gate |
| small_alpha / SugarCrepe (n=1000): pathway off acc | 78.8 | `results/knockouts/j7_small_alpha_sugarcrepe.json` | rows[gamma=0].acc |
| small_alpha / SugarCrepe (n=1000): pathway on acc | 78.7 | `results/knockouts/j7_small_alpha_sugarcrepe.json` | rows[gamma=1].acc |
| small_alpha / SugarCrepe (n=1000): delta on-off | -0.1 | `results/knockouts/j7_small_alpha_sugarcrepe.json` | rows[..=1].delta_vs_0_pp |
| small_alpha / SugarCrepe (n=1000): 95% CI | [-0.6, +0.3] | `results/knockouts/j7_small_alpha_sugarcrepe.json` | rows[..].ci95 (paired bootstrap 5,000, seed 0) |
| small_alpha / SugarCrepe (n=1000): flips % (items whose prediction differs off vs on) | 0.5 (5/1000) | `results/knockouts/j7_small_alpha_sugarcrepe.json` | per_item: sign(s_pos-s_neg) differs between arms |
| small_alpha / SugarCrepe (n=1000): bias_abs_max_grid_boxes | 0.7773 | `results/knockouts/j7_small_alpha_sugarcrepe.json` | unit_check.bias_abs_max_grid_boxes |
| small_alpha / SugarCrepe (n=1000): bias_abs_mean_grid_boxes | 0.3636 | `results/knockouts/j7_small_alpha_sugarcrepe.json` | unit_check.bias_abs_mean_grid_boxes |
| small_alpha / SugarCrepe (n=1000): bias_abs_max_zero_boxes_protocol | 0 | `results/knockouts/j7_small_alpha_sugarcrepe.json` | unit_check.bias_abs_max_zero_boxes_protocol |
| small_alpha / GQA testdev (n=12578): learned gate | -0.6005 | `results/knockouts/j7_small_alpha_gqa.json` | learned_gate |
| small_alpha / GQA testdev (n=12578): pathway off acc | 60.09 | `results/knockouts/j7_small_alpha_gqa.json` | rows[gamma=0].acc |
| small_alpha / GQA testdev (n=12578): pathway on acc | 60.14 | `results/knockouts/j7_small_alpha_gqa.json` | rows[gamma=1].acc |
| small_alpha / GQA testdev (n=12578): delta on-off | +0.06 | `results/knockouts/j7_small_alpha_gqa.json` | rows[..=1].delta_vs_0_pp |
| small_alpha / GQA testdev (n=12578): 95% CI | [-0.15, +0.26] | `results/knockouts/j7_small_alpha_gqa.json` | rows[..].ci95 (paired bootstrap 5,000, seed 0) |
| small_alpha / GQA testdev (n=12578): flips % (items whose prediction differs off vs on) | 2.6 (325/12578) | `results/knockouts/j7_small_alpha_gqa.json` | per_item.pred[i][idx_off]!=per_item.pred[i][idx_on] |
| small_alpha / GQA testdev (n=12578): bias_abs_max_grid_boxes | 0.6953 | `results/knockouts/j7_small_alpha_gqa.json` | unit_check.bias_abs_max_grid_boxes |
| small_alpha / GQA testdev (n=12578): bias_abs_mean_grid_boxes | 0.2855 | `results/knockouts/j7_small_alpha_gqa.json` | unit_check.bias_abs_mean_grid_boxes |
| small_alpha / GQA testdev (n=12578): bias_abs_max_zero_boxes_protocol | 0 | `results/knockouts/j7_small_alpha_gqa.json` | unit_check.bias_abs_max_zero_boxes_protocol |
| small_alpha / GQA testdev (n=12578): item set | testdev dict order, semantic==rel filter, first n | `results/knockouts/j7_small_alpha_gqa.json` | gqa_items / item_ids |
| stage2 / GQA-rel (n=1000): learned gate | -8.4962 | `results/knockouts/j7_stage2_gqa_rel.json` | learned_gate |
| stage2 / GQA-rel (n=1000): pathway off acc | 53.1 | `results/knockouts/j7_stage2_gqa_rel.json` | rows[gamma=0].acc |
| stage2 / GQA-rel (n=1000): pathway on acc | 53.3 | `results/knockouts/j7_stage2_gqa_rel.json` | rows[gamma=1].acc |
| stage2 / GQA-rel (n=1000): delta on-off | +0.2 | `results/knockouts/j7_stage2_gqa_rel.json` | rows[..=1].delta_vs_0_pp |
| stage2 / GQA-rel (n=1000): 95% CI | [-0.3, +0.8] | `results/knockouts/j7_stage2_gqa_rel.json` | rows[..].ci95 (paired bootstrap 5,000, seed 0) |
| stage2 / GQA-rel (n=1000): flips % (items whose prediction differs off vs on) | 1.5 (15/1000) | `results/knockouts/j7_stage2_gqa_rel.json` | per_item.pred[i][idx_off]!=per_item.pred[i][idx_on] |
| stage2 / GQA-rel (n=1000): bias_abs_max_grid_boxes | 2.584e-08 | `results/knockouts/j7_stage2_gqa_rel.json` | unit_check.bias_abs_max_grid_boxes |
| stage2 / GQA-rel (n=1000): bias_abs_mean_grid_boxes | 1.037e-08 | `results/knockouts/j7_stage2_gqa_rel.json` | unit_check.bias_abs_mean_grid_boxes |
| stage2 / GQA-rel (n=1000): bias_abs_max_zero_boxes_protocol | 0 | `results/knockouts/j7_stage2_gqa_rel.json` | unit_check.bias_abs_max_zero_boxes_protocol |
| stage2 / SugarCrepe (n=1000): learned gate | -8.4962 | `results/knockouts/j7_stage2_sugarcrepe.json` | learned_gate |
| stage2 / SugarCrepe (n=1000): pathway off acc | 77.2 | `results/knockouts/j7_stage2_sugarcrepe.json` | rows[gamma=0].acc |
| stage2 / SugarCrepe (n=1000): pathway on acc | 77.3 | `results/knockouts/j7_stage2_sugarcrepe.json` | rows[gamma=1].acc |
| stage2 / SugarCrepe (n=1000): delta on-off | +0.1 | `results/knockouts/j7_stage2_sugarcrepe.json` | rows[..=1].delta_vs_0_pp |
| stage2 / SugarCrepe (n=1000): 95% CI | [-0.3, +0.5] | `results/knockouts/j7_stage2_sugarcrepe.json` | rows[..].ci95 (paired bootstrap 5,000, seed 0) |
| stage2 / SugarCrepe (n=1000): flips % (items whose prediction differs off vs on) | 0.5 (5/1000) | `results/knockouts/j7_stage2_sugarcrepe.json` | per_item: sign(s_pos-s_neg) differs between arms |
| stage2 / SugarCrepe (n=1000): bias_abs_max_grid_boxes | 2.491e-08 | `results/knockouts/j7_stage2_sugarcrepe.json` | unit_check.bias_abs_max_grid_boxes |
| stage2 / SugarCrepe (n=1000): bias_abs_mean_grid_boxes | 9.172e-09 | `results/knockouts/j7_stage2_sugarcrepe.json` | unit_check.bias_abs_mean_grid_boxes |
| stage2 / SugarCrepe (n=1000): bias_abs_max_zero_boxes_protocol | 0 | `results/knockouts/j7_stage2_sugarcrepe.json` | unit_check.bias_abs_max_zero_boxes_protocol |
| no_dropout / GQA-rel (n=1000): learned gate | -0.3623 | `results/knockouts/j7_no_dropout_gqa_rel.json` | learned_gate |
| no_dropout / GQA-rel (n=1000): pathway off acc | 50.5 | `results/knockouts/j7_no_dropout_gqa_rel.json` | rows[gamma=0].acc |
| no_dropout / GQA-rel (n=1000): pathway on acc | 50.6 | `results/knockouts/j7_no_dropout_gqa_rel.json` | rows[gamma=-0.362262].acc |
| no_dropout / GQA-rel (n=1000): delta on-off | +0.1 | `results/knockouts/j7_no_dropout_gqa_rel.json` | rows[..=-0.362262].delta_vs_0_pp |
| no_dropout / GQA-rel (n=1000): 95% CI | [-1.1, +1.3] | `results/knockouts/j7_no_dropout_gqa_rel.json` | rows[..].ci95 (paired bootstrap 5,000, seed 0) |
| no_dropout / GQA-rel (n=1000): flips % (items whose prediction differs off vs on) | 9.7 (97/1000) | `results/knockouts/j7_no_dropout_gqa_rel.json` | per_item.pred[i][idx_off]!=per_item.pred[i][idx_on] |
| no_dropout / GQA-rel (n=1000): prefix absent acc | 53.4 | `results/knockouts/j7_no_dropout_gqa_rel.json` | rows[label=prefix_absent].acc |
| no_dropout / GQA-rel (n=1000): absent - off | +2.9 [+1.0, +4.8] | `results/knockouts/j7_no_dropout_gqa_rel.json` | rows[label=prefix_absent].delta_vs_0_pp / ci95 |
| no_dropout / SugarCrepe (n=1000): learned gate | -0.3623 | `results/knockouts/j7_no_dropout_sugarcrepe.json` | learned_gate |
| no_dropout / SugarCrepe (n=1000): pathway off acc | 80.1 | `results/knockouts/j7_no_dropout_sugarcrepe.json` | rows[gamma=0].acc |
| no_dropout / SugarCrepe (n=1000): pathway on acc | 81.5 | `results/knockouts/j7_no_dropout_sugarcrepe.json` | rows[gamma=-0.362262].acc |
| no_dropout / SugarCrepe (n=1000): delta on-off | +1.4 | `results/knockouts/j7_no_dropout_sugarcrepe.json` | rows[..=-0.362262].delta_vs_0_pp |
| no_dropout / SugarCrepe (n=1000): 95% CI | [+0.5, +2.3] | `results/knockouts/j7_no_dropout_sugarcrepe.json` | rows[..].ci95 (paired bootstrap 5,000, seed 0) |
| no_dropout / SugarCrepe (n=1000): flips % (items whose prediction differs off vs on) | 2.2 (22/1000) | `results/knockouts/j7_no_dropout_sugarcrepe.json` | per_item: sign(s_pos-s_neg) differs between arms |
| no_dropout / SugarCrepe (n=1000): prefix absent acc | 77.1 | `results/knockouts/j7_no_dropout_sugarcrepe.json` | rows[label=prefix_absent].acc |
| no_dropout / SugarCrepe (n=1000): absent - off | -3.0 [-4.4, -1.7] | `results/knockouts/j7_no_dropout_sugarcrepe.json` | rows[label=prefix_absent].delta_vs_0_pp / ci95 |
| no_dropout / GQA testdev (n=2000): learned gate | -0.3623 | `results/knockouts/j7_no_dropout_gqa.json` | learned_gate |
| no_dropout / GQA testdev (n=2000): pathway off acc | 58.6 | `results/knockouts/j7_no_dropout_gqa.json` | rows[gamma=0].acc |
| no_dropout / GQA testdev (n=2000): pathway on acc | 59.6 | `results/knockouts/j7_no_dropout_gqa.json` | rows[gamma=-0.362262].acc |
| no_dropout / GQA testdev (n=2000): delta on-off | +0.9 | `results/knockouts/j7_no_dropout_gqa.json` | rows[..=-0.362262].delta_vs_0_pp |
| no_dropout / GQA testdev (n=2000): 95% CI | [+0.1, +1.8] | `results/knockouts/j7_no_dropout_gqa.json` | rows[..].ci95 (paired bootstrap 5,000, seed 0) |
| no_dropout / GQA testdev (n=2000): flips % (items whose prediction differs off vs on) | 8.3 (166/2000) | `results/knockouts/j7_no_dropout_gqa.json` | per_item.pred[i][idx_off]!=per_item.pred[i][idx_on] |
| no_dropout / GQA testdev (n=2000): prefix absent acc | 60.1 | `results/knockouts/j7_no_dropout_gqa.json` | rows[label=prefix_absent].acc |
| no_dropout / GQA testdev (n=2000): absent - off | +1.5 [+0.0, +2.9] | `results/knockouts/j7_no_dropout_gqa.json` | rows[label=prefix_absent].delta_vs_0_pp / ci95 |
| no_dropout / GQA testdev (n=2000): item set | testdev dict order, semantic==rel filter, first n | `results/knockouts/j7_no_dropout_gqa.json` | gqa_items / item_ids |
| raw_gated / GQA-rel (n=1000): learned gate | +0.0052 | `results/knockouts/j7_raw_gated_gqa_rel.json` | learned_gate |
| raw_gated / GQA-rel (n=1000): pathway off acc | 50.7 | `results/knockouts/j7_raw_gated_gqa_rel.json` | rows[gamma=0].acc |
| raw_gated / GQA-rel (n=1000): pathway on acc | 44.0 | `results/knockouts/j7_raw_gated_gqa_rel.json` | rows[gamma=0.00519311].acc |
| raw_gated / GQA-rel (n=1000): delta on-off | -6.7 | `results/knockouts/j7_raw_gated_gqa_rel.json` | rows[..=0.00519311].delta_vs_0_pp |
| raw_gated / GQA-rel (n=1000): 95% CI | [-8.4, -5.0] | `results/knockouts/j7_raw_gated_gqa_rel.json` | rows[..].ci95 (paired bootstrap 5,000, seed 0) |
| raw_gated / GQA-rel (n=1000): flips % (items whose prediction differs off vs on) | 19.5 (195/1000) | `results/knockouts/j7_raw_gated_gqa_rel.json` | per_item.pred[i][idx_off]!=per_item.pred[i][idx_on] |
| raw_gated / GQA-rel (n=1000): prefix absent acc | 54.3 | `results/knockouts/j7_raw_gated_gqa_rel.json` | rows[label=prefix_absent].acc |
| raw_gated / GQA-rel (n=1000): absent - off | +3.6 [+1.8, +5.4] | `results/knockouts/j7_raw_gated_gqa_rel.json` | rows[label=prefix_absent].delta_vs_0_pp / ci95 |
| raw_gated / SugarCrepe (n=1000): learned gate | +0.0052 | `results/knockouts/j7_raw_gated_sugarcrepe.json` | learned_gate |
| raw_gated / SugarCrepe (n=1000): pathway off acc | 78.5 | `results/knockouts/j7_raw_gated_sugarcrepe.json` | rows[gamma=0].acc |
| raw_gated / SugarCrepe (n=1000): pathway on acc | 80.0 | `results/knockouts/j7_raw_gated_sugarcrepe.json` | rows[gamma=0.00519311].acc |
| raw_gated / SugarCrepe (n=1000): delta on-off | +1.5 | `results/knockouts/j7_raw_gated_sugarcrepe.json` | rows[..=0.00519311].delta_vs_0_pp |
| raw_gated / SugarCrepe (n=1000): 95% CI | [+0.7, +2.4] | `results/knockouts/j7_raw_gated_sugarcrepe.json` | rows[..].ci95 (paired bootstrap 5,000, seed 0) |
| raw_gated / SugarCrepe (n=1000): flips % (items whose prediction differs off vs on) | 1.9 (19/1000) | `results/knockouts/j7_raw_gated_sugarcrepe.json` | per_item: sign(s_pos-s_neg) differs between arms |
| raw_gated / SugarCrepe (n=1000): prefix absent acc | 76.8 | `results/knockouts/j7_raw_gated_sugarcrepe.json` | rows[label=prefix_absent].acc |
| raw_gated / SugarCrepe (n=1000): absent - off | -1.7 [-2.8, -0.6] | `results/knockouts/j7_raw_gated_sugarcrepe.json` | rows[label=prefix_absent].delta_vs_0_pp / ci95 |
| raw_gated / GQA testdev (n=2000): learned gate | +0.0052 | `results/knockouts/j7_raw_gated_gqa.json` | learned_gate |
| raw_gated / GQA testdev (n=2000): pathway off acc | 59.4 | `results/knockouts/j7_raw_gated_gqa.json` | rows[gamma=0].acc |
| raw_gated / GQA testdev (n=2000): pathway on acc | 54.1 | `results/knockouts/j7_raw_gated_gqa.json` | rows[gamma=0.00519311].acc |
| raw_gated / GQA testdev (n=2000): delta on-off | -5.2 | `results/knockouts/j7_raw_gated_gqa.json` | rows[..=0.00519311].delta_vs_0_pp |
| raw_gated / GQA testdev (n=2000): 95% CI | [-6.3, -4.2] | `results/knockouts/j7_raw_gated_gqa.json` | rows[..].ci95 (paired bootstrap 5,000, seed 0) |
| raw_gated / GQA testdev (n=2000): flips % (items whose prediction differs off vs on) | 13.6 (271/2000) | `results/knockouts/j7_raw_gated_gqa.json` | per_item.pred[i][idx_off]!=per_item.pred[i][idx_on] |
| raw_gated / GQA testdev (n=2000): prefix absent acc | 60.6 | `results/knockouts/j7_raw_gated_gqa.json` | rows[label=prefix_absent].acc |
| raw_gated / GQA testdev (n=2000): absent - off | +1.2 [-0.3, +2.6] | `results/knockouts/j7_raw_gated_gqa.json` | rows[label=prefix_absent].delta_vs_0_pp / ci95 |
| raw_gated / GQA testdev (n=2000): item set | testdev dict order, semantic==rel filter, first n | `results/knockouts/j7_raw_gated_gqa.json` | gqa_items / item_ids |

## tab_gated_prefix_supp

| quantity | value | source file | key path / derivation |
|---|---|---|---|
| base_r1: learned tanh gamma | -0.1538 | `results/knockouts/j7_base_r1_gqa_rel.json` | learned_gate |
| base_r1 / GQA-rel: gamma=0 | 49.9 | `results/knockouts/j7_base_r1_gqa_rel.json` | rows[gamma=0].acc |
| base_r1 / GQA-rel: learned | 50.2 | `results/knockouts/j7_base_r1_gqa_rel.json` | rows[gamma=-0.153756].acc |
| base_r1 / GQA-rel: gamma=1 | 53.3 (delta +3.4 [+1.4, +5.3]) | `results/knockouts/j7_base_r1_gqa_rel.json` | rows[gamma=1].acc / delta_vs_0_pp / ci95 |
| base_r1 / GQA-rel: absent | 53.2 (delta +3.3 [+1.3, +5.2]) | `results/knockouts/j7_base_r1_gqa_rel.json` | rows[label=prefix_absent] |
| base_r1 / GQA-rel: delta learned-0 [CI] | +0.3 [-0.5, +1.2] | `results/knockouts/j7_base_r1_gqa_rel.json` | rows[gamma=learned].delta_vs_0_pp / ci95 |
| base_r1 / SugarCrepe: gamma=0 | 79.4 | `results/knockouts/j7_base_r1_sugarcrepe.json` | rows[gamma=0].acc |
| base_r1 / SugarCrepe: learned | 80.7 | `results/knockouts/j7_base_r1_sugarcrepe.json` | rows[gamma=-0.153756].acc |
| base_r1 / SugarCrepe: gamma=1 | 77.5 (delta -1.9 [-3.0, -0.8]) | `results/knockouts/j7_base_r1_sugarcrepe.json` | rows[gamma=1].acc / delta_vs_0_pp / ci95 |
| base_r1 / SugarCrepe: absent | 76.9 (delta -2.5 [-3.8, -1.3]) | `results/knockouts/j7_base_r1_sugarcrepe.json` | rows[label=prefix_absent] |
| base_r1 / SugarCrepe: delta learned-0 [CI] | +1.3 [+0.6, +2.1] | `results/knockouts/j7_base_r1_sugarcrepe.json` | rows[gamma=learned].delta_vs_0_pp / ci95 |
| soft_prompt: learned tanh gamma | -0.1963 | `results/knockouts/j7_soft_prompt_gqa_rel.json` | learned_gate |
| soft_prompt / GQA-rel: gamma=0 | 51.6 | `results/knockouts/j7_soft_prompt_gqa_rel.json` | rows[gamma=0].acc |
| soft_prompt / GQA-rel: learned | 51.1 | `results/knockouts/j7_soft_prompt_gqa_rel.json` | rows[gamma=-0.196284].acc |
| soft_prompt / GQA-rel: gamma=1 | 52.8 (delta +1.2 [-0.7, +3.0]) | `results/knockouts/j7_soft_prompt_gqa_rel.json` | rows[gamma=1].acc / delta_vs_0_pp / ci95 |
| soft_prompt / GQA-rel: absent | 53.7 (delta +2.1 [+0.2, +4.0]) | `results/knockouts/j7_soft_prompt_gqa_rel.json` | rows[label=prefix_absent] |
| soft_prompt / GQA-rel: delta learned-0 [CI] | -0.5 [-1.3, +0.3] | `results/knockouts/j7_soft_prompt_gqa_rel.json` | rows[gamma=learned].delta_vs_0_pp / ci95 |
| soft_prompt / SugarCrepe: gamma=0 | 79.1 | `results/knockouts/j7_soft_prompt_sugarcrepe.json` | rows[gamma=0].acc |
| soft_prompt / SugarCrepe: learned | 79.3 | `results/knockouts/j7_soft_prompt_sugarcrepe.json` | rows[gamma=-0.196284].acc |
| soft_prompt / SugarCrepe: gamma=1 | 78.2 (delta -0.9 [-2.0, +0.1]) | `results/knockouts/j7_soft_prompt_sugarcrepe.json` | rows[gamma=1].acc / delta_vs_0_pp / ci95 |
| soft_prompt / SugarCrepe: absent | 77.3 (delta -1.8 [-3.0, -0.6]) | `results/knockouts/j7_soft_prompt_sugarcrepe.json` | rows[label=prefix_absent] |
| soft_prompt / SugarCrepe: delta learned-0 [CI] | +0.2 [-0.5, +0.9] | `results/knockouts/j7_soft_prompt_sugarcrepe.json` | rows[gamma=learned].delta_vs_0_pp / ci95 |
| no_dropout: learned tanh gamma | -0.3623 | `results/knockouts/j7_no_dropout_gqa_rel.json` | learned_gate |
| no_dropout / GQA-rel: gamma=0 | 50.5 | `results/knockouts/j7_no_dropout_gqa_rel.json` | rows[gamma=0].acc |
| no_dropout / GQA-rel: learned | 50.6 | `results/knockouts/j7_no_dropout_gqa_rel.json` | rows[gamma=-0.362262].acc |
| no_dropout / GQA-rel: gamma=1 | 53.1 (delta +2.6 [+0.7, +4.4]) | `results/knockouts/j7_no_dropout_gqa_rel.json` | rows[gamma=1].acc / delta_vs_0_pp / ci95 |
| no_dropout / GQA-rel: absent | 53.4 (delta +2.9 [+1.0, +4.8]) | `results/knockouts/j7_no_dropout_gqa_rel.json` | rows[label=prefix_absent] |
| no_dropout / GQA-rel: delta learned-0 [CI] | +0.1 [-1.1, +1.3] | `results/knockouts/j7_no_dropout_gqa_rel.json` | rows[gamma=learned].delta_vs_0_pp / ci95 |
| no_dropout / SugarCrepe: gamma=0 | 80.1 | `results/knockouts/j7_no_dropout_sugarcrepe.json` | rows[gamma=0].acc |
| no_dropout / SugarCrepe: learned | 81.5 | `results/knockouts/j7_no_dropout_sugarcrepe.json` | rows[gamma=-0.362262].acc |
| no_dropout / SugarCrepe: gamma=1 | 79.5 (delta -0.6 [-1.6, +0.4]) | `results/knockouts/j7_no_dropout_sugarcrepe.json` | rows[gamma=1].acc / delta_vs_0_pp / ci95 |
| no_dropout / SugarCrepe: absent | 77.1 (delta -3.0 [-4.4, -1.7]) | `results/knockouts/j7_no_dropout_sugarcrepe.json` | rows[label=prefix_absent] |
| no_dropout / SugarCrepe: delta learned-0 [CI] | +1.4 [+0.5, +2.3] | `results/knockouts/j7_no_dropout_sugarcrepe.json` | rows[gamma=learned].delta_vs_0_pp / ci95 |
| euclidean: learned tanh gamma | +0.1849 | `results/knockouts/j7_euclidean_gqa_rel.json` | learned_gate |
| euclidean / GQA-rel: gamma=0 | 51.0 | `results/knockouts/j7_euclidean_gqa_rel.json` | rows[gamma=0].acc |
| euclidean / GQA-rel: learned | 50.4 | `results/knockouts/j7_euclidean_gqa_rel.json` | rows[gamma=0.184864].acc |
| euclidean / GQA-rel: gamma=1 | 50.7 (delta -0.3 [-1.7, +1.2]) | `results/knockouts/j7_euclidean_gqa_rel.json` | rows[gamma=1].acc / delta_vs_0_pp / ci95 |
| euclidean / GQA-rel: absent | 53.5 (delta +2.5 [+0.7, +4.3]) | `results/knockouts/j7_euclidean_gqa_rel.json` | rows[label=prefix_absent] |
| euclidean / GQA-rel: delta learned-0 [CI] | -0.6 [-1.3, +0.0] | `results/knockouts/j7_euclidean_gqa_rel.json` | rows[gamma=learned].delta_vs_0_pp / ci95 |
| euclidean / SugarCrepe: gamma=0 | 79.4 | `results/knockouts/j7_euclidean_sugarcrepe.json` | rows[gamma=0].acc |
| euclidean / SugarCrepe: learned | 78.9 | `results/knockouts/j7_euclidean_sugarcrepe.json` | rows[gamma=0.184864].acc |
| euclidean / SugarCrepe: gamma=1 | 77.8 (delta -1.6 [-2.7, -0.6]) | `results/knockouts/j7_euclidean_sugarcrepe.json` | rows[gamma=1].acc / delta_vs_0_pp / ci95 |
| euclidean / SugarCrepe: absent | 76.9 (delta -2.5 [-3.7, -1.3]) | `results/knockouts/j7_euclidean_sugarcrepe.json` | rows[label=prefix_absent] |
| euclidean / SugarCrepe: delta learned-0 [CI] | -0.5 [-1.1, +0.1] | `results/knockouts/j7_euclidean_sugarcrepe.json` | rows[gamma=learned].delta_vs_0_pp / ci95 |
| synthesis: learned tanh gamma | +0.1186 | `results/knockouts/j7_synthesis_gqa_rel.json` | learned_gate |
| synthesis / GQA-rel: gamma=0 | 49.6 | `results/knockouts/j7_synthesis_gqa_rel.json` | rows[gamma=0].acc |
| synthesis / GQA-rel: learned | 48.1 | `results/knockouts/j7_synthesis_gqa_rel.json` | rows[gamma=0.118626].acc |
| synthesis / GQA-rel: gamma=1 | 49.9 (delta +0.3 [-1.1, +1.7]) | `results/knockouts/j7_synthesis_gqa_rel.json` | rows[gamma=1].acc / delta_vs_0_pp / ci95 |
| synthesis / GQA-rel: absent | 54.3 (delta +4.7 [+2.7, +6.7]) | `results/knockouts/j7_synthesis_gqa_rel.json` | rows[label=prefix_absent] |
| synthesis / GQA-rel: delta learned-0 [CI] | -1.5 [-2.6, -0.5] | `results/knockouts/j7_synthesis_gqa_rel.json` | rows[gamma=learned].delta_vs_0_pp / ci95 |
| synthesis / SugarCrepe: gamma=0 | 79.9 | `results/knockouts/j7_synthesis_sugarcrepe.json` | rows[gamma=0].acc |
| synthesis / SugarCrepe: learned | 80.4 | `results/knockouts/j7_synthesis_sugarcrepe.json` | rows[gamma=0.118626].acc |
| synthesis / SugarCrepe: gamma=1 | 82.8 (delta +2.9 [+1.9, +4.0]) | `results/knockouts/j7_synthesis_sugarcrepe.json` | rows[gamma=1].acc / delta_vs_0_pp / ci95 |
| synthesis / SugarCrepe: absent | 77.2 (delta -2.7 [-4.0, -1.4]) | `results/knockouts/j7_synthesis_sugarcrepe.json` | rows[label=prefix_absent] |
| synthesis / SugarCrepe: delta learned-0 [CI] | +0.5 [+0.0, +1.1] | `results/knockouts/j7_synthesis_sugarcrepe.json` | rows[gamma=learned].delta_vs_0_pp / ci95 |
| lora_all_linear_r1: learned tanh gamma | +0.0235 | `results/knockouts/j7_lora_all_linear_r1_gqa_rel.json` | learned_gate |
| lora_all_linear_r1 / GQA-rel: gamma=0 | 54.5 | `results/knockouts/j7_lora_all_linear_r1_gqa_rel.json` | rows[gamma=0].acc |
| lora_all_linear_r1 / GQA-rel: learned | 54.2 | `results/knockouts/j7_lora_all_linear_r1_gqa_rel.json` | rows[gamma=0.0235248].acc |
| lora_all_linear_r1 / GQA-rel: gamma=1 | 55.1 (delta +0.6 [-0.6, +1.9]) | `results/knockouts/j7_lora_all_linear_r1_gqa_rel.json` | rows[gamma=1].acc / delta_vs_0_pp / ci95 |
| lora_all_linear_r1 / GQA-rel: absent | 54.9 (delta +0.4 [-0.8, +1.5]) | `results/knockouts/j7_lora_all_linear_r1_gqa_rel.json` | rows[label=prefix_absent] |
| lora_all_linear_r1 / GQA-rel: delta learned-0 [CI] | -0.3 [-0.9, +0.3] | `results/knockouts/j7_lora_all_linear_r1_gqa_rel.json` | rows[gamma=learned].delta_vs_0_pp / ci95 |
| lora_all_linear_r1 / SugarCrepe: gamma=0 | 80.6 | `results/knockouts/j7_lora_all_linear_r1_sugarcrepe.json` | rows[gamma=0].acc |
| lora_all_linear_r1 / SugarCrepe: learned | 79.8 | `results/knockouts/j7_lora_all_linear_r1_sugarcrepe.json` | rows[gamma=0.0235248].acc |
| lora_all_linear_r1 / SugarCrepe: gamma=1 | 80.6 (delta +0.0 [-0.9, +0.9]) | `results/knockouts/j7_lora_all_linear_r1_sugarcrepe.json` | rows[gamma=1].acc / delta_vs_0_pp / ci95 |
| lora_all_linear_r1 / SugarCrepe: absent | 78.8 (delta -1.8 [-3.0, -0.6]) | `results/knockouts/j7_lora_all_linear_r1_sugarcrepe.json` | rows[label=prefix_absent] |
| lora_all_linear_r1 / SugarCrepe: delta learned-0 [CI] | -0.8 [-1.4, -0.3] | `results/knockouts/j7_lora_all_linear_r1_sugarcrepe.json` | rows[gamma=learned].delta_vs_0_pp / ci95 |
| lora_all_linear_s2: learned tanh gamma | -0.0300 | `results/knockouts/j7_lora_all_linear_s2_gqa_rel.json` | learned_gate |
| lora_all_linear_s2 / GQA-rel: gamma=0 | 54.3 | `results/knockouts/j7_lora_all_linear_s2_gqa_rel.json` | rows[gamma=0].acc |
| lora_all_linear_s2 / GQA-rel: learned | 54.6 | `results/knockouts/j7_lora_all_linear_s2_gqa_rel.json` | rows[gamma=-0.0300311].acc |
| lora_all_linear_s2 / GQA-rel: gamma=1 | 53.6 (delta -0.7 [-1.9, +0.5]) | `results/knockouts/j7_lora_all_linear_s2_gqa_rel.json` | rows[gamma=1].acc / delta_vs_0_pp / ci95 |
| lora_all_linear_s2 / GQA-rel: absent | 54.2 (delta -0.1 [-1.3, +1.1]) | `results/knockouts/j7_lora_all_linear_s2_gqa_rel.json` | rows[label=prefix_absent] |
| lora_all_linear_s2 / GQA-rel: delta learned-0 [CI] | +0.3 [-0.4, +1.0] | `results/knockouts/j7_lora_all_linear_s2_gqa_rel.json` | rows[gamma=learned].delta_vs_0_pp / ci95 |
| lora_all_linear_s2 / SugarCrepe: gamma=0 | 80.3 | `results/knockouts/j7_lora_all_linear_s2_sugarcrepe.json` | rows[gamma=0].acc |
| lora_all_linear_s2 / SugarCrepe: learned | 80.1 | `results/knockouts/j7_lora_all_linear_s2_sugarcrepe.json` | rows[gamma=-0.0300311].acc |
| lora_all_linear_s2 / SugarCrepe: gamma=1 | 78.7 (delta -1.6 [-2.7, -0.5]) | `results/knockouts/j7_lora_all_linear_s2_sugarcrepe.json` | rows[gamma=1].acc / delta_vs_0_pp / ci95 |
| lora_all_linear_s2 / SugarCrepe: absent | 77.9 (delta -2.4 [-3.7, -1.1]) | `results/knockouts/j7_lora_all_linear_s2_sugarcrepe.json` | rows[label=prefix_absent] |
| lora_all_linear_s2 / SugarCrepe: delta learned-0 [CI] | -0.2 [-0.7, +0.3] | `results/knockouts/j7_lora_all_linear_s2_sugarcrepe.json` | rows[gamma=learned].delta_vs_0_pp / ci95 |
| raw_gated: learned tanh gamma | +0.0052 | `results/knockouts/j7_raw_gated_gqa_rel.json` | learned_gate |
| raw_gated / GQA-rel: gamma=0 | 50.7 | `results/knockouts/j7_raw_gated_gqa_rel.json` | rows[gamma=0].acc |
| raw_gated / GQA-rel: learned | 44.0 | `results/knockouts/j7_raw_gated_gqa_rel.json` | rows[gamma=0.00519311].acc |
| raw_gated / GQA-rel: gamma=1 | 12.2 (delta -38.5 [-41.7, -35.2]) | `results/knockouts/j7_raw_gated_gqa_rel.json` | rows[gamma=1].acc / delta_vs_0_pp / ci95 |
| raw_gated / GQA-rel: absent | 54.3 (delta +3.6 [+1.8, +5.4]) | `results/knockouts/j7_raw_gated_gqa_rel.json` | rows[label=prefix_absent] |
| raw_gated / GQA-rel: delta learned-0 [CI] | -6.7 [-8.4, -5.0] | `results/knockouts/j7_raw_gated_gqa_rel.json` | rows[gamma=learned].delta_vs_0_pp / ci95 |
| raw_gated / SugarCrepe: gamma=0 | 78.5 | `results/knockouts/j7_raw_gated_sugarcrepe.json` | rows[gamma=0].acc |
| raw_gated / SugarCrepe: learned | 80.0 | `results/knockouts/j7_raw_gated_sugarcrepe.json` | rows[gamma=0.00519311].acc |
| raw_gated / SugarCrepe: gamma=1 | 79.5 (delta +1.0 [-1.6, +3.6]) | `results/knockouts/j7_raw_gated_sugarcrepe.json` | rows[gamma=1].acc / delta_vs_0_pp / ci95 |
| raw_gated / SugarCrepe: absent | 76.8 (delta -1.7 [-2.8, -0.6]) | `results/knockouts/j7_raw_gated_sugarcrepe.json` | rows[label=prefix_absent] |
| raw_gated / SugarCrepe: delta learned-0 [CI] | +1.5 [+0.7, +2.4] | `results/knockouts/j7_raw_gated_sugarcrepe.json` | rows[gamma=learned].delta_vs_0_pp / ci95 |

## tab_s5_sweep_supp

| quantity | value | source file | key path / derivation |
|---|---|---|---|
| raw_gated GQA-rel gamma=0: acc / delta / CI | 50.7 / +0.0 [+0.0, +0.0] | `results/knockouts/j7_raw_gated_gqa_rel.json` | rows[gamma=0] |
| raw_gated SugarCrepe gamma=0: acc / delta / CI | 78.5 / +0.0 [+0.0, +0.0] | `results/knockouts/j7_raw_gated_sugarcrepe.json` | rows[gamma=0] |
| raw_gated GQA-rel gamma=0.00519311: acc / delta / CI | 44.0 / -6.7 [-8.4, -5.0] | `results/knockouts/j7_raw_gated_gqa_rel.json` | rows[gamma=0.00519311] |
| raw_gated SugarCrepe gamma=0.00519311: acc / delta / CI | 80.0 / +1.5 [+0.7, +2.4] | `results/knockouts/j7_raw_gated_sugarcrepe.json` | rows[gamma=0.00519311] |
| raw_gated GQA-rel gamma=-0.00519311: acc / delta / CI | 53.5 / +2.8 [+1.5, +4.2] | `results/knockouts/j7_raw_gated_gqa_rel.json` | rows[gamma=-0.00519311] |
| raw_gated SugarCrepe gamma=-0.00519311: acc / delta / CI | 78.0 / -0.5 [-1.4, +0.4] | `results/knockouts/j7_raw_gated_sugarcrepe.json` | rows[gamma=-0.00519311] |
| raw_gated GQA-rel gamma=0.001: acc / delta / CI | 50.5 / -0.2 [-1.0, +0.6] | `results/knockouts/j7_raw_gated_gqa_rel.json` | rows[gamma=0.001] |
| raw_gated SugarCrepe gamma=0.001: acc / delta / CI | 79.1 / +0.6 [+0.1, +1.2] | `results/knockouts/j7_raw_gated_sugarcrepe.json` | rows[gamma=0.001] |
| raw_gated GQA-rel gamma=0.003: acc / delta / CI | 47.4 / -3.3 [-4.7, -2.0] | `results/knockouts/j7_raw_gated_gqa_rel.json` | rows[gamma=0.003] |
| raw_gated SugarCrepe gamma=0.003: acc / delta / CI | 79.4 / +0.9 [+0.3, +1.6] | `results/knockouts/j7_raw_gated_sugarcrepe.json` | rows[gamma=0.003] |
| raw_gated GQA-rel gamma=0.01: acc / delta / CI | 36.4 / -14.3 [-16.7, -12.0] | `results/knockouts/j7_raw_gated_gqa_rel.json` | rows[gamma=0.01] |
| raw_gated SugarCrepe gamma=0.01: acc / delta / CI | 81.2 / +2.7 [+1.6, +3.9] | `results/knockouts/j7_raw_gated_sugarcrepe.json` | rows[gamma=0.01] |
| raw_gated GQA-rel gamma=0.03: acc / delta / CI | 39.2 / -11.5 [-13.9, -9.2] | `results/knockouts/j7_raw_gated_gqa_rel.json` | rows[gamma=0.03] |
| raw_gated SugarCrepe gamma=0.03: acc / delta / CI | 85.3 / +6.8 [+5.1, +8.5] | `results/knockouts/j7_raw_gated_sugarcrepe.json` | rows[gamma=0.03] |
| raw_gated GQA-rel gamma=0.1: acc / delta / CI | 33.9 / -16.8 [-19.5, -14.2] | `results/knockouts/j7_raw_gated_gqa_rel.json` | rows[gamma=0.1] |
| raw_gated SugarCrepe gamma=0.1: acc / delta / CI | 84.7 / +6.2 [+4.5, +7.9] | `results/knockouts/j7_raw_gated_sugarcrepe.json` | rows[gamma=0.1] |
| raw_gated GQA-rel gamma=0.3: acc / delta / CI | 49.1 / -1.6 [-3.7, +0.5] | `results/knockouts/j7_raw_gated_gqa_rel.json` | rows[gamma=0.3] |
| raw_gated SugarCrepe gamma=0.3: acc / delta / CI | 82.3 / +3.8 [+1.7, +5.9] | `results/knockouts/j7_raw_gated_sugarcrepe.json` | rows[gamma=0.3] |
| raw_gated GQA-rel gamma=1: acc / delta / CI | 12.2 / -38.5 [-41.7, -35.2] | `results/knockouts/j7_raw_gated_gqa_rel.json` | rows[gamma=1] |
| raw_gated SugarCrepe gamma=1: acc / delta / CI | 79.5 / +1.0 [-1.6, +3.6] | `results/knockouts/j7_raw_gated_sugarcrepe.json` | rows[gamma=1] |
| raw_gated GQA-rel gamma=-1: acc / delta / CI | 46.5 / -4.2 [-7.2, -1.4] | `results/knockouts/j7_raw_gated_gqa_rel.json` | rows[gamma=-1] |
| raw_gated SugarCrepe gamma=-1: acc / delta / CI | 84.3 / +5.8 [+4.0, +7.7] | `results/knockouts/j7_raw_gated_sugarcrepe.json` | rows[gamma=-1] |
| raw_gated GQA-rel prefix absent: acc / delta / CI | 54.3 / +3.6 [+1.8, +5.4] | `results/knockouts/j7_raw_gated_gqa_rel.json` | rows[label=prefix_absent] |
| raw_gated SugarCrepe prefix absent: acc / delta / CI | 76.8 / -1.7 [-2.8, -0.6] | `results/knockouts/j7_raw_gated_sugarcrepe.json` | rows[label=prefix_absent] |

## tab_2x2

> 'on' = prefix genuinely on (s=1 for the ungated models; learned tanh gamma for the gated ones), 'off' = prefix multiplied by 0 with the slots kept. Raw/None: HyperVis no-LN checkpoint (hypertrip repo), grid boxes -- NOT the constant-prefix 34.75 of the scene-graph path (that number is in tab_prefix_scale_supp).

| quantity | value | source file | key path / derivation |
|---|---|---|---|
| hypervis_gqa_gridboxes / GQA: s=1 (on) | 50.4 | `results/knockouts/j5_hypervis_gqa_gridboxes.json` | rows[scale=1].acc |
| hypervis_gqa_gridboxes / GQA: s=0 (off) | 59.0 | `results/knockouts/j5_hypervis_gqa_gridboxes.json` | rows[scale=0].acc |
| hypervis_gqa_gridboxes / GQA: delta on-off [CI] | -8.6 [-10.5, -6.6] | `results/knockouts/j5_hypervis_gqa_gridboxes.json` | rows[scale=1].delta_vs_0_pp / ci95 |
| hypervis_gqa_gridboxes / GQA: prefix absent | 59.9 | `results/knockouts/j5_hypervis_gqa_gridboxes.json` | rows[label=prefix_absent].acc |
| hypervis_gqa_gridboxes / GQA: box field | grid | `results/knockouts/j5_hypervis_gqa_gridboxes.json` | gqa_boxes |
| hypervis_sugarcrepe / SugarCrepe: s=1 (on) | 79.9 | `results/knockouts/j5_hypervis_sugarcrepe.json` | rows[scale=1].acc |
| hypervis_sugarcrepe / SugarCrepe: s=0 (off) | 73.4 | `results/knockouts/j5_hypervis_sugarcrepe.json` | rows[scale=0].acc |
| hypervis_sugarcrepe / SugarCrepe: delta on-off [CI] | +6.5 [+5.1, +8.0] | `results/knockouts/j5_hypervis_sugarcrepe.json` | rows[scale=1].delta_vs_0_pp / ci95 |
| hypervis_sugarcrepe / SugarCrepe: prefix absent | 73.0 | `results/knockouts/j5_hypervis_sugarcrepe.json` | rows[label=prefix_absent].acc |
| hypervis_sugarcrepe / SugarCrepe: box field | repo | `results/knockouts/j5_hypervis_sugarcrepe.json` | gqa_boxes |
| raw_gated / gqa_rel: learned gamma (on) | 44.0 | `results/knockouts/j7_raw_gated_gqa_rel.json` | rows[gamma=0.00519311].acc |
| raw_gated / gqa_rel: gamma=0 (off) | 50.7 | `results/knockouts/j7_raw_gated_gqa_rel.json` | rows[gamma=0].acc |
| raw_gated / gqa_rel: delta on-off [CI] | -6.7 [-8.4, -5.0] | `results/knockouts/j7_raw_gated_gqa_rel.json` | rows[gamma=learned].delta_vs_0_pp / ci95 |
| raw_gated / gqa_rel: prefix absent | 54.3 | `results/knockouts/j7_raw_gated_gqa_rel.json` | rows[label=prefix_absent].acc |
| raw_gated / sugarcrepe: learned gamma (on) | 80.0 | `results/knockouts/j7_raw_gated_sugarcrepe.json` | rows[gamma=0.00519311].acc |
| raw_gated / sugarcrepe: gamma=0 (off) | 78.5 | `results/knockouts/j7_raw_gated_sugarcrepe.json` | rows[gamma=0].acc |
| raw_gated / sugarcrepe: delta on-off [CI] | +1.5 [+0.7, +2.4] | `results/knockouts/j7_raw_gated_sugarcrepe.json` | rows[gamma=learned].delta_vs_0_pp / ci95 |
| raw_gated / sugarcrepe: prefix absent | 76.8 | `results/knockouts/j7_raw_gated_sugarcrepe.json` | rows[label=prefix_absent].acc |
| accv_rms_ungated_gqa / GQA: s=1 (on) | 48.5 | `results/knockouts/j5_accv_rms_ungated_gqa.json` | rows[scale=1].acc |
| accv_rms_ungated_gqa / GQA: s=0 (off) | 60.8 | `results/knockouts/j5_accv_rms_ungated_gqa.json` | rows[scale=0].acc |
| accv_rms_ungated_gqa / GQA: delta on-off [CI] | -12.3 [-14.1, -10.5] | `results/knockouts/j5_accv_rms_ungated_gqa.json` | rows[scale=1].delta_vs_0_pp / ci95 |
| accv_rms_ungated_gqa / GQA: prefix absent | 61.3 | `results/knockouts/j5_accv_rms_ungated_gqa.json` | rows[label=prefix_absent].acc |
| accv_rms_ungated_gqa / GQA: box field | 6x6 grid | `results/knockouts/j5_accv_rms_ungated_gqa.json` | gqa_boxes |
| accv_rms_ungated_sugarcrepe / SugarCrepe: s=1 (on) | 85.2 | `results/knockouts/j5_accv_rms_ungated_sugarcrepe.json` | rows[scale=1].acc |
| accv_rms_ungated_sugarcrepe / SugarCrepe: s=0 (off) | 75.8 | `results/knockouts/j5_accv_rms_ungated_sugarcrepe.json` | rows[scale=0].acc |
| accv_rms_ungated_sugarcrepe / SugarCrepe: delta on-off [CI] | +9.3 [+7.8, +10.8] | `results/knockouts/j5_accv_rms_ungated_sugarcrepe.json` | rows[scale=1].delta_vs_0_pp / ci95 |
| accv_rms_ungated_sugarcrepe / SugarCrepe: prefix absent | 75.2 | `results/knockouts/j5_accv_rms_ungated_sugarcrepe.json` | rows[label=prefix_absent].acc |
| accv_rms_ungated_sugarcrepe / SugarCrepe: box field | 6x6 grid | `results/knockouts/j5_accv_rms_ungated_sugarcrepe.json` | gqa_boxes |
| base_r1 / gqa_rel: learned gamma (on) | 50.2 | `results/knockouts/j7_base_r1_gqa_rel.json` | rows[gamma=-0.153756].acc |
| base_r1 / gqa_rel: gamma=0 (off) | 49.9 | `results/knockouts/j7_base_r1_gqa_rel.json` | rows[gamma=0].acc |
| base_r1 / gqa_rel: delta on-off [CI] | +0.3 [-0.5, +1.2] | `results/knockouts/j7_base_r1_gqa_rel.json` | rows[gamma=learned].delta_vs_0_pp / ci95 |
| base_r1 / gqa_rel: prefix absent | 53.2 | `results/knockouts/j7_base_r1_gqa_rel.json` | rows[label=prefix_absent].acc |
| base_r1 / sugarcrepe: learned gamma (on) | 80.7 | `results/knockouts/j7_base_r1_sugarcrepe.json` | rows[gamma=-0.153756].acc |
| base_r1 / sugarcrepe: gamma=0 (off) | 79.4 | `results/knockouts/j7_base_r1_sugarcrepe.json` | rows[gamma=0].acc |
| base_r1 / sugarcrepe: delta on-off [CI] | +1.3 [+0.6, +2.1] | `results/knockouts/j7_base_r1_sugarcrepe.json` | rows[gamma=learned].delta_vs_0_pp / ci95 |
| base_r1 / sugarcrepe: prefix absent | 76.9 | `results/knockouts/j7_base_r1_sugarcrepe.json` | rows[label=prefix_absent].acc |
| hypervis_gqa_gridboxes: prefix token L2 at s=1 / text-token median L2 / ratio | 24.51 / 0.73 / 33.5 | `results/knockouts/j5_hypervis_gqa_gridboxes.json` | token_norms.{prefix_l2_at_1,text_l2_median,prefix_over_text_l2_ratio} |
| accv_rms_ungated_gqa: prefix token L2 at s=1 / text-token median L2 / ratio | 0.71 / 0.73 / 1.0 | `results/knockouts/j5_accv_rms_ungated_gqa.json` | token_norms.{prefix_l2_at_1,text_l2_median,prefix_over_text_l2_ratio} |

## tab_prefix_scale_supp

> GQA = first 2,000 testdev questions (file order); SC = first 2,000 SugarCrepe pairs. HyperVis 'GQA sg-path' = the repo's scene-graph proposal path, which on testdev yields an empty proposal set, i.e. an image-independent constant prefix (the published 34.75 condition); 'GQA grid' = 6x6 grid proposals. HyperVis SC uses the grid as published.

| quantity | value | source file | key path / derivation |
|---|---|---|---|
| rms_ungated GQA s=0: acc / delta / CI | 60.8 / +0.0 [+0.0, +0.0] | `results/knockouts/j5_accv_rms_ungated_gqa.json` | rows[scale=0] |
| rms_ungated SugarCrepe s=0: acc / delta / CI | 75.8 / +0.0 [+0.0, +0.0] | `results/knockouts/j5_accv_rms_ungated_sugarcrepe.json` | rows[scale=0] |
| hypervis GQA grid boxes s=0: acc / delta / CI | 59.0 / +0.0 [+0.0, +0.0] | `results/knockouts/j5_hypervis_gqa_gridboxes.json` | rows[scale=0] |
| hypervis GQA scene-graph path (empty proposals) s=0: acc / delta / CI | 59.0 / +0.0 [+0.0, +0.0] | `results/knockouts/j5_hypervis_gqa.json` | rows[scale=0] |
| hypervis SugarCrepe s=0: acc / delta / CI | 73.4 / +0.0 [+0.0, +0.0] | `results/knockouts/j5_hypervis_sugarcrepe.json` | rows[scale=0] |
| rms_ungated GQA s=0.0003: acc / delta / CI | 60.8 / -0.1 [-0.3, +0.2] | `results/knockouts/j5_accv_rms_ungated_gqa.json` | rows[scale=0.0003] |
| rms_ungated SugarCrepe s=0.0003: acc / delta / CI | 75.9 / +0.1 [-0.2, +0.4] | `results/knockouts/j5_accv_rms_ungated_sugarcrepe.json` | rows[scale=0.0003] |
| hypervis GQA grid boxes s=0.0003: acc / delta / CI | 59.1 / +0.1 [-0.4, +0.5] | `results/knockouts/j5_hypervis_gqa_gridboxes.json` | rows[scale=0.0003] |
| hypervis GQA scene-graph path (empty proposals) s=0.0003: acc / delta / CI | 59.0 / +0.0 [-0.5, +0.5] | `results/knockouts/j5_hypervis_gqa.json` | rows[scale=0.0003] |
| hypervis SugarCrepe s=0.0003: acc / delta / CI | 73.6 / +0.1 [-0.1, +0.4] | `results/knockouts/j5_hypervis_sugarcrepe.json` | rows[scale=0.0003] |
| rms_ungated GQA s=0.001: acc / delta / CI | 60.7 / -0.1 [-0.4, +0.1] | `results/knockouts/j5_accv_rms_ungated_gqa.json` | rows[scale=0.001] |
| rms_ungated SugarCrepe s=0.001: acc / delta / CI | 75.9 / +0.1 [-0.3, +0.4] | `results/knockouts/j5_accv_rms_ungated_sugarcrepe.json` | rows[scale=0.001] |
| hypervis GQA grid boxes s=0.001: acc / delta / CI | 59.0 / +0.1 [-0.4, +0.6] | `results/knockouts/j5_hypervis_gqa_gridboxes.json` | rows[scale=0.001] |
| hypervis GQA scene-graph path (empty proposals) s=0.001: acc / delta / CI | 59.1 / +0.1 [-0.4, +0.7] | `results/knockouts/j5_hypervis_gqa.json` | rows[scale=0.001] |
| hypervis SugarCrepe s=0.001: acc / delta / CI | 73.4 / +0.0 [-0.3, +0.2] | `results/knockouts/j5_hypervis_sugarcrepe.json` | rows[scale=0.001] |
| rms_ungated GQA s=0.003: acc / delta / CI | 60.8 / +0.0 [-0.3, +0.3] | `results/knockouts/j5_accv_rms_ungated_gqa.json` | rows[scale=0.003] |
| rms_ungated SugarCrepe s=0.003: acc / delta / CI | 75.6 / -0.2 [-0.6, +0.2] | `results/knockouts/j5_accv_rms_ungated_sugarcrepe.json` | rows[scale=0.003] |
| hypervis GQA grid boxes s=0.003: acc / delta / CI | 58.9 / -0.1 [-0.7, +0.4] | `results/knockouts/j5_hypervis_gqa_gridboxes.json` | rows[scale=0.003] |
| hypervis GQA scene-graph path (empty proposals) s=0.003: acc / delta / CI | 58.8 / -0.1 [-0.8, +0.4] | `results/knockouts/j5_hypervis_gqa.json` | rows[scale=0.003] |
| hypervis SugarCrepe s=0.003: acc / delta / CI | 73.3 / -0.1 [-0.4, +0.1] | `results/knockouts/j5_hypervis_sugarcrepe.json` | rows[scale=0.003] |
| rms_ungated GQA s=0.01: acc / delta / CI | 61.0 / +0.1 [-0.2, +0.5] | `results/knockouts/j5_accv_rms_ungated_gqa.json` | rows[scale=0.01] |
| rms_ungated SugarCrepe s=0.01: acc / delta / CI | 76.0 / +0.1 [-0.2, +0.5] | `results/knockouts/j5_accv_rms_ungated_sugarcrepe.json` | rows[scale=0.01] |
| hypervis GQA grid boxes s=0.01: acc / delta / CI | 59.2 / +0.2 [-0.4, +0.8] | `results/knockouts/j5_hypervis_gqa_gridboxes.json` | rows[scale=0.01] |
| hypervis GQA scene-graph path (empty proposals) s=0.01: acc / delta / CI | 58.9 / -0.1 [-0.8, +0.5] | `results/knockouts/j5_hypervis_gqa.json` | rows[scale=0.01] |
| hypervis SugarCrepe s=0.01: acc / delta / CI | 73.9 / +0.4 [+0.1, +0.9] | `results/knockouts/j5_hypervis_sugarcrepe.json` | rows[scale=0.01] |
| rms_ungated GQA s=0.03: acc / delta / CI | 60.7 / -0.1 [-0.5, +0.1] | `results/knockouts/j5_accv_rms_ungated_gqa.json` | rows[scale=0.03] |
| rms_ungated SugarCrepe s=0.03: acc / delta / CI | 76.1 / +0.2 [-0.1, +0.7] | `results/knockouts/j5_accv_rms_ungated_sugarcrepe.json` | rows[scale=0.03] |
| hypervis GQA grid boxes s=0.03: acc / delta / CI | 59.1 / +0.1 [-1.1, +1.4] | `results/knockouts/j5_hypervis_gqa_gridboxes.json` | rows[scale=0.03] |
| hypervis GQA scene-graph path (empty proposals) s=0.03: acc / delta / CI | 58.4 / -0.6 [-1.7, +0.4] | `results/knockouts/j5_hypervis_gqa.json` | rows[scale=0.03] |
| hypervis SugarCrepe s=0.03: acc / delta / CI | 75.5 / +2.1 [+1.4, +2.9] | `results/knockouts/j5_hypervis_sugarcrepe.json` | rows[scale=0.03] |
| rms_ungated GQA s=0.1: acc / delta / CI | 60.8 / +0.0 [-0.5, +0.4] | `results/knockouts/j5_accv_rms_ungated_gqa.json` | rows[scale=0.1] |
| rms_ungated SugarCrepe s=0.1: acc / delta / CI | 76.7 / +0.9 [+0.4, +1.4] | `results/knockouts/j5_accv_rms_ungated_sugarcrepe.json` | rows[scale=0.1] |
| hypervis GQA grid boxes s=0.1: acc / delta / CI | 59.0 / +0.0 [-1.5, +1.5] | `results/knockouts/j5_hypervis_gqa_gridboxes.json` | rows[scale=0.1] |
| hypervis GQA scene-graph path (empty proposals) s=0.1: acc / delta / CI | 58.9 / -0.1 [-1.6, +1.4] | `results/knockouts/j5_hypervis_gqa.json` | rows[scale=0.1] |
| hypervis SugarCrepe s=0.1: acc / delta / CI | 77.8 / +4.4 [+3.5, +5.3] | `results/knockouts/j5_hypervis_sugarcrepe.json` | rows[scale=0.1] |
| rms_ungated GQA s=0.3: acc / delta / CI | 61.5 / +0.7 [-0.2, +1.6] | `results/knockouts/j5_accv_rms_ungated_gqa.json` | rows[scale=0.3] |
| rms_ungated SugarCrepe s=0.3: acc / delta / CI | 77.5 / +1.7 [+1.1, +2.4] | `results/knockouts/j5_accv_rms_ungated_sugarcrepe.json` | rows[scale=0.3] |
| hypervis GQA grid boxes s=0.3: acc / delta / CI | 58.9 / -0.1 [-1.7, +1.5] | `results/knockouts/j5_hypervis_gqa_gridboxes.json` | rows[scale=0.3] |
| hypervis GQA scene-graph path (empty proposals) s=0.3: acc / delta / CI | 58.8 / -0.1 [-1.8, +1.5] | `results/knockouts/j5_hypervis_gqa.json` | rows[scale=0.3] |
| hypervis SugarCrepe s=0.3: acc / delta / CI | 76.5 / +3.1 [+2.1, +4.2] | `results/knockouts/j5_hypervis_sugarcrepe.json` | rows[scale=0.3] |
| rms_ungated GQA s=1: acc / delta / CI | 48.5 / -12.3 [-14.1, -10.5] | `results/knockouts/j5_accv_rms_ungated_gqa.json` | rows[scale=1] |
| rms_ungated SugarCrepe s=1: acc / delta / CI | 85.2 / +9.3 [+7.8, +10.8] | `results/knockouts/j5_accv_rms_ungated_sugarcrepe.json` | rows[scale=1] |
| hypervis GQA grid boxes s=1: acc / delta / CI | 50.4 / -8.6 [-10.5, -6.6] | `results/knockouts/j5_hypervis_gqa_gridboxes.json` | rows[scale=1] |
| hypervis GQA scene-graph path (empty proposals) s=1: acc / delta / CI | 34.7 / -24.2 [-26.5, -22.1] | `results/knockouts/j5_hypervis_gqa.json` | rows[scale=1] |
| hypervis SugarCrepe s=1: acc / delta / CI | 79.9 / +6.5 [+5.1, +8.0] | `results/knockouts/j5_hypervis_sugarcrepe.json` | rows[scale=1] |
| rms_ungated GQA s=-1: acc / delta / CI | 61.2 / +0.4 [-0.4, +1.2] | `results/knockouts/j5_accv_rms_ungated_gqa.json` | rows[scale=-1] |
| rms_ungated SugarCrepe s=-1: acc / delta / CI | 75.3 / -0.5 [-1.1, +0.1] | `results/knockouts/j5_accv_rms_ungated_sugarcrepe.json` | rows[scale=-1] |
| hypervis GQA grid boxes s=-1: acc / delta / CI | 31.4 / -27.5 [-29.9, -25.1] | `results/knockouts/j5_hypervis_gqa_gridboxes.json` | rows[scale=-1] |
| hypervis GQA scene-graph path (empty proposals) s=-1: acc / delta / CI | 34.8 / -24.2 [-26.6, -21.8] | `results/knockouts/j5_hypervis_gqa.json` | rows[scale=-1] |
| hypervis SugarCrepe s=-1: acc / delta / CI | 80.3 / +6.9 [+5.3, +8.6] | `results/knockouts/j5_hypervis_sugarcrepe.json` | rows[scale=-1] |
| rms_ungated GQA prefix absent: acc / delta / CI | 61.3 / +0.4 [-0.9, +1.7] | `results/knockouts/j5_accv_rms_ungated_gqa.json` | rows[label=prefix_absent] |
| rms_ungated GQA: gqa_boxes field | 6x6 grid | `results/knockouts/j5_accv_rms_ungated_gqa.json` | gqa_boxes |
| rms_ungated SugarCrepe prefix absent: acc / delta / CI | 75.2 / -0.7 [-1.5, +0.1] | `results/knockouts/j5_accv_rms_ungated_sugarcrepe.json` | rows[label=prefix_absent] |
| rms_ungated SugarCrepe: gqa_boxes field | 6x6 grid | `results/knockouts/j5_accv_rms_ungated_sugarcrepe.json` | gqa_boxes |
| hypervis GQA grid prefix absent: acc / delta / CI | 59.9 / +0.9 [-0.3, +2.1] | `results/knockouts/j5_hypervis_gqa_gridboxes.json` | rows[label=prefix_absent] |
| hypervis GQA grid: gqa_boxes field | grid | `results/knockouts/j5_hypervis_gqa_gridboxes.json` | gqa_boxes |
| hypervis GQA scene-graph path prefix absent: acc / delta / CI | 59.9 / +0.9 [-0.3, +2.1] | `results/knockouts/j5_hypervis_gqa.json` | rows[label=prefix_absent] |
| hypervis GQA scene-graph path: gqa_boxes field | repo | `results/knockouts/j5_hypervis_gqa.json` | gqa_boxes |
| hypervis SugarCrepe prefix absent: acc / delta / CI | 73.0 / -0.4 [-1.1, +0.4] | `results/knockouts/j5_hypervis_sugarcrepe.json` | rows[label=prefix_absent] |
| hypervis SugarCrepe: gqa_boxes field | repo | `results/knockouts/j5_hypervis_sugarcrepe.json` | gqa_boxes |

## tab_matched

> every ACT arm uses 6x6 grid proposals (the attention-bias ACT arm: eval_gqa_spatial.py hra model with the grid fallback; VSR: eval_vsr.py grid boxes).

| quantity | value | source file | key path / derivation |
|---|---|---|---|
| vanilla / rel: DROP (=vanilla) acc | 52.94 | `results/matched_mode/gqa_drop/vanilla_rel.json` | mean(correct) over all items |
| vanilla / attr: DROP (=vanilla) acc | 66.12 | `results/matched_mode/gqa_drop/vanilla_attr.json` | mean(correct) over all items |
| vanilla / vsr: DROP (=vanilla) acc | 66.21 | `results/matched_mode/vsr/vanilla/vsr_predictions.json` | mean(correct) over all items |
| rmsPub / rel: DROP acc | 53.41 | `results/matched_mode/gqa_drop/rmsPub_DROP_rel.json` | mean(correct), shared items |
| rmsPub / rel: ACT acc | 37.92 | `results/matched_mode/gqa_active/rmsPub_ACT_rel.json` | mean(correct), shared items |
| rmsPub / rel: delta DROP-ACT [CI] p (n) | +15.49 [+14.34, +16.65] p=0.0000 (n=5308) | `results/matched_mode` | paired bootstrap 10,000 (analyze_iclr_e1.paired_bootstrap, RandomState(42)) |
| rmsPub / attr: DROP acc | 66.62 | `results/matched_mode/gqa_drop/rmsPub_DROP_attr.json` | mean(correct), shared items |
| rmsPub / attr: ACT acc | 59.26 | `results/matched_mode/gqa_active/rmsPub_ACT_attr.json` | mean(correct), shared items |
| rmsPub / attr: delta DROP-ACT [CI] p (n) | +7.37 [+6.15, +8.58] p=0.0000 (n=5186) | `results/matched_mode` | paired bootstrap 10,000 (analyze_iclr_e1.paired_bootstrap, RandomState(42)) |
| rmsPub / vsr: DROP acc | 66.49 | `results/matched_mode/vsr/rmsPub_DROP/vsr_predictions.json` | mean(correct), shared items |
| rmsPub / vsr: ACT acc | 55.56 | `results/matched_mode/vsr/rmsPub_ACT/vsr_predictions.json` | mean(correct), shared items |
| rmsPub / vsr: delta DROP-ACT [CI] p (n) | +10.93 [+9.04, +12.87] p=0.0000 (n=2113) | `results/matched_mode` | paired bootstrap 10,000 (analyze_iclr_e1.paired_bootstrap, RandomState(42)) |
| rmsR2 / rel: DROP acc | 53.28 | `results/matched_mode/gqa_drop/rmsR2_DROP_rel.json` | mean(correct), shared items |
| rmsR2 / rel: ACT acc | 45.89 | `results/matched_mode/gqa_active/rmsR2_ACT_rel.json` | mean(correct), shared items |
| rmsR2 / rel: delta DROP-ACT [CI] p (n) | +7.39 [+6.35, +8.44] p=0.0000 (n=5308) | `results/matched_mode` | paired bootstrap 10,000 (analyze_iclr_e1.paired_bootstrap, RandomState(42)) |
| rmsR2 / attr: DROP acc | 66.24 | `results/matched_mode/gqa_drop/rmsR2_DROP_attr.json` | mean(correct), shared items |
| rmsR2 / attr: ACT acc | 62.36 | `results/matched_mode/gqa_active/rmsR2_ACT_attr.json` | mean(correct), shared items |
| rmsR2 / attr: delta DROP-ACT [CI] p (n) | +3.88 [+2.78, +4.99] p=0.0000 (n=5186) | `results/matched_mode` | paired bootstrap 10,000 (analyze_iclr_e1.paired_bootstrap, RandomState(42)) |
| rmsR2 / vsr: DROP acc | 68.53 | `results/matched_mode/vsr/rmsR2_DROP/vsr_predictions.json` | mean(correct), shared items |
| rmsR2 / vsr: ACT acc | 56.22 | `results/matched_mode/vsr/rmsR2_ACT/vsr_predictions.json` | mean(correct), shared items |
| rmsR2 / vsr: delta DROP-ACT [CI] p (n) | +12.30 [+10.03, +14.58] p=0.0000 (n=2113) | `results/matched_mode` | paired bootstrap 10,000 (analyze_iclr_e1.paired_bootstrap, RandomState(42)) |
| eucGeo / rel: DROP acc | 53.18 | `results/matched_mode/gqa_drop/eucGeo_DROP_rel.json` | mean(correct), shared items |
| eucGeo / rel: ACT acc | 52.35 | `results/matched_mode/gqa_active/eucGeo_ACT_rel.json` | mean(correct), shared items |
| eucGeo / rel: delta DROP-ACT [CI] p (n) | +0.83 [+0.15, +1.51] p=0.0176 (n=5308) | `results/matched_mode` | paired bootstrap 10,000 (analyze_iclr_e1.paired_bootstrap, RandomState(42)) |
| eucGeo / attr: DROP acc | 66.72 | `results/matched_mode/gqa_drop/eucGeo_DROP_attr.json` | mean(correct), shared items |
| eucGeo / attr: ACT acc | 67.95 | `results/matched_mode/gqa_active/eucGeo_ACT_attr.json` | mean(correct), shared items |
| eucGeo / attr: delta DROP-ACT [CI] p (n) | -1.23 [-2.18, -0.33] p=0.0114 (n=5186) | `results/matched_mode` | paired bootstrap 10,000 (analyze_iclr_e1.paired_bootstrap, RandomState(42)) |
| eucGeo / vsr: DROP acc | 63.80 | `results/matched_mode/vsr/eucGeo_DROP/vsr_predictions.json` | mean(correct), shared items |
| eucGeo / vsr: ACT acc | 56.55 | `results/matched_mode/vsr/eucGeo_ACT/vsr_predictions.json` | mean(correct), shared items |
| eucGeo / vsr: delta DROP-ACT [CI] p (n) | +7.24 [+5.82, +8.71] p=0.0000 (n=2113) | `results/matched_mode` | paired bootstrap 10,000 (analyze_iclr_e1.paired_bootstrap, RandomState(42)) |
| soft / rel: DROP acc | 53.03 | `results/matched_mode/gqa_drop/soft_DROP_rel.json` | mean(correct), shared items |
| soft / rel: ACT acc | 50.09 | `results/matched_mode/gqa_active/soft_ACT_rel.json` | mean(correct), shared items |
| soft / rel: delta DROP-ACT [CI] p (n) | +2.94 [+2.11, +3.79] p=0.0000 (n=5308) | `results/matched_mode` | paired bootstrap 10,000 (analyze_iclr_e1.paired_bootstrap, RandomState(42)) |
| soft / attr: DROP acc | 65.10 | `results/matched_mode/gqa_drop/soft_DROP_attr.json` | mean(correct), shared items |
| soft / attr: ACT acc | 67.18 | `results/matched_mode/gqa_active/soft_ACT_attr.json` | mean(correct), shared items |
| soft / attr: delta DROP-ACT [CI] p (n) | -2.08 [-3.18, -1.00] p=0.0002 (n=5186) | `results/matched_mode` | paired bootstrap 10,000 (analyze_iclr_e1.paired_bootstrap, RandomState(42)) |
| soft / vsr: DROP acc | 63.61 | `results/matched_mode/vsr/soft_DROP/vsr_predictions.json` | mean(correct), shared items |
| soft / vsr: ACT acc | 56.03 | `results/matched_mode/vsr/soft_ACT/vsr_predictions.json` | mean(correct), shared items |
| soft / vsr: delta DROP-ACT [CI] p (n) | +7.57 [+6.11, +9.09] p=0.0000 (n=2113) | `results/matched_mode` | paired bootstrap 10,000 (analyze_iclr_e1.paired_bootstrap, RandomState(42)) |
| euc / rel: DROP acc | 52.32 | `results/matched_mode/gqa_drop/euc_DROP_rel.json` | mean(correct), shared items |
| euc / rel: ACT acc | 49.64 | `results/matched_mode/gqa_active/euc_ACT_rel.json` | mean(correct), shared items |
| euc / rel: delta DROP-ACT [CI] p (n) | +2.68 [+1.87, +3.49] p=0.0000 (n=5308) | `results/matched_mode` | paired bootstrap 10,000 (analyze_iclr_e1.paired_bootstrap, RandomState(42)) |
| euc / attr: DROP acc | 65.35 | `results/matched_mode/gqa_drop/euc_DROP_attr.json` | mean(correct), shared items |
| euc / attr: ACT acc | 67.05 | `results/matched_mode/gqa_active/euc_ACT_attr.json` | mean(correct), shared items |
| euc / attr: delta DROP-ACT [CI] p (n) | -1.70 [-2.70, -0.71] p=0.0010 (n=5186) | `results/matched_mode` | paired bootstrap 10,000 (analyze_iclr_e1.paired_bootstrap, RandomState(42)) |
| euc / vsr: DROP acc | 62.85 | `results/matched_mode/vsr/euc_DROP/vsr_predictions.json` | mean(correct), shared items |
| euc / vsr: ACT acc | 55.99 | `results/matched_mode/vsr/euc_ACT/vsr_predictions.json` | mean(correct), shared items |
| euc / vsr: delta DROP-ACT [CI] p (n) | +6.86 [+5.44, +8.33] p=0.0000 (n=2113) | `results/matched_mode` | paired bootstrap 10,000 (analyze_iclr_e1.paired_bootstrap, RandomState(42)) |
| g2 / rel: DROP acc | 52.34 | `results/matched_mode/gqa_drop/g2_DROP_rel.json` | mean(correct), shared items |
| g2 / rel: ACT acc | 49.55 | `results/matched_mode/gqa_active/g2_ACT_rel.json` | mean(correct), shared items |
| g2 / rel: delta DROP-ACT [CI] p (n) | +2.79 [+1.94, +3.65] p=0.0000 (n=5308) | `results/matched_mode` | paired bootstrap 10,000 (analyze_iclr_e1.paired_bootstrap, RandomState(42)) |
| g2 / attr: DROP acc | 65.14 | `results/matched_mode/gqa_drop/g2_DROP_attr.json` | mean(correct), shared items |
| g2 / attr: ACT acc | 67.12 | `results/matched_mode/gqa_active/g2_ACT_attr.json` | mean(correct), shared items |
| g2 / attr: delta DROP-ACT [CI] p (n) | -1.99 [-3.07, -0.94] p=0.0004 (n=5186) | `results/matched_mode` | paired bootstrap 10,000 (analyze_iclr_e1.paired_bootstrap, RandomState(42)) |
| g2 / vsr: DROP acc | 63.37 | `results/matched_mode/vsr/g2_DROP/vsr_predictions.json` | mean(correct), shared items |
| g2 / vsr: ACT acc | 55.37 | `results/matched_mode/vsr/g2_ACT/vsr_predictions.json` | mean(correct), shared items |
| g2 / vsr: delta DROP-ACT [CI] p (n) | +8.00 [+6.53, +9.51] p=0.0000 (n=2113) | `results/matched_mode` | paired bootstrap 10,000 (analyze_iclr_e1.paired_bootstrap, RandomState(42)) |
| smallA / rel: DROP acc | 52.43 | `results/matched_mode/gqa_drop/smallA_DROP_rel.json` | mean(correct), shared items |
| smallA / rel: ACT acc | 52.56 | `results/matched_mode/gqa_active/smallA_ACT_rel.json` | mean(correct), shared items |
| smallA / rel: delta DROP-ACT [CI] p (n) | -0.13 [-0.43, +0.17] p=0.4006 (n=5308) | `results/matched_mode` | paired bootstrap 10,000 (analyze_iclr_e1.paired_bootstrap, RandomState(42)) |
| smallA / attr: DROP acc | 67.47 | `results/matched_mode/gqa_drop/smallA_DROP_attr.json` | mean(correct), shared items |
| smallA / attr: ACT acc | 67.47 | `results/matched_mode/gqa_active/smallA_ACT_attr.json` | mean(correct), shared items |
| smallA / attr: delta DROP-ACT [CI] p (n) | +0.00 [-0.35, +0.35] p=1.0000 (n=5186) | `results/matched_mode` | paired bootstrap 10,000 (analyze_iclr_e1.paired_bootstrap, RandomState(42)) |
| smallA / vsr: DROP acc | 54.47 | `results/matched_mode/vsr/smallA_DROP/vsr_predictions.json` | mean(correct), shared items |
| smallA / vsr: ACT acc | 54.52 | `results/matched_mode/vsr/smallA_ACT/vsr_predictions.json` | mean(correct), shared items |
| smallA / vsr: delta DROP-ACT [CI] p (n) | -0.05 [-0.24, +0.09] p=0.7832 (n=2113) | `results/matched_mode` | paired bootstrap 10,000 (analyze_iclr_e1.paired_bootstrap, RandomState(42)) |

## tab_matched_contrasts

| quantity | value | source file | key path / derivation |
|---|---|---|---|
| RMS r1 - vanilla / rel: delta [CI] p | +0.47 [-0.26, +1.22] p=0.2266 (accs 53.41 vs 52.94, n=5308) | `results/matched_mode` | paired bootstrap 10,000 on rmsPub_DROP vs vanilla |
| RMS r1 - vanilla / attr: delta [CI] p | +0.50 [-0.44, +1.45] p=0.3050 (accs 66.62 vs 66.12, n=5186) | `results/matched_mode` | paired bootstrap 10,000 on rmsPub_DROP vs vanilla |
| RMS r1 - vanilla / vsr: delta [CI] p | +0.28 [-0.99, +1.61] p=0.6998 (accs 66.49 vs 66.21, n=2113) | `results/matched_mode` | paired bootstrap 10,000 on rmsPub_DROP vs vanilla |
| RMS r2 - vanilla / rel: delta [CI] p | +0.34 [-0.40, +1.07] p=0.3908 (accs 53.28 vs 52.94, n=5308) | `results/matched_mode` | paired bootstrap 10,000 on rmsR2_DROP vs vanilla |
| RMS r2 - vanilla / attr: delta [CI] p | +0.12 [-0.77, +1.02] p=0.8128 (accs 66.24 vs 66.12, n=5186) | `results/matched_mode` | paired bootstrap 10,000 on rmsR2_DROP vs vanilla |
| RMS r2 - vanilla / vsr: delta [CI] p | +2.32 [+0.99, +3.64] p=0.0006 (accs 68.53 vs 66.21, n=2113) | `results/matched_mode` | paired bootstrap 10,000 on rmsR2_DROP vs vanilla |
| Euclid.\ containment - vanilla / rel: delta [CI] p | +0.24 [-0.55, +1.04] p=0.5400 (accs 53.18 vs 52.94, n=5308) | `results/matched_mode` | paired bootstrap 10,000 on eucGeo_DROP vs vanilla |
| Euclid.\ containment - vanilla / attr: delta [CI] p | +0.60 [-0.35, +1.54] p=0.2290 (accs 66.72 vs 66.12, n=5186) | `results/matched_mode` | paired bootstrap 10,000 on eucGeo_DROP vs vanilla |
| Euclid.\ containment - vanilla / vsr: delta [CI] p | -2.41 [-3.98, -0.80] p=0.0036 (accs 63.80 vs 66.21, n=2113) | `results/matched_mode` | paired bootstrap 10,000 on eucGeo_DROP vs vanilla |
| soft prompt - vanilla / rel: delta [CI] p | +0.09 [-0.77, +0.94] p=0.8388 (accs 53.03 vs 52.94, n=5308) | `results/matched_mode` | paired bootstrap 10,000 on soft_DROP vs vanilla |
| soft prompt - vanilla / attr: delta [CI] p | -1.02 [-2.12, +0.08] p=0.0704 (accs 65.10 vs 66.12, n=5186) | `results/matched_mode` | paired bootstrap 10,000 on soft_DROP vs vanilla |
| soft prompt - vanilla / vsr: delta [CI] p | -2.60 [-4.21, -0.99] p=0.0018 (accs 63.61 vs 66.21, n=2113) | `results/matched_mode` | paired bootstrap 10,000 on soft_DROP vs vanilla |
| Euclid.\ gated - vanilla / rel: delta [CI] p | -0.62 [-1.45, +0.23] p=0.1596 (accs 52.32 vs 52.94, n=5308) | `results/matched_mode` | paired bootstrap 10,000 on euc_DROP vs vanilla |
| Euclid.\ gated - vanilla / attr: delta [CI] p | -0.77 [-1.87, +0.35] p=0.1764 (accs 65.35 vs 66.12, n=5186) | `results/matched_mode` | paired bootstrap 10,000 on euc_DROP vs vanilla |
| Euclid.\ gated - vanilla / vsr: delta [CI] p | -3.36 [-5.06, -1.70] p=0.0002 (accs 62.85 vs 66.21, n=2113) | `results/matched_mode` | paired bootstrap 10,000 on euc_DROP vs vanilla |
| G2 - vanilla / rel: delta [CI] p | -0.60 [-1.45, +0.24] p=0.1808 (accs 52.34 vs 52.94, n=5308) | `results/matched_mode` | paired bootstrap 10,000 on g2_DROP vs vanilla |
| G2 - vanilla / attr: delta [CI] p | -0.98 [-2.10, +0.13] p=0.0896 (accs 65.14 vs 66.12, n=5186) | `results/matched_mode` | paired bootstrap 10,000 on g2_DROP vs vanilla |
| G2 - vanilla / vsr: delta [CI] p | -2.84 [-4.40, -1.23] p=0.0004 (accs 63.37 vs 66.21, n=2113) | `results/matched_mode` | paired bootstrap 10,000 on g2_DROP vs vanilla |
| attn.\ bias - vanilla / rel: delta [CI] p | -0.51 [-1.36, +0.36] p=0.2600 (accs 52.43 vs 52.94, n=5308) | `results/matched_mode` | paired bootstrap 10,000 on smallA_DROP vs vanilla |
| attn.\ bias - vanilla / attr: delta [CI] p | +1.35 [+0.46, +2.26] p=0.0030 (accs 67.47 vs 66.12, n=5186) | `results/matched_mode` | paired bootstrap 10,000 on smallA_DROP vs vanilla |
| attn.\ bias - vanilla / vsr: delta [CI] p | -11.74 [-13.91, -9.56] p=0.0000 (accs 54.47 vs 66.21, n=2113) | `results/matched_mode` | paired bootstrap 10,000 on smallA_DROP vs vanilla |
| Euclid.\ containment - Euclid.\ gated (flat: geo.\ loss vs.\ none) / rel: delta [CI] p | +0.87 [+0.34, +1.41] p=0.0014 (accs 53.18 vs 52.32, n=5308) | `results/matched_mode` | paired bootstrap 10,000 on eucGeo_DROP vs euc_DROP |
| Euclid.\ containment - Euclid.\ gated (flat: geo.\ loss vs.\ none) / attr: delta [CI] p | +1.37 [+0.69, +2.04] p=0.0000 (accs 66.72 vs 65.35, n=5186) | `results/matched_mode` | paired bootstrap 10,000 on eucGeo_DROP vs euc_DROP |
| Euclid.\ containment - Euclid.\ gated (flat: geo.\ loss vs.\ none) / vsr: delta [CI] p | +0.95 [+0.09, +1.80] p=0.0342 (accs 63.80 vs 62.85, n=2113) | `results/matched_mode` | paired bootstrap 10,000 on eucGeo_DROP vs euc_DROP |
| RMS r2 - Euclid.\ containment (hyp.\ vs.\ flat) / rel: delta [CI] p | +0.09 [-0.60, +0.77] p=0.8358 (accs 53.28 vs 53.18, n=5308) | `results/matched_mode` | paired bootstrap 10,000 on rmsR2_DROP vs eucGeo_DROP |
| RMS r2 - Euclid.\ containment (hyp.\ vs.\ flat) / attr: delta [CI] p | -0.48 [-1.45, +0.48] p=0.3426 (accs 66.24 vs 66.72, n=5186) | `results/matched_mode` | paired bootstrap 10,000 on rmsR2_DROP vs eucGeo_DROP |
| RMS r2 - Euclid.\ containment (hyp.\ vs.\ flat) / vsr: delta [CI] p | +4.73 [+2.93, +6.44] p=0.0000 (accs 68.53 vs 63.80, n=2113) | `results/matched_mode` | paired bootstrap 10,000 on rmsR2_DROP vs eucGeo_DROP |
| RMS r1 - Euclid.\ containment (hyp.\ vs.\ flat) / rel: delta [CI] p | +0.23 [-0.38, +0.83] p=0.4920 (accs 53.41 vs 53.18, n=5308) | `results/matched_mode` | paired bootstrap 10,000 on rmsPub_DROP vs eucGeo_DROP |
| RMS r1 - Euclid.\ containment (hyp.\ vs.\ flat) / attr: delta [CI] p | -0.10 [-0.93, +0.73] p=0.8444 (accs 66.62 vs 66.72, n=5186) | `results/matched_mode` | paired bootstrap 10,000 on rmsPub_DROP vs eucGeo_DROP |
| RMS r1 - Euclid.\ containment (hyp.\ vs.\ flat) / vsr: delta [CI] p | +2.70 [+1.51, +3.88] p=0.0000 (accs 66.49 vs 63.80, n=2113) | `results/matched_mode` | paired bootstrap 10,000 on rmsPub_DROP vs eucGeo_DROP |
| RMS r2 - soft prompt / rel: delta [CI] p | +0.24 [-0.47, +0.96] p=0.5308 (accs 53.28 vs 53.03, n=5308) | `results/matched_mode` | paired bootstrap 10,000 on rmsR2_DROP vs soft_DROP |
| RMS r2 - soft prompt / attr: delta [CI] p | +1.14 [+0.08, +2.22] p=0.0342 (accs 66.24 vs 65.10, n=5186) | `results/matched_mode` | paired bootstrap 10,000 on rmsR2_DROP vs soft_DROP |
| RMS r2 - soft prompt / vsr: delta [CI] p | +4.92 [+3.12, +6.67] p=0.0000 (accs 68.53 vs 63.61, n=2113) | `results/matched_mode` | paired bootstrap 10,000 on rmsR2_DROP vs soft_DROP |

## tab_vsr_seeds_supp

| quantity | value | source file | key path / derivation |
|---|---|---|---|
| VSR DROP hyperbolic rms_ungated r1 | 66.49 | `results/matched_mode/vsr/rmsPub_DROP/vsr_predictions.json` | mean(correct) |
| VSR DROP hyperbolic rms_ungated r2 | 68.53 | `results/matched_mode/vsr/rmsR2_DROP/vsr_predictions.json` | mean(correct) |
| VSR DROP hyperbolic rms_ungated r3 | 68.53 | `results/matched_mode/vsr/rmsR3_DROP/vsr_predictions.json` | mean(correct) |
| VSR DROP flat euclidean_geo r1 | 63.80 | `results/matched_mode/vsr/eucGeo_DROP/vsr_predictions.json` | mean(correct) |
| VSR DROP flat euclidean_geo s2 | 66.82 | `results/matched_mode/vsr/eucGeoS2_DROP/vsr_predictions.json` | mean(correct) |
| VSR DROP flat euclidean_geo s3 | 65.45 | `results/matched_mode/vsr/eucGeoS3_DROP/vsr_predictions.json` | mean(correct) |
| VSR vanilla | 66.21 | `results/matched_mode/vsr/vanilla/vsr_predictions.json` | mean(correct) |
| hyp mean | 67.85 | `results/matched_mode` | numpy mean / std(ddof=1) over the 3 seed accuracies |
| hyp sd | 1.17 | `results/matched_mode` | numpy mean / std(ddof=1) over the 3 seed accuracies |
| flat mean | 65.36 | `results/matched_mode` | numpy mean / std(ddof=1) over the 3 seed accuracies |
| flat sd | 1.52 | `results/matched_mode` | numpy mean / std(ddof=1) over the 3 seed accuracies |
| hyp - flat: delta, Welch t, df, p | +2.49, t=2.25, df=3.8, p=0.092 | `results/matched_mode` | Welch two-sample t (scipy.stats.t.sf) |
| hyp - vanilla: delta, one-sample t, df, p | +1.64, t=2.42, df=2, p=0.137 | `results/matched_mode` | one-sample t of the 3 hyp seeds against the vanilla point value |
| flat - vanilla: delta, one-sample t, df, p | -0.85, t=-0.97, df=2, p=0.433 | `results/matched_mode` | one-sample t of the 3 flat seeds against the vanilla point value |

## tab_backbones

| quantity | value | source file | key path / derivation |
|---|---|---|---|
| LLaVA final gate hra_v1 (beta) | -0.6958 | `results/knockouts/j7_hra_v1_gqa_rel.json` | learned_gate |
| LLaVA final gate small_alpha (beta) | -0.6005 | `results/knockouts/j7_small_alpha_gqa_rel.json` | learned_gate |
| LLaVA final gate stage2 (beta) | -8.4962 | `results/knockouts/j7_stage2_gqa_rel.json` | learned_gate |
| LLaVA final gate no_dropout (tanh gamma) | -0.3623 | `results/knockouts/j7_no_dropout_gqa_rel.json` | learned_gate |
| LLaVA final gate raw_gated (tanh gamma) | +0.0052 | `results/knockouts/j7_raw_gated_gqa_rel.json` | learned_gate |
| LLaVA final gate base_r1 (tanh gamma) | -0.1538 | `results/knockouts/j7_base_r1_gqa_rel.json` | learned_gate |
| LLaVA final gate soft_prompt (tanh gamma) | -0.1963 | `results/knockouts/j7_soft_prompt_gqa_rel.json` | learned_gate |
| LLaVA final gate euclidean (tanh gamma) | +0.1849 | `results/knockouts/j7_euclidean_gqa_rel.json` | learned_gate |
| LLaVA final gate synthesis (tanh gamma) | +0.1186 | `results/knockouts/j7_synthesis_gqa_rel.json` | learned_gate |
| LLaVA final gate lora_all_linear_r1 (tanh gamma) | +0.0235 | `results/knockouts/j7_lora_all_linear_r1_gqa_rel.json` | learned_gate |
| LLaVA final gate lora_all_linear_s2 (tanh gamma) | -0.0300 | `results/knockouts/j7_lora_all_linear_s2_gqa_rel.json` | learned_gate |
| LLaVA hra_v1 GQA-rel bias x3 delta [CI] | +1.0 [-0.1, +2.1] | `results/knockouts/j7_hra_v1_gqa_rel.json` | rows[gamma=3].delta_vs_0_pp / ci95 |
| LLaVA small_alpha GQA-rel bias x3 delta [CI] | +0.1 [-0.8, +1.0] | `results/knockouts/j7_small_alpha_gqa_rel.json` | rows[gamma=3].delta_vs_0_pp / ci95 |
| LLaVA stage2 GQA-rel bias x3 delta [CI] | +0.0 [-0.5, +0.5] | `results/knockouts/j7_stage2_gqa_rel.json` | rows[gamma=3].delta_vs_0_pp / ci95 |
| LLaVA base_r1 GQA-rel gamma=1 delta [CI] | +3.4 [+1.4, +5.3] | `results/knockouts/j7_base_r1_gqa_rel.json` | rows[gamma=1].delta_vs_0_pp / ci95 |
| LLaVA soft_prompt GQA-rel gamma=1 delta [CI] | +1.2 [-0.7, +3.0] | `results/knockouts/j7_soft_prompt_gqa_rel.json` | rows[gamma=1].delta_vs_0_pp / ci95 |
| LLaVA no_dropout GQA-rel gamma=1 delta [CI] | +2.6 [+0.7, +4.4] | `results/knockouts/j7_no_dropout_gqa_rel.json` | rows[gamma=1].delta_vs_0_pp / ci95 |
| LLaVA euclidean GQA-rel gamma=1 delta [CI] | -0.3 [-1.7, +1.2] | `results/knockouts/j7_euclidean_gqa_rel.json` | rows[gamma=1].delta_vs_0_pp / ci95 |
| LLaVA synthesis GQA-rel gamma=1 delta [CI] | +0.3 [-1.1, +1.7] | `results/knockouts/j7_synthesis_gqa_rel.json` | rows[gamma=1].delta_vs_0_pp / ci95 |
| LLaVA lora_all_linear_r1 GQA-rel gamma=1 delta [CI] | +0.6 [-0.6, +1.9] | `results/knockouts/j7_lora_all_linear_r1_gqa_rel.json` | rows[gamma=1].delta_vs_0_pp / ci95 |
| LLaVA lora_all_linear_s2 GQA-rel gamma=1 delta [CI] | -0.7 [-1.9, +0.5] | `results/knockouts/j7_lora_all_linear_s2_gqa_rel.json` | rows[gamma=1].delta_vs_0_pp / ci95 |
| LLaVA raw_gated GQA-rel gamma=1 delta [CI] | -38.5 [-41.7, -35.2] | `results/knockouts/j7_raw_gated_gqa_rel.json` | rows[gamma=1] |
| LLaVA raw_gated GQA-rel learned gamma delta [CI] | -6.7 [-8.4, -5.0] | `results/knockouts/j7_raw_gated_gqa_rel.json` | rows[gamma=learned] |
| Qwen2-VL-7B final tanh gate g1b seed0 | +0.000633 | `results/external/gate-objective/g1b/seed0/FINAL.json` | final_gate_tanh |
| Qwen2-VL-7B final tanh gate g1b seed1 | -0.001018 | `results/external/gate-objective/g1b/seed1/FINAL.json` | final_gate_tanh |
| Qwen2-VL-7B final tanh gate g1b seed2 | -0.000970 | `results/external/gate-objective/g1b/seed2/FINAL.json` | final_gate_tanh |
| Qwen2-VL-7B final tanh gate g1b_count seed0 | +0.001423 | `results/external/gate-objective/g1b_count/seed0/FINAL.json` | final_gate_tanh |
| Qwen2-VL-7B final tanh gate g1b_count seed1 | +0.002044 | `results/external/gate-objective/g1b_count/seed1/FINAL.json` | final_gate_tanh |
| Qwen2-VL-7B final tanh gate g1b_count seed2 | -0.001212 | `results/external/gate-objective/g1b_count/seed2/FINAL.json` | final_gate_tanh |
| Qwen2-VL-7B final tanh gate g1b_attribute seed0 | +0.001667 | `results/external/gate-objective/g1b_attribute/seed0/FINAL.json` | final_gate_tanh |
| Qwen2-VL-7B final tanh gate g1b_attribute seed1 | -0.005122 | `results/external/gate-objective/g1b_attribute/seed1/FINAL.json` | final_gate_tanh |
| Qwen2-VL-7B final tanh gate g1b_attribute seed2 | -0.003925 | `results/external/gate-objective/g1b_attribute/seed2/FINAL.json` | final_gate_tanh |
| Qwen2-VL-7B final tanh gate g1b_presence seed0 | +0.005464 | `results/external/gate-objective/g1b_presence/seed0/FINAL.json` | final_gate_tanh |
| Qwen2-VL-7B final tanh gate g1b_presence seed1 | -0.004377 | `results/external/gate-objective/g1b_presence/seed1/FINAL.json` | final_gate_tanh |
| Qwen2-VL-7B final tanh gate g1b_presence seed2 | -0.001921 | `results/external/gate-objective/g1b_presence/seed2/FINAL.json` | final_gate_tanh |
| Qwen2-VL-7B final tanh gate g1b_detrap seed0 | -0.014394 | `results/external/gate-objective/g1b_detrap/seed0/FINAL.json` | final_gate_tanh |
| Qwen2-VL-7B final tanh gate g1b_detrap seed1 | +0.017545 | `results/external/gate-objective/g1b_detrap/seed1/FINAL.json` | final_gate_tanh |
| Qwen2-VL-7B final tanh gate g1b_detrap seed2 | -0.011806 | `results/external/gate-objective/g1b_detrap/seed2/FINAL.json` | final_gate_tanh |
| Qwen2-VL-7B max \|tanh gate\| over runs | 0.0175 (15 runs) | `results/external/gate-objective` | max \|final_gate_tanh\| |
| Qwen2-VL-7B forced gamma=0 GQA-rel (n=5,308): acc / delta / CI | 57.76 / +0.00 [+0.00, +0.00] | `results/external/gate-objective/gsweep/w4aud_g1b0_gqa.json` | rows[gamma=0] |
| Qwen2-VL-7B forced gamma=1.5 GQA-rel (n=5,308): acc / delta / CI | 57.93 / +0.17 [-0.21, +0.53] | `results/external/gate-objective/gsweep/w4aud_g1b0_gqa.json` | rows[gamma=1.5] |
| Qwen2-VL-7B forced gamma=2 GQA-rel (n=5,308): acc / delta / CI | 57.91 / +0.15 [-0.24, +0.55] | `results/external/gate-objective/gsweep/w4aud_g1b0_gqa.json` | rows[gamma=2] |
| InternVL3-8B final tanh gate seed0 | +0.006160 | `results/probes/internvl_seed0.json` | final_gate_tanh |
| InternVL3-8B seed0 forced gamma=1: d_CE [CI] (n=300) | -0.0085 [-0.0274, +0.0086] (CE 0.7768 vs 0.7853 at 0) | `results/probes/internvl_knockout.json` | gammas['1.0'].d_ce / ci_lo / ci_hi |
| InternVL3-8B seed0 forced gamma=2: d_CE [CI] (n=300) | -0.0075 [-0.0257, +0.0094] (CE 0.7778 vs 0.7853 at 0) | `results/probes/internvl_knockout.json` | gammas['2.0'].d_ce / ci_lo / ci_hi |
| InternVL3-8B final tanh gate seed1 | -0.001476 | `results/probes/internvl_seed1.json` | final_gate_tanh |
| InternVL3-8B seed1 forced gamma=1: d_CE [CI] (n=300) | +0.0111 [-0.0003, +0.0241] (CE 0.7611 vs 0.7501 at 0) | `results/probes/internvl_knockout_seed1.json` | gammas['1.0'].d_ce / ci_lo / ci_hi |
| InternVL3-8B seed1 forced gamma=2: d_CE [CI] (n=300) | +0.0142 [+0.0009, +0.0304] (CE 0.7643 vs 0.7501 at 0) | `results/probes/internvl_knockout_seed1.json` | gammas['2.0'].d_ce / ci_lo / ci_hi |
| InternVL3-8B final tanh gate seed2 | -0.004375 | `results/probes/internvl_seed2.json` | final_gate_tanh |
| InternVL3-8B seed2 forced gamma=1: d_CE [CI] (n=300) | +0.0013 [-0.0126, +0.0147] (CE 0.7931 vs 0.7918 at 0) | `results/probes/internvl_knockout_seed2.json` | gammas['1.0'].d_ce / ci_lo / ci_hi |
| InternVL3-8B seed2 forced gamma=2: d_CE [CI] (n=300) | +0.0014 [-0.0109, +0.0134] (CE 0.7932 vs 0.7918 at 0) | `results/probes/internvl_knockout_seed2.json` | gammas['2.0'].d_ce / ci_lo / ci_hi |
| Qwen runs with \|tanh gate\| <= 0.006 | 12 of 15 | `results/external/gate-objective` | count over FINAL.json |

## tab_gradient_supp

| quantity | value | source file | key path / derivation |
|---|---|---|---|
| grad_pair_discriminative: mean / se / \|t\| | +0.6743 / 3.5124 / 0.19 | `results/probes/gate_gradient.json` | grad_pair_discriminative.{mean,se,abs_t} |
| grad_task_single_caption: mean / se / \|t\| | -15.8935 / 6.5713 / 2.42 | `results/probes/gate_gradient.json` | grad_task_single_caption.{mean,se,abs_t} |
| curvature_task: mean / se / \|t\| | -18.3684 / 1208.4609 / 0.02 | `results/probes/gate_gradient.json` | curvature_task.{mean,se,abs_t} |
| n / eps / variant | 120 / 0.01 / rms_ungated | `results/probes/gate_gradient.json` | n, eps, variant |

## tab_attention_supp

> x chance = mean over layers of the item-mean attention mass from text queries onto the K=4 injected slots, divided by K/L. J6 gamma=0 keeps the 4 zero-vector slots: the mass they attract with the pathway closed is the attention-sink component (same magnitude as at the landed gate), so attention mass is not evidence of use.

| quantity | value | source file | key path / derivation |
|---|---|---|---|
| layer-profile probe rms_ungated / learned: mass % at L0,8,16,24,31 | 1.73, 17.46, 14.12, 9.05, 7.66 | `results/probes/layerwise_attention.json` | mass_per_layer[0,8,16,24,31] |
| layer-profile probe rms_ungated / learned: mean mass % / chance % / x chance | 13.45 / 0.65 / 20.6 | `results/probes/layerwise_attention.json` | mean(mass_per_layer) / chance |
| layer-profile probe rms_ungated / open: mass % at L0,8,16,24,31 | 1.73, 17.46, 14.12, 9.06, 7.64 | `results/probes/layerwise_attention.json` | mass_per_layer[0,8,16,24,31] |
| layer-profile probe rms_ungated / open: mean mass % / chance % / x chance | 13.46 / 0.65 / 20.6 | `results/probes/layerwise_attention.json` | mean(mass_per_layer) / chance |
| layer-profile probe soft_prompt / learned: mass % at L0,8,16,24,31 | 1.49, 28.13, 26.41, 19.15, 3.96 | `results/probes/layerwise_attention.json` | mass_per_layer[0,8,16,24,31] |
| layer-profile probe soft_prompt / learned: mean mass % / chance % / x chance | 24.62 / 0.65 / 37.7 | `results/probes/layerwise_attention.json` | mean(mass_per_layer) / chance |
| layer-profile probe soft_prompt / open: mass % at L0,8,16,24,31 | 0.96, 1.04, 1.27, 1.44, 2.83 | `results/probes/layerwise_attention.json` | mass_per_layer[0,8,16,24,31] |
| layer-profile probe soft_prompt / open: mean mass % / chance % / x chance | 1.37 / 0.65 / 2.1 | `results/probes/layerwise_attention.json` | mean(mass_per_layer) / chance |
| layer-profile probe raw_gated / learned: mass % at L0,8,16,24,31 | 1.47, 27.81, 25.61, 18.36, 4.50 | `results/probes/layerwise_attention.json` | mass_per_layer[0,8,16,24,31] |
| layer-profile probe raw_gated / learned: mean mass % / chance % / x chance | 23.98 / 0.65 / 36.7 | `results/probes/layerwise_attention.json` | mean(mass_per_layer) / chance |
| layer-profile probe raw_gated / open: mass % at L0,8,16,24,31 | 1.52, 5.89, 6.96, 3.89, 2.69 | `results/probes/layerwise_attention.json` | mass_per_layer[0,8,16,24,31] |
| layer-profile probe raw_gated / open: mean mass % / chance % / x chance | 7.25 / 0.65 / 11.1 | `results/probes/layerwise_attention.json` | mean(mass_per_layer) / chance |
| J6 base_r1 / 0 (value +0.0000): x chance (mean over layers) [CI] | 37.54 [37.17, 37.90] | `results/probes/j6/j6_llava_base_r1.json` | by_gamma['0'].x_chance_mean_over_layers / mean_layer_ratio_all_ci95 |
| J6 base_r1 / 0: mass % L0,8,16,24,31 / mean | 1.19, 27.98, 26.24, 19.13, 3.78 / 24.53 | `results/probes/j6/j6_llava_base_r1.json` | by_gamma['0'].mass_L0_L8_L16_L24_L31 / mean_mass_all_over_layers |
| J6 base_r1 / landed (value -0.1538): x chance (mean over layers) [CI] | 39.91 [39.53, 40.29] | `results/probes/j6/j6_llava_base_r1.json` | by_gamma['landed'].x_chance_mean_over_layers / mean_layer_ratio_all_ci95 |
| J6 base_r1 / landed: mass % L0,8,16,24,31 / mean | 1.40, 29.60, 27.88, 20.79, 4.40 / 26.08 | `results/probes/j6/j6_llava_base_r1.json` | by_gamma['landed'].mass_L0_L8_L16_L24_L31 / mean_mass_all_over_layers |
| J6 base_r1 / 1 (value +1.0000): x chance (mean over layers) [CI] | 2.15 [2.10, 2.20] | `results/probes/j6/j6_llava_base_r1.json` | by_gamma['1'].x_chance_mean_over_layers / mean_layer_ratio_all_ci95 |
| J6 base_r1 / 1: mass % L0,8,16,24,31 / mean | 0.96, 1.54, 1.53, 1.67, 2.75 / 1.41 | `results/probes/j6/j6_llava_base_r1.json` | by_gamma['1'].mass_L0_L8_L16_L24_L31 / mean_mass_all_over_layers |
| J6 base_r1: chance (K/L) | 0.653 % | `results/probes/j6/j6_llava_base_r1.json` | by_gamma['0'].chance_all |
| J6 soft_prompt / 0 (value +0.0000): x chance (mean over layers) [CI] | 36.77 [36.39, 37.12] | `results/probes/j6/j6_llava_soft_prompt.json` | by_gamma['0'].x_chance_mean_over_layers / mean_layer_ratio_all_ci95 |
| J6 soft_prompt / 0: mass % L0,8,16,24,31 / mean | 1.19, 27.42, 25.75, 18.52, 3.52 / 24.02 | `results/probes/j6/j6_llava_soft_prompt.json` | by_gamma['0'].mass_L0_L8_L16_L24_L31 / mean_mass_all_over_layers |
| J6 soft_prompt / landed (value -0.1963): x chance (mean over layers) [CI] | 37.69 [37.34, 38.01] | `results/probes/j6/j6_llava_soft_prompt.json` | by_gamma['landed'].x_chance_mean_over_layers / mean_layer_ratio_all_ci95 |
| J6 soft_prompt / landed: mass % L0,8,16,24,31 / mean | 1.49, 28.13, 26.41, 19.15, 3.96 / 24.62 | `results/probes/j6/j6_llava_soft_prompt.json` | by_gamma['landed'].mass_L0_L8_L16_L24_L31 / mean_mass_all_over_layers |
| J6 soft_prompt / 1 (value +1.0000): x chance (mean over layers) [CI] | 2.10 [2.06, 2.15] | `results/probes/j6/j6_llava_soft_prompt.json` | by_gamma['1'].x_chance_mean_over_layers / mean_layer_ratio_all_ci95 |
| J6 soft_prompt / 1: mass % L0,8,16,24,31 / mean | 0.96, 1.04, 1.27, 1.45, 2.83 / 1.37 | `results/probes/j6/j6_llava_soft_prompt.json` | by_gamma['1'].mass_L0_L8_L16_L24_L31 / mean_mass_all_over_layers |
| J6 soft_prompt: chance (K/L) | 0.653 % | `results/probes/j6/j6_llava_soft_prompt.json` | by_gamma['0'].chance_all |
| J6 no_dropout / 0 (value +0.0000): x chance (mean over layers) [CI] | 33.53 [33.08, 33.96] | `results/probes/j6/j6_llava_no_dropout.json` | by_gamma['0'].x_chance_mean_over_layers / mean_layer_ratio_all_ci95 |
| J6 no_dropout / 0: mass % L0,8,16,24,31 / mean | 1.19, 25.48, 23.46, 16.21, 4.36 / 21.91 | `results/probes/j6/j6_llava_no_dropout.json` | by_gamma['0'].mass_L0_L8_L16_L24_L31 / mean_mass_all_over_layers |
| J6 no_dropout / landed (value -0.3623): x chance (mean over layers) [CI] | 39.97 [39.58, 40.34] | `results/probes/j6/j6_llava_no_dropout.json` | by_gamma['landed'].x_chance_mean_over_layers / mean_layer_ratio_all_ci95 |
| J6 no_dropout / landed: mass % L0,8,16,24,31 / mean | 1.99, 30.36, 27.99, 20.03, 5.84 / 26.11 | `results/probes/j6/j6_llava_no_dropout.json` | by_gamma['landed'].mass_L0_L8_L16_L24_L31 / mean_mass_all_over_layers |
| J6 no_dropout / 1 (value +1.0000): x chance (mean over layers) [CI] | 1.66 [1.62, 1.70] | `results/probes/j6/j6_llava_no_dropout.json` | by_gamma['1'].x_chance_mean_over_layers / mean_layer_ratio_all_ci95 |
| J6 no_dropout / 1: mass % L0,8,16,24,31 / mean | 0.82, 1.62, 0.83, 0.79, 1.78 / 1.08 | `results/probes/j6/j6_llava_no_dropout.json` | by_gamma['1'].mass_L0_L8_L16_L24_L31 / mean_mass_all_over_layers |
| J6 no_dropout: chance (K/L) | 0.653 % | `results/probes/j6/j6_llava_no_dropout.json` | by_gamma['0'].chance_all |
| J6 euclidean / 0 (value +0.0000): x chance (mean over layers) [CI] | 36.06 [35.66, 36.44] | `results/probes/j6/j6_llava_euclidean.json` | by_gamma['0'].x_chance_mean_over_layers / mean_layer_ratio_all_ci95 |
| J6 euclidean / 0: mass % L0,8,16,24,31 / mean | 1.19, 27.22, 25.30, 17.85, 3.63 / 23.56 | `results/probes/j6/j6_llava_euclidean.json` | by_gamma['0'].mass_L0_L8_L16_L24_L31 / mean_mass_all_over_layers |
| J6 euclidean / landed (value +0.1849): x chance (mean over layers) [CI] | 35.86 [35.46, 36.24] | `results/probes/j6/j6_llava_euclidean.json` | by_gamma['landed'].x_chance_mean_over_layers / mean_layer_ratio_all_ci95 |
| J6 euclidean / landed: mass % L0,8,16,24,31 / mean | 0.92, 27.12, 25.15, 17.85, 3.24 / 23.43 | `results/probes/j6/j6_llava_euclidean.json` | by_gamma['landed'].mass_L0_L8_L16_L24_L31 / mean_mass_all_over_layers |
| J6 euclidean / 1 (value +1.0000): x chance (mean over layers) [CI] | 9.64 [9.06, 10.20] | `results/probes/j6/j6_llava_euclidean.json` | by_gamma['1'].x_chance_mean_over_layers / mean_layer_ratio_all_ci95 |
| J6 euclidean / 1: mass % L0,8,16,24,31 / mean | 0.78, 8.52, 6.92, 2.55, 2.88 / 6.30 | `results/probes/j6/j6_llava_euclidean.json` | by_gamma['1'].mass_L0_L8_L16_L24_L31 / mean_mass_all_over_layers |
| J6 euclidean: chance (K/L) | 0.653 % | `results/probes/j6/j6_llava_euclidean.json` | by_gamma['0'].chance_all |
| J6 synthesis / 0 (value +0.0000): x chance (mean over layers) [CI] | 35.54 [35.12, 35.92] | `results/probes/j6/j6_llava_synthesis.json` | by_gamma['0'].x_chance_mean_over_layers / mean_layer_ratio_all_ci95 |
| J6 synthesis / 0: mass % L0,8,16,24,31 / mean | 1.19, 26.13, 24.80, 18.06, 4.06 / 23.21 | `results/probes/j6/j6_llava_synthesis.json` | by_gamma['0'].mass_L0_L8_L16_L24_L31 / mean_mass_all_over_layers |
| J6 synthesis / landed (value +0.1186): x chance (mean over layers) [CI] | 36.68 [36.27, 37.08] | `results/probes/j6/j6_llava_synthesis.json` | by_gamma['landed'].x_chance_mean_over_layers / mean_layer_ratio_all_ci95 |
| J6 synthesis / landed: mass % L0,8,16,24,31 / mean | 1.25, 27.00, 25.54, 18.78, 4.12 / 23.97 | `results/probes/j6/j6_llava_synthesis.json` | by_gamma['landed'].mass_L0_L8_L16_L24_L31 / mean_mass_all_over_layers |
| J6 synthesis / 1 (value +1.0000): x chance (mean over layers) [CI] | 25.49 [24.84, 26.10] | `results/probes/j6/j6_llava_synthesis.json` | by_gamma['1'].x_chance_mean_over_layers / mean_layer_ratio_all_ci95 |
| J6 synthesis / 1: mass % L0,8,16,24,31 / mean | 1.43, 19.99, 18.90, 11.23, 6.57 / 16.65 | `results/probes/j6/j6_llava_synthesis.json` | by_gamma['1'].mass_L0_L8_L16_L24_L31 / mean_mass_all_over_layers |
| J6 synthesis: chance (K/L) | 0.653 % | `results/probes/j6/j6_llava_synthesis.json` | by_gamma['0'].chance_all |
| J6 raw_gated / 0 (value +0.0000): x chance (mean over layers) [CI] | 35.48 [35.06, 35.86] | `results/probes/j6/j6_llava_raw_gated.json` | by_gamma['0'].x_chance_mean_over_layers / mean_layer_ratio_all_ci95 |
| J6 raw_gated / 0: mass % L0,8,16,24,31 / mean | 1.19, 26.81, 24.87, 17.43, 3.98 / 23.18 | `results/probes/j6/j6_llava_raw_gated.json` | by_gamma['0'].mass_L0_L8_L16_L24_L31 / mean_mass_all_over_layers |
| J6 raw_gated / landed (value +0.0052): x chance (mean over layers) [CI] | 36.70 [36.31, 37.06] | `results/probes/j6/j6_llava_raw_gated.json` | by_gamma['landed'].x_chance_mean_over_layers / mean_layer_ratio_all_ci95 |
| J6 raw_gated / landed: mass % L0,8,16,24,31 / mean | 1.47, 27.80, 25.61, 18.36, 4.50 / 23.97 | `results/probes/j6/j6_llava_raw_gated.json` | by_gamma['landed'].mass_L0_L8_L16_L24_L31 / mean_mass_all_over_layers |
| J6 raw_gated / 1 (value +1.0000): x chance (mean over layers) [CI] | 11.11 [11.02, 11.21] | `results/probes/j6/j6_llava_raw_gated.json` | by_gamma['1'].x_chance_mean_over_layers / mean_layer_ratio_all_ci95 |
| J6 raw_gated / 1: mass % L0,8,16,24,31 / mean | 1.52, 5.90, 6.95, 3.90, 2.69 / 7.26 | `results/probes/j6/j6_llava_raw_gated.json` | by_gamma['1'].mass_L0_L8_L16_L24_L31 / mean_mass_all_over_layers |
| J6 raw_gated: chance (K/L) | 0.653 % | `results/probes/j6/j6_llava_raw_gated.json` | by_gamma['0'].chance_all |
| J6 rms_ungated / 0 (value +0.0000): x chance (mean over layers) [CI] | 13.38 [12.97, 13.76] | `results/probes/j6/j6_llava_rms_ungated.json` | by_gamma['0'].x_chance_mean_over_layers / mean_layer_ratio_all_ci95 |
| J6 rms_ungated / 0: mass % L0,8,16,24,31 / mean | 1.19, 11.93, 8.93, 4.25, 2.65 / 8.74 | `results/probes/j6/j6_llava_rms_ungated.json` | by_gamma['0'].mass_L0_L8_L16_L24_L31 / mean_mass_all_over_layers |
| J6 rms_ungated / landed (value +1.0000): x chance (mean over layers) [CI] | 20.59 [20.20, 20.99] | `results/probes/j6/j6_llava_rms_ungated.json` | by_gamma['landed'].x_chance_mean_over_layers / mean_layer_ratio_all_ci95 |
| J6 rms_ungated / landed: mass % L0,8,16,24,31 / mean | 1.73, 17.46, 14.12, 9.05, 7.66 / 13.45 | `results/probes/j6/j6_llava_rms_ungated.json` | by_gamma['landed'].mass_L0_L8_L16_L24_L31 / mean_mass_all_over_layers |
| J6 rms_ungated / 1 (value +1.0000): x chance (mean over layers) [CI] | 20.59 [20.20, 20.99] | `results/probes/j6/j6_llava_rms_ungated.json` | by_gamma['1'].x_chance_mean_over_layers / mean_layer_ratio_all_ci95 |
| J6 rms_ungated / 1: mass % L0,8,16,24,31 / mean | 1.73, 17.46, 14.12, 9.05, 7.66 / 13.45 | `results/probes/j6/j6_llava_rms_ungated.json` | by_gamma['1'].mass_L0_L8_L16_L24_L31 / mean_mass_all_over_layers |
| J6 rms_ungated: chance (K/L) | 0.653 % | `results/probes/j6/j6_llava_rms_ungated.json` | by_gamma['0'].chance_all |
| J6 lora_all_linear_r1 / 0 (value +0.0000): x chance (mean over layers) [CI] | 12.49 [12.13, 12.83] | `results/probes/j6/j6_llava_lora_all_linear_r1.json` | by_gamma['0'].x_chance_mean_over_layers / mean_layer_ratio_all_ci95 |
| J6 lora_all_linear_r1 / 0: mass % L0,8,16,24,31 / mean | 1.19, 10.67, 8.42, 4.09, 3.53 / 8.16 | `results/probes/j6/j6_llava_lora_all_linear_r1.json` | by_gamma['0'].mass_L0_L8_L16_L24_L31 / mean_mass_all_over_layers |
| J6 lora_all_linear_r1 / landed (value +0.0235): x chance (mean over layers) [CI] | 12.37 [11.96, 12.77] | `results/probes/j6/j6_llava_lora_all_linear_r1.json` | by_gamma['landed'].x_chance_mean_over_layers / mean_layer_ratio_all_ci95 |
| J6 lora_all_linear_r1 / landed: mass % L0,8,16,24,31 / mean | 1.19, 10.53, 8.12, 3.88, 3.58 / 8.08 | `results/probes/j6/j6_llava_lora_all_linear_r1.json` | by_gamma['landed'].mass_L0_L8_L16_L24_L31 / mean_mass_all_over_layers |
| J6 lora_all_linear_r1 / 1 (value +1.0000): x chance (mean over layers) [CI] | 17.73 [16.54, 18.84] | `results/probes/j6/j6_llava_lora_all_linear_r1.json` | by_gamma['1'].x_chance_mean_over_layers / mean_layer_ratio_all_ci95 |
| J6 lora_all_linear_r1 / 1: mass % L0,8,16,24,31 / mean | 1.55, 14.80, 12.04, 6.56, 3.68 / 11.58 | `results/probes/j6/j6_llava_lora_all_linear_r1.json` | by_gamma['1'].mass_L0_L8_L16_L24_L31 / mean_mass_all_over_layers |
| J6 lora_all_linear_r1: chance (K/L) | 0.653 % | `results/probes/j6/j6_llava_lora_all_linear_r1.json` | by_gamma['0'].chance_all |
| J6 lora_all_linear_s2 / 0 (value +0.0000): x chance (mean over layers) [CI] | 16.27 [15.81, 16.71] | `results/probes/j6/j6_llava_lora_all_linear_s2.json` | by_gamma['0'].x_chance_mean_over_layers / mean_layer_ratio_all_ci95 |
| J6 lora_all_linear_s2 / 0: mass % L0,8,16,24,31 / mean | 1.19, 14.27, 11.00, 5.22, 3.18 / 10.63 | `results/probes/j6/j6_llava_lora_all_linear_s2.json` | by_gamma['0'].mass_L0_L8_L16_L24_L31 / mean_mass_all_over_layers |
| J6 lora_all_linear_s2 / landed (value -0.0300): x chance (mean over layers) [CI] | 17.68 [17.26, 18.08] | `results/probes/j6/j6_llava_lora_all_linear_s2.json` | by_gamma['landed'].x_chance_mean_over_layers / mean_layer_ratio_all_ci95 |
| J6 lora_all_linear_s2 / landed: mass % L0,8,16,24,31 / mean | 1.18, 15.36, 12.06, 5.88, 3.23 / 11.55 | `results/probes/j6/j6_llava_lora_all_linear_s2.json` | by_gamma['landed'].mass_L0_L8_L16_L24_L31 / mean_mass_all_over_layers |
| J6 lora_all_linear_s2 / 1 (value +1.0000): x chance (mean over layers) [CI] | 1.97 [1.78, 2.17] | `results/probes/j6/j6_llava_lora_all_linear_s2.json` | by_gamma['1'].x_chance_mean_over_layers / mean_layer_ratio_all_ci95 |
| J6 lora_all_linear_s2 / 1: mass % L0,8,16,24,31 / mean | 1.40, 1.35, 0.92, 0.77, 1.88 / 1.28 | `results/probes/j6/j6_llava_lora_all_linear_s2.json` | by_gamma['1'].mass_L0_L8_L16_L24_L31 / mean_mass_all_over_layers |
| J6 lora_all_linear_s2: chance (K/L) | 0.653 % | `results/probes/j6/j6_llava_lora_all_linear_s2.json` | by_gamma['0'].chance_all |

## tab_generation_supp

> rms_dropout_r1 captioning: no sourced result file in the released result files (scorer failed: no Java); reported as n/a.

> prefix-on POPE: scripts/eval_pope_prefix_on.py, graph built with build_prefix_graph_for_eval('rms_ungated') (ungated=True), 6x6 grid boxes; captioning not measured in this condition.

| quantity | value | source file | key path / derivation |
|---|---|---|---|
| vanilla POPE random accuracy (n=3000) | 86.30 | `results/generation/pope_captioning_summary.json` | vanilla.pope.random.accuracy |
| vanilla POPE popular accuracy (n=3000) | 85.13 | `results/generation/pope_captioning_summary.json` | vanilla.pope.popular.accuracy |
| vanilla POPE adversarial accuracy (n=3000) | 83.40 | `results/generation/pope_captioning_summary.json` | vanilla.pope.adversarial.accuracy |
| vanilla captioning Bleu_4 | 0.324 | `results/generation/captioning_metrics_final.json` | vanilla.Bleu_4 |
| vanilla captioning METEOR | 0.280 | `results/generation/captioning_metrics_final.json` | vanilla.METEOR |
| vanilla captioning ROUGE_L | 0.562 | `results/generation/captioning_metrics_final.json` | vanilla.ROUGE_L |
| vanilla captioning CIDEr | 1.132 | `results/generation/captioning_metrics_final.json` | vanilla.CIDEr |
| small_alpha POPE random accuracy (n=3000) | 86.30 | `results/generation/pope_captioning_summary.json` | small_alpha.pope.random.accuracy |
| small_alpha POPE popular accuracy (n=3000) | 85.53 | `results/generation/pope_captioning_summary.json` | small_alpha.pope.popular.accuracy |
| small_alpha POPE adversarial accuracy (n=3000) | 84.13 | `results/generation/pope_captioning_summary.json` | small_alpha.pope.adversarial.accuracy |
| small_alpha captioning Bleu_4 | 0.314 | `results/generation/captioning_metrics_final.json` | small_alpha.Bleu_4 |
| small_alpha captioning METEOR | 0.255 | `results/generation/captioning_metrics_final.json` | small_alpha.METEOR |
| small_alpha captioning ROUGE_L | 0.545 | `results/generation/captioning_metrics_final.json` | small_alpha.ROUGE_L |
| small_alpha captioning CIDEr | 1.069 | `results/generation/captioning_metrics_final.json` | small_alpha.CIDEr |
| rms_ungated POPE random accuracy (n=3000) | 83.97 | `results/generation/pope_captioning_summary.json` | rms_ungated.pope.random.accuracy |
| rms_ungated POPE popular accuracy (n=3000) | 83.30 | `results/generation/pope_captioning_summary.json` | rms_ungated.pope.popular.accuracy |
| rms_ungated POPE adversarial accuracy (n=3000) | 82.30 | `results/generation/pope_captioning_summary.json` | rms_ungated.pope.adversarial.accuracy |
| rms_ungated captioning Bleu_4 | 0.321 | `results/generation/captioning_metrics_final.json` | rms_ungated.Bleu_4 |
| rms_ungated captioning METEOR | 0.259 | `results/generation/captioning_metrics_final.json` | rms_ungated.METEOR |
| rms_ungated captioning ROUGE_L | 0.550 | `results/generation/captioning_metrics_final.json` | rms_ungated.ROUGE_L |
| rms_ungated captioning CIDEr | 1.096 | `results/generation/captioning_metrics_final.json` | rms_ungated.CIDEr |
| soft_prompt POPE random accuracy (n=3000) | 86.63 | `results/generation/pope_captioning_summary.json` | soft_prompt.pope.random.accuracy |
| soft_prompt POPE popular accuracy (n=3000) | 85.90 | `results/generation/pope_captioning_summary.json` | soft_prompt.pope.popular.accuracy |
| soft_prompt POPE adversarial accuracy (n=3000) | 84.20 | `results/generation/pope_captioning_summary.json` | soft_prompt.pope.adversarial.accuracy |
| soft_prompt captioning Bleu_4 | 0.287 | `results/generation/captioning_metrics_final.json` | soft_prompt.Bleu_4 |
| soft_prompt captioning METEOR | 0.235 | `results/generation/captioning_metrics_final.json` | soft_prompt.METEOR |
| soft_prompt captioning ROUGE_L | 0.532 | `results/generation/captioning_metrics_final.json` | soft_prompt.ROUGE_L |
| soft_prompt captioning CIDEr | 0.987 | `results/generation/captioning_metrics_final.json` | soft_prompt.CIDEr |
| rms_dropout_r1 POPE random accuracy (n=3000) | 86.40 | `results/generation/rms_dropout_r1_pope_captioning.json` | pope.random.accuracy |
| rms_dropout_r1 POPE popular accuracy (n=3000) | 85.70 | `results/generation/rms_dropout_r1_pope_captioning.json` | pope.popular.accuracy |
| rms_dropout_r1 POPE adversarial accuracy (n=3000) | 84.07 | `results/generation/rms_dropout_r1_pope_captioning.json` | pope.adversarial.accuracy |
| rms_dropout_r1 captioning metrics | n/a -- scorer failed (no java): captioning.metrics.error | `results/generation/rms_dropout_r1_pope_captioning.json` | captioning.metrics.error |
| rms_ungated PREFIX ON POPE random accuracy (n=3000) | 85.50 | `results/generation/rms_ungated_pope_prefix_on.json` | pope.random.accuracy |
| rms_ungated PREFIX ON POPE popular accuracy (n=3000) | 84.80 | `results/generation/rms_ungated_pope_prefix_on.json` | pope.popular.accuracy |
| rms_ungated PREFIX ON POPE adversarial accuracy (n=3000) | 83.37 | `results/generation/rms_ungated_pope_prefix_on.json` | pope.adversarial.accuracy |

## tab_verify_supp

| quantity | value | source file | key path / derivation |
|---|---|---|---|
| n150 A_pub_on_zero_boxes vs D_wrapper_bias_absent: flips | 0/150 = 0.00 % | `results/knockouts/verify_zero_boxes/verify_zero_boxes_n150.json` | flips['A_pub_on_zero_boxes vs D_wrapper_bias_absent'].n_flips / flip_pct |
| n150 A_pub_on_zero_boxes vs C_grid_on: flips | 8/150 = 5.33 % | `results/knockouts/verify_zero_boxes/verify_zero_boxes_n150.json` | flips['A_pub_on_zero_boxes vs C_grid_on'].n_flips / flip_pct |
| n150 A_pub_on_zero_boxes vs B_pub_off_lora_only: flips | 7/150 = 4.67 % | `results/knockouts/verify_zero_boxes/verify_zero_boxes_n150.json` | flips['A_pub_on_zero_boxes vs B_pub_off_lora_only'].n_flips / flip_pct |
| n150 A_pub_on_zero_boxes vs PUB_on_file: flips | 0/150 = 0.00 % | `results/knockouts/verify_zero_boxes/verify_zero_boxes_n150.json` | flips['A_pub_on_zero_boxes vs PUB_on_file'].n_flips / flip_pct |
| n150 B_pub_off_lora_only vs PUB_off_file: flips | 0/150 = 0.00 % | `results/knockouts/verify_zero_boxes/verify_zero_boxes_n150.json` | flips['B_pub_off_lora_only vs PUB_off_file'].n_flips / flip_pct |
| n150 A_pub_on_zero_boxes: acc | 56.67 | `results/knockouts/verify_zero_boxes/verify_zero_boxes_n150.json` | conditions.A_pub_on_zero_boxes.acc |
| n150 A_pub_on_zero_boxes: max / mean \|bias\|, items non-zero | 0.0000 / 0.0000, 0 | `results/knockouts/verify_zero_boxes/verify_zero_boxes_n150.json` | conditions.A_pub_on_zero_boxes.{max_abs_bias_over_items,mean_abs_bias_over_items,n_items_with_nonzero_bias} |
| n150 D_wrapper_bias_absent: acc | 56.67 | `results/knockouts/verify_zero_boxes/verify_zero_boxes_n150.json` | conditions.D_wrapper_bias_absent.acc |
| n150 D_wrapper_bias_absent: max / mean \|bias\|, items non-zero | 0.0000 / 0.0000, 0 | `results/knockouts/verify_zero_boxes/verify_zero_boxes_n150.json` | conditions.D_wrapper_bias_absent.{max_abs_bias_over_items,mean_abs_bias_over_items,n_items_with_nonzero_bias} |
| n150 C_grid_on: acc | 57.33 | `results/knockouts/verify_zero_boxes/verify_zero_boxes_n150.json` | conditions.C_grid_on.acc |
| n150 C_grid_on: max / mean \|bias\|, items non-zero | 0.7969 / 0.4059, 150 | `results/knockouts/verify_zero_boxes/verify_zero_boxes_n150.json` | conditions.C_grid_on.{max_abs_bias_over_items,mean_abs_bias_over_items,n_items_with_nonzero_bias} |
| n150 B_pub_off_lora_only: acc | 55.33 | `results/knockouts/verify_zero_boxes/verify_zero_boxes_n150.json` | conditions.B_pub_off_lora_only.acc |
| n150: n / n_with_scene_graph / beta | 150 / 0 / -0.6958 | `results/knockouts/verify_zero_boxes/verify_zero_boxes_n150.json` | n, n_with_scene_graph, bias_scale_beta |
| S4 n100: matches published on-file, grid / zero-box | 100 / 89 | `results/knockouts/verify_zero_boxes/verify_s4_box_source_n100.json` | conditions.{grid_6x6,zero_boxes_sg_protocol}.n_match_published_s4_file |
| S4 n100: matches LoRA-only off-file, grid / zero-box | 82 / 87 | `results/knockouts/verify_zero_boxes/verify_s4_box_source_n100.json` | conditions.*.n_match_lora_only_file |
| S4 n100: prefix max abs diff across images, grid / zero-box | 0.5840 / 0.0000 | `results/knockouts/verify_zero_boxes/verify_s4_box_source_n100.json` | conditions.*.prefix.max_abs_diff_across_images |
| S4 n100: prefix L2 mean, grid / zero-box | 27.04 / 25.51 | `results/knockouts/verify_zero_boxes/verify_s4_box_source_n100.json` | conditions.*.prefix.prefix_l2_mean |
| S4 n100: published on-file vs LoRA-only off-file differ | 18 / 100 = 18.0 % | `results/knockouts/verify_zero_boxes/verify_s4_box_source_n100.json` | published_s4_vs_lora_only_on_these |
| S4 n100: gate (raw parameter) | -0.3795 | `results/knockouts/verify_zero_boxes/verify_s4_box_source_n100.json` | gate |
| S4 published on-file vs LoRA-only off-file, all shared items: differing / n / % | 346 / 2000 / 17.30 | `results/published/s4_no_dropout_with_prefix.json` | prediction strings differ; off-file results/published/no_dropout_lora_only_gqa_testdev_predictions.json |
| S4 published files: acc on / off over shared items | 59.40 / 60.05 | `results/published/s4_no_dropout_with_prefix.json` | mean(correct) |

## Table 4 (main results): inference mode and box source per row

| quantity | value | source file | key path / derivation |
|---|---|---|---|
| LLaVA-1.5-7B vanilla: GQA / SC | 60.38 / 73.69 | `paper, Table 4 (typed, not generated)` | tab:main; inference mode / box source: no adaptation, no pathway; GQA: eval_gqa_parallel.py --model vanilla, full testdev; SC: no proposals used |
| + LoRA only (no geo. loss): GQA / SC | 57.21 / 74.98 | `paper, Table 4 (typed, not generated)` | tab:main; inference mode / box source: LoRA-only by construction (no pathway); full testdev |
| + Raw prefix (HyperVis): GQA / SC | 61.03 / 79.94 | `paper, Table 4 (typed, not generated)` | tab:main; inference mode / box source: values reported by HyperVis: GQA 61.03 = its LoRA-only evaluation (prefix absent); prefix on = 34.75 through its scene-graph proposal path -> empty proposal set on testdev -> image-independent CONSTANT prefix (results/knockouts/j5_hypervis_gqa.json reproduces 34.7 at s=1); SC with 6x6 grid boxes. Grid-proposal GQA: 50.4 (tab_2x2 / tab_prefix_scale_supp) |
| + Attn. bias (small alpha): GQA / SC | 60.16 / 75.73 | `paper, Table 4 (typed, not generated)` | tab:main; inference mode / box source: GQA: pathway absent -- scene-graph proposal path -> zero proposals -> bias identically zero (pair mask), LoRA-only in effect; full testdev. SC: 6x6 grid boxes, bias on. Grid knock-out: tab_knockout_main S2 |
| + RMS prefix, ungated: GQA / SC | 60.75 / 84.90 | `paper, Table 4 (typed, not generated)` | tab:main; inference mode / box source: GQA: LoRA-only (--model lora_only), full testdev; SC 84.90: full SugarCrepe with the prefix on at scale 1, 6x6 grid (results/published/rms_ungated_prefix_on_compositionality.json) |
| + Soft prompt (no geometry): GQA / SC | 59.90 / 77.00 | `paper, Table 4 (typed, not generated)` | tab:main; inference mode / box source: content-free at inference (learned embeddings x tanh gamma, no image dependence); GQA: --model lora_only, full testdev; SC: grid, prefix on |
| + Euclidean, gated: GQA / SC | 59.73 / 76.56 | `paper, Table 4 (typed, not generated)` | tab:main; inference mode / box source: GQA 59.73: full testdev, --model lora_only (prefix absent; results/published/euclidean_lora_only_gqa_testdev_results.json); SC: 6x6 grid, prefix on |
| + Raw prefix, gated (S5): GQA / SC | 60.00 / 76.18 | `paper, Table 4 (typed, not generated)` | tab:main; inference mode / box source: GQA 60.00: full testdev, --model lora_only (prefix absent; results/published/raw_gated_lora_only_gqa_testdev_results.json); SC: 6x6 grid, prefix on. Grid ablation: tab_knockout_main S5 (44.0 vs 50.7 GQA-rel) |
| + Hyp. prefix, no geo. losses (G2): GQA / SC | 59.59 / 78.84 | `paper, Table 4 (typed, not generated)` | tab:main; inference mode / box source: GQA: --model lora_only, full testdev (prefix absent); GQA-rel/attr of Table 5: eval_gqa_spatial.py prefix model with the 6x6 grid fallback and the prefix scaled by tanh(gate) ~ -0.30; SC: prefix wrapper, grid |
| + Soft prompt, 2nd run (synthesis): GQA / SC | 60.07 / 77.45 | `paper, Table 4 (typed, not generated)` | tab:main; inference mode / box source: content-free soft prompt at inference; GQA: --model lora_only, full testdev; GQA-rel/attr/VSR of Table 5: eval_gqa_spatial.py grid fallback / eval_vsr.py; SC: grid |

## reproducibility facts and quoted numbers (sources)

| quantity | value | source file | key path / derivation |
|---|---|---|---|
| training set size | 943,000 questions over 72,140 images (GQA train_balanced) | `paper (supplement S3 / Sec. 5.2)` | training-set size |
| hardware | NVIDIA H100 80GB HBM3, 2 GPUs per run | `paper (supplement S3 / Sec. 5.2)` | hardware and wall-clock |
| LoRA all-linear trainable parameters | 39,976,960 vs 8,388,608 (q/v only) = 4.8x  (39.98M vs 8.39M) | `paper (supplement S3 / Sec. 5.2)` | LoRA parameter counts |
| entailment loss at end of training, flat vs hyperbolic | 0.6620 (euclidean_geo_r1) vs 0.0108 (rms_ungated_r2), ratio 61x | `paper (supplement S3 / Sec. 5.2)` | final entailment-loss values |
| multi-seed flat-vs-hyperbolic GQA-rel delta (1-epoch horizon) | -0.10 [-0.55, +0.34] p=0.65recomputation -0.10 [-0.54, +0.34] p=0.66 | `paper (supplement S3 / Sec. 5.2)` | Euclidean n=3: 53.32+-0.23; hyperbolic n=2: 53.22+-0.19 (1-epoch seeds) |
| recomputed here: hyperbolic (2 seeds) - flat (3 seeds), GQA-rel 1-epoch, seed-mean per question, paired bootstrap 10,000 | -0.10 [-0.55, +0.35] p=0.65 (means 53.22 vs 53.32, n=5308); per-seed accs hyp 53.35, 53.09; flat 53.30, 53.56, 53.11 | `results/probes` | hyp_r{2,3}_1ep_rel.json, euclid_s{1,2,3}_1ep_rel.json |

## Numeric tokens per generated table (auto-extracted from the .tex as printed)

Every token below appears verbatim in the named table file; the sections above give each one its source file and key path at full precision.

- `tables/tab_knockout_main.tex` (sources: results/knockouts/j7_hra_v1_gqa.json, results/knockouts/j7_hra_v1_gqa_rel.json, results/knockouts/j7_hra_v1_sugarcrepe.json, results/knockouts/j7_no_dropout_gqa.json, results/knockouts/j7_no_dropout_gqa_rel.json, results/knockouts/j7_no_dropout_sugarcrepe.json, results/knockouts/j7_raw_gated_gqa.json, results/knockouts/j7_raw_gated_gqa_rel.json, results/knockouts/j7_raw_gated_sugarcrepe.json, results/knockouts/j7_small_alpha_gqa.json, results/knockouts/j7_small_alpha_gqa_rel.json, results/knockouts/j7_small_alpha_sugarcrepe.json, results/knockouts/j7_stage2_gqa_rel.json, results/knockouts/j7_stage2_sugarcrepe.json): 2, 4, 95, 0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.3, 1.4, 1.5, 1.8, 1.9, 2.2, 2.3, 2.4, 2.6, 4.0, 4.4, 8.3, 9.7, 0.01, 0.05, 0.06, 0.26, 0.39, 0.62, 0.86, 13.6, 19.5, 44.0, 50.5, 50.6, 50.7, 51.7, 52.2, 53.1, 53.2, 53.3, 53.4, 54.1, 54.3, 58.6, 59.4, 59.6, 60.1, 60.6, 76.8, 77.1, 77.2, 77.3, 77.4, 78.5, 78.7, 78.8, 80.0, 80.1, 81.5, 0.005, 1,000, 2,000, 57.93, 58.55, 60.09, 60.14, 12,578
- `tables/tab_gated_prefix_supp.tex` (sources: results/knockouts/j7_base_r1_gqa_rel.json, results/knockouts/j7_base_r1_sugarcrepe.json, results/knockouts/j7_euclidean_gqa_rel.json, results/knockouts/j7_euclidean_sugarcrepe.json, results/knockouts/j7_lora_all_linear_r1_gqa_rel.json, results/knockouts/j7_lora_all_linear_r1_sugarcrepe.json, results/knockouts/j7_lora_all_linear_s2_gqa_rel.json, results/knockouts/j7_lora_all_linear_s2_sugarcrepe.json, results/knockouts/j7_no_dropout_gqa_rel.json, results/knockouts/j7_no_dropout_sugarcrepe.json, results/knockouts/j7_raw_gated_gqa_rel.json, results/knockouts/j7_raw_gated_sugarcrepe.json, results/knockouts/j7_soft_prompt_gqa_rel.json, results/knockouts/j7_soft_prompt_sugarcrepe.json, results/knockouts/j7_synthesis_gqa_rel.json, results/knockouts/j7_synthesis_sugarcrepe.json): 0, 1, 3, 5, 8, 96, 0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.3, 2.4, 12.2, 44.0, 48.1, 49.6, 49.9, 50.2, 50.4, 50.5, 50.6, 50.7, 51.0, 51.1, 51.6, 52.8, 53.1, 53.2, 53.3, 53.4, 53.5, 53.6, 53.7, 54.2, 54.3, 54.5, 54.6, 54.9, 55.1, 76.8, 76.9, 77.1, 77.2, 77.3, 77.5, 77.8, 77.9, 78.2, 78.5, 78.7, 78.8, 78.9, 79.1, 79.3, 79.4, 79.5, 79.8, 79.9, 80.0, 80.1, 80.3, 80.4, 80.6, 80.7, 81.5, 82.8, 0.005, 0.024, 0.119, 0.185, 1{,}000
- `tables/tab_s5_sweep_supp.tex` (sources: results/knockouts/j7_raw_gated_{gqa_rel,sugarcrepe}.json): 0, 1, 2, 4, 95, 0.0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0, 1.2, 1.5, 1.6, 1.7, 1.8, 2.4, 2.7, 2.8, 3.6, 3.8, 3.9, 4.0, 4.2, 4.5, 5.1, 5.4, 5.8, 5.9, 6.2, 6.8, 7.7, 7.9, 8.5, 0.01, 0.03, 12.2, 33.9, 36.4, 39.2, 44.0, 46.5, 47.4, 49.1, 50.5, 50.7, 53.5, 54.3, 76.8, 78.0, 78.5, 79.1, 79.4, 79.5, 80.0, 81.2, 82.3, 84.3, 84.7, 85.3, 0.001, 0.003, 0.0052, 1{,}000
- `tables/tab_2x2.tex` (sources: results/knockouts/j5_hypervis_gqa_gridboxes.json, j5_hypervis_sugarcrepe.json, j5_accv_rms_ungated_{gqa,sugarcrepe}.json, j7_raw_gated_{gqa_rel,sugarcrepe}.json, j7_base_r1_{gqa_rel,sugarcrepe}.json): 0, 1, 95, 0.3, 0.6, 0.7, 1.2, 1.3, 1.5, 2.1, 2.4, 5.1, 6.5, 7.8, 8.0, 9.3, 10.8, 44.0, 48.5, 49.9, 50.2, 50.4, 50.7, 59.0, 60.8, 73.4, 75.8, 78.5, 79.4, 79.9, 80.0, 80.7, 85.2
- `tables/tab_prefix_scale_supp.tex` (sources: results/knockouts/j5_accv_rms_ungated_{gqa,sugarcrepe}.json, j5_hypervis_gqa_gridboxes.json, j5_hypervis_gqa.json, j5_hypervis_sugarcrepe.json): 0, 1, 2, 3, 4, 6, 95, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1, 1.2, 1.4, 1.6, 1.7, 2.4, 7.8, 9.3, 0.01, 0.03, 10.8, 31.4, 34.7, 34.8, 48.5, 50.4, 58.4, 58.8, 58.9, 59.0, 59.1, 59.2, 59.9, 60.7, 60.8, 61.0, 61.2, 61.3, 61.5, 73.0, 73.3, 73.4, 73.6, 73.9, 75.2, 75.3, 75.5, 75.6, 75.8, 75.9, 76.0, 76.1, 76.5, 76.7, 77.5, 77.8, 79.9, 80.3, 85.2, 0.001, 0.003, 0.0003, 2{,}000
- `tables/tab_matched.tex` (sources: results/matched_mode/gqa_drop/*.json, gqa_active/*.json, vsr/*/vsr_predictions.json): 2, 3, 5, 8, 95, 0.0, 0.1, 0.2, 0.3, 0.8, 1.5, 1.9, 2.1, 2.7, 2.8, 2.9, 3.5, 3.7, 3.8, 3.9, 5.0, 5.4, 5.8, 6.1, 6.2, 6.3, 6.5, 6.9, 7.2, 7.4, 7.6, 8.0, 8.3, 8.4, 8.6, 8.7, 9.0, 9.1, 9.5, 10.0, 10.9, 12.3, 12.9, 14.3, 14.6, 15.5, 16.7, 37.9, 45.9, 49.5, 49.6, 50.1, 52.3, 52.4, 52.6, 52.9, 53.0, 53.2, 53.3, 53.4, 54.5, 55.4, 55.6, 56.0, 56.2, 56.6, 59.3, 62.4, 62.8, 63.4, 63.6, 63.8, 65.1, 65.3, 66.1, 66.2, 66.5, 66.6, 66.7, 67.0, 67.1, 67.2, 67.5, 68.0, 68.5, 2{,}113, 5{,}186, 5{,}308
- `tables/tab_matched_contrasts.tex` (sources: results/matched_mode/gqa_drop/*.json, vsr/*/vsr_predictions.json): 2, 4, 6, 95, 0.08, 0.09, 0.12, 0.13, 0.23, 0.24, 0.28, 0.34, 0.35, 0.36, 0.46, 0.47, 0.48, 0.50, 0.60, 0.69, 0.73, 0.77, 0.83, 0.87, 0.94, 0.95, 0.96, 0.99, 1.02, 1.04, 1.07, 1.14, 1.22, 1.35, 1.37, 1.41, 1.45, 1.51, 1.54, 1.61, 1.80, 2.04, 2.22, 2.26, 2.32, 2.70, 2.93, 3.12, 3.64, 3.88, 4.73, 4.92, 6.44, 6.67, 0.001, 0.002, 0.003, 0.004, 0.034, 0.070, 0.090, 0.160, 0.176, 0.181, 0.227, 0.229, 0.260, 0.305, 0.343, 0.391, 0.492, 0.531, 0.540, 0.700, 0.813, 0.836, 0.839, 0.844, 2{,}113, 5{,}186, 5{,}308
- `tables/tab_vsr_seeds_supp.tex` (sources: results/matched_mode/vsr/{rmsPub,rmsR2,rmsR3,eucGeo,eucGeoS2,eucGeoS3}_DROP/vsr_predictions.json, vsr/vanilla/): 2, 3.8, 1.17, 1.52, 1.64, 2.25, 2.42, 2.49, 0.092, 0.137, 0.433, 63.80, 65.36, 65.45, 66.21, 66.49, 66.82, 67.85, 68.53, 2{,}113
- `tables/tab_backbones.tex` (sources: results/knockouts/j7_*_gqa_rel.json (LLaVA gates + forced), results/external/gate-objective/{g1b*}/seed*/FINAL.json, gsweep/w4aud_g1b0_gqa.json, results/probes/internvl_seed*.json, internvl_knockout*.json): 1, 2, 3, 5, 6, 7, 15, 95, 0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 2.1, 3.4, 0.15, 0.17, 0.53, 0.55, 0.001, 0.005, 0.006, 0.009, 0.013, 0.014, 0.030, 0.185
- `tables/tab_gradient_supp.tex` (sources: results/probes/gate_gradient.json): 0, 1, 120, 0.01, 0.02, 0.19, 2.42, 0.674, 3.512, 6.571, 1208.461
- `tables/tab_attention_supp.tex` (sources: results/probes/layerwise_attention.json; results/probes/j6/j6_llava_*.json): 0, 1, 6, 9, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 100, 2.0, 2.1, 2.2, 2.5, 2.6, 2.7, 2.8, 2.9, 3.2, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.3, 4.4, 4.5, 5.2, 5.8, 5.9, 6.3, 6.6, 6.9, 7.0, 7.3, 7.6, 7.7, 8.1, 8.2, 8.4, 8.5, 8.7, 8.9, 9.0, 9.1, 9.6, 10.5, 10.6, 10.7, 11.0, 11.1, 11.2, 11.6, 11.9, 12.0, 12.1, 12.4, 12.5, 13.4, 13.5, 14.1, 14.3, 14.8, 15.4, 16.2, 16.3, 16.7, 17.4, 17.5, 17.7, 17.8, 17.9, 18.1, 18.4, 18.5, 18.8, 18.9, 19.1, 20.0, 20.6, 20.8, 21.9, 23.2, 23.4, 23.5, 23.6, 24.0, 24.5, 24.6, 24.8, 24.9, 25.2, 25.3, 25.5, 25.6, 25.8, 26.1, 26.2, 26.4, 26.8, 27.0, 27.1, 27.2, 27.4, 27.8, 27.9, 28.0, 28.1, 29.6, 30.4, 33.5, 35.5, 35.9, 36.1, 36.7, 36.8, 37.5, 37.7, 39.9, 40.0, 0.005, 0.024, 0.119, 0.185
- `tables/tab_generation_supp.tex` (sources: results/generation/pope_captioning_summary.json, captioning_metrics_final.json, rms_dropout_r1_pope_captioning.json): 1, 2, 3, 4, 5, 50, 0.235, 0.255, 0.259, 0.280, 0.287, 0.314, 0.321, 0.324, 0.532, 0.545, 0.550, 0.562, 0.987, 1.069, 1.096, 1.132, 82.30, 83.30, 83.37, 83.40, 83.97, 84.07, 84.13, 84.20, 84.80, 85.13, 85.50, 85.53, 85.70, 85.90, 86.30, 86.40, 86.63, 3{,}000
- `tables/tab_verify_supp.tex` (sources: results/knockouts/verify_zero_boxes/verify_zero_boxes_n150.json, verify_s4_box_source_n100.json, results/published/s4_no_dropout_with_prefix.json): 0, 2, 6, 7, 8, 10, 18, 89, 0.0, 100, 150, 346, 4.7, 5.3, 17.3, 18.0, 55.3, 56.7, 57.3, 0.000, 0.406, 0.584, 0.797, 2,000, 2{,}000

## Cross-checks against the numbers quoted in the paper

179 of 181 checks agree within tolerance.

| check | computed | expected | ok |
|---|---|---|---|
| VSR hyp mean (quoted 67.85) | 67.8498 | 67.8500 | yes |
| VSR flat mean (quoted 65.36) | 65.3573 | 65.3600 | yes |
| VSR Welch t hyp-flat (quoted 2.25) | 2.2503 | 2.2500 | yes |
| VSR Welch p hyp-flat (quoted 0.091) | 0.0918 | 0.0910 | yes |
| VSR hyp-vanilla p (quoted 0.137) | 0.1367 | 0.1370 | yes |
| VSR flat-vanilla p (quoted 0.431) | 0.4332 | 0.4310 | **NO** |
| stage2 beta (expected -8.496) | -8.4962 | -8.4960 | yes |
| Qwen max |tanh g| <= 0.018 | 0.0175 | 0.0000 | yes |
| Qwen gamma=1.5 delta (+0.17) | 0.1700 | 0.1700 | yes |
| Qwen gamma=1.5 CI lo (-0.21) | -0.2070 | -0.2100 | yes |
| Qwen gamma=1.5 CI hi (+0.53) | 0.5280 | 0.5300 | yes |
| S4 item flips over 2,000 (17.3 %) | 17.3000 | 17.3000 | yes |
| 1-ep multi-seed delta (-0.10) | -0.1005 | -0.1000 | yes |
| quoted hra_v1/gqa_rel value 0 acc 51.7 | 51.7000 | 51.7000 | yes |
| quoted hra_v1/gqa_rel value 1 acc 52.2 | 52.2000 | 52.2000 | yes |
| quoted hra_v1/gqa_rel value 1 delta +0.5 | 0.5000 | 0.5000 | yes |
| quoted hra_v1/gqa_rel value 1 CI lo -0.4 | -0.4000 | -0.4000 | yes |
| quoted hra_v1/gqa_rel value 1 CI hi +1.3 | 1.3025 | 1.3000 | yes |
| quoted hra_v1/sugarcrepe value 0 acc 77.4 | 77.4000 | 77.4000 | yes |
| quoted hra_v1/sugarcrepe value 1 acc 77.2 | 77.2000 | 77.2000 | yes |
| quoted hra_v1/sugarcrepe value 1 delta -0.2 | -0.2000 | -0.2000 | yes |
| quoted hra_v1/sugarcrepe value 1 CI lo -0.7 | -0.7000 | -0.7000 | yes |
| quoted hra_v1/sugarcrepe value 1 CI hi +0.3 | 0.3000 | 0.3000 | yes |
| quoted stage2/gqa_rel value 0 acc 53.1 | 53.1000 | 53.1000 | yes |
| quoted stage2/gqa_rel value 1 acc 53.3 | 53.3000 | 53.3000 | yes |
| quoted stage2/gqa_rel value 1 delta +0.2 | 0.2000 | 0.2000 | yes |
| quoted stage2/gqa_rel value 1 CI lo -0.3 | -0.3025 | -0.3000 | yes |
| quoted stage2/gqa_rel value 1 CI hi +0.8 | 0.8000 | 0.8000 | yes |
| quoted stage2/sugarcrepe value 0 acc 77.2 | 77.2000 | 77.2000 | yes |
| quoted stage2/sugarcrepe value 1 acc 77.3 | 77.3000 | 77.3000 | yes |
| quoted stage2/sugarcrepe value 1 delta +0.1 | 0.1000 | 0.1000 | yes |
| quoted base_r1/gqa_rel value 0 acc 49.9 | 49.9000 | 49.9000 | yes |
| quoted base_r1/gqa_rel value 1 acc 53.3 | 53.3000 | 53.3000 | yes |
| quoted base_r1/gqa_rel value 1 delta +3.4 | 3.4000 | 3.4000 | yes |
| quoted base_r1/sugarcrepe value 0 acc 79.4 | 79.4000 | 79.4000 | yes |
| quoted base_r1/sugarcrepe value 1 acc 77.5 | 77.5000 | 77.5000 | yes |
| quoted raw_gated/gqa_rel value 0 acc 50.7 | 50.7000 | 50.7000 | yes |
| quoted raw_gated/gqa_rel value 1 acc 12.2 | 12.2000 | 12.2000 | yes |
| quoted raw_gated/gqa_rel value 1 delta -38.5 | -38.5000 | -38.5000 | yes |
| quoted raw_gated/gqa_rel value 0.01 acc 36.4 | 36.4000 | 36.4000 | yes |
| quoted raw_gated/gqa_rel value 0.1 acc 33.9 | 33.9000 | 33.9000 | yes |
| quoted raw_gated/sugarcrepe value 0 acc 78.5 | 78.5000 | 78.5000 | yes |
| quoted raw_gated/sugarcrepe value 1 acc 79.5 | 79.5000 | 79.5000 | yes |
| quoted raw_gated/sugarcrepe value 0.03 acc 85.3 | 85.3000 | 85.3000 | yes |
| quoted raw_gated/sugarcrepe value 0.03 delta +6.8 | 6.8000 | 6.8000 | yes |
| quoted raw_gated/sugarcrepe value 0.03 CI lo +5.1 | 5.1000 | 5.1000 | yes |
| quoted raw_gated/sugarcrepe value 0.03 CI hi +8.5 | 8.5000 | 8.5000 | yes |
| quoted no_dropout/gqa_rel value 0 acc 50.5 | 50.5000 | 50.5000 | yes |
| quoted no_dropout/gqa_rel value 1 acc 53.1 | 53.1000 | 53.1000 | yes |
| quoted no_dropout/sugarcrepe value 0 acc 80.1 | 80.1000 | 80.1000 | yes |
| quoted no_dropout/sugarcrepe value 1 acc 79.5 | 79.5000 | 79.5000 | yes |
| quoted soft_prompt/gqa_rel value 0 acc 51.6 | 51.6000 | 51.6000 | yes |
| quoted euclidean/gqa_rel value 0 acc 51.0 | 51.0000 | 51.0000 | yes |
| quoted synthesis/gqa_rel value 0 acc 49.6 | 49.6000 | 49.6000 | yes |
| quoted lora_all_linear_r1/gqa_rel value 0 acc 54.5 | 54.5000 | 54.5000 | yes |
| quoted lora_all_linear_s2/gqa_rel value 0 acc 54.3 | 54.3000 | 54.3000 | yes |
| quoted base_r1/gqa_rel learned acc 50.2 | 50.2000 | 50.2000 | yes |
| quoted base_r1/gqa_rel learned delta +0.3 | 0.3000 | 0.3000 | yes |
| quoted base_r1/gqa_rel learned CI lo -0.5 | -0.5000 | -0.5000 | yes |
| quoted base_r1/gqa_rel learned CI hi +1.2 | 1.2000 | 1.2000 | yes |
| quoted base_r1/sugarcrepe learned acc 80.7 | 80.7000 | 80.7000 | yes |
| quoted base_r1/sugarcrepe learned delta +1.3 | 1.3000 | 1.3000 | yes |
| quoted base_r1/sugarcrepe learned CI lo +0.6 | 0.6000 | 0.6000 | yes |
| quoted base_r1/sugarcrepe learned CI hi +2.1 | 2.1000 | 2.1000 | yes |
| quoted raw_gated/gqa_rel learned acc 44.0 | 44.0000 | 44.0000 | yes |
| quoted raw_gated/gqa_rel learned delta -6.7 | -6.7000 | -6.7000 | yes |
| quoted raw_gated/gqa_rel learned CI lo -8.4 | -8.4000 | -8.4000 | yes |
| quoted raw_gated/gqa_rel learned CI hi -5.0 | -5.0000 | -5.0000 | yes |
| quoted raw_gated/sugarcrepe learned acc 80.0 | 80.0000 | 80.0000 | yes |
| quoted raw_gated/sugarcrepe learned delta +1.5 | 1.5000 | 1.5000 | yes |
| quoted raw_gated/sugarcrepe learned CI lo +0.7 | 0.7000 | 0.7000 | yes |
| quoted raw_gated/sugarcrepe learned CI hi +2.4 | 2.4000 | 2.4000 | yes |
| quoted soft_prompt/gqa_rel learned acc 51.1 | 51.1000 | 51.1000 | yes |
| quoted no_dropout/gqa_rel learned acc 50.6 | 50.6000 | 50.6000 | yes |
| quoted no_dropout/sugarcrepe learned acc 81.5 | 81.5000 | 81.5000 | yes |
| quoted euclidean/gqa_rel learned acc 50.4 | 50.4000 | 50.4000 | yes |
| quoted synthesis/gqa_rel learned acc 48.1 | 48.1000 | 48.1000 | yes |
| quoted synthesis/sugarcrepe learned acc 80.4 | 80.4000 | 80.4000 | yes |
| quoted lora_all_linear_r1/gqa_rel learned acc 54.2 | 54.2000 | 54.2000 | yes |
| quoted lora_all_linear_s2/gqa_rel learned acc 54.6 | 54.6000 | 54.6000 | yes |
| quoted raw_gated -learned acc 53.5 / +2.8 [1.5, 4.2] | 53.5000 | 53.5000 | yes |
| quoted raw_gated -learned delta +2.8 | 2.8000 | 2.8000 | yes |
| quoted base_r1/gqa_rel absent 53.2 | 53.2000 | 53.2000 | yes |
| quoted soft_prompt/gqa_rel absent 53.7 | 53.7000 | 53.7000 | yes |
| quoted no_dropout/gqa_rel absent 53.4 | 53.4000 | 53.4000 | yes |
| quoted euclidean/gqa_rel absent 53.5 | 53.5000 | 53.5000 | yes |
| quoted synthesis/gqa_rel absent 54.3 | 54.3000 | 54.3000 | yes |
| quoted raw_gated/gqa_rel absent 54.3 | 54.3000 | 54.3000 | yes |
| quoted lora_all_linear_r1/gqa_rel absent 54.9 | 54.9000 | 54.9000 | yes |
| quoted lora_all_linear_s2/gqa_rel absent 54.2 | 54.2000 | 54.2000 | yes |
| quoted j5 accv_rms_ungated_gqa s=0 acc 60.8 | 60.8000 | 60.8000 | yes |
| quoted j5 accv_rms_ungated_gqa s=1 acc 48.5 | 48.5000 | 48.5000 | yes |
| quoted j5 accv_rms_ungated_gqa s=1 delta -12.3 | -12.3000 | -12.3000 | yes |
| quoted j5 accv_rms_ungated_gqa s=0.3 acc 61.5 | 61.5000 | 61.5000 | yes |
| quoted j5 accv_rms_ungated_sugarcrepe s=0 acc 75.8 | 75.8500 | 75.8000 | yes |
| quoted j5 accv_rms_ungated_sugarcrepe s=1 acc 85.2 | 85.1500 | 85.2000 | yes |
| quoted j5 accv_rms_ungated_sugarcrepe s=1 delta +9.3 | 9.3000 | 9.3000 | yes |
| quoted j5 accv_rms_ungated_sugarcrepe s=1 CI lo +7.9 | 7.8500 | 7.9000 | yes |
| quoted j5 accv_rms_ungated_sugarcrepe s=1 CI hi +10.8 | 10.8000 | 10.8000 | yes |
| quoted j5 accv_rms_ungated_sugarcrepe s=0.3 acc 77.5 | 77.5500 | 77.5000 | yes |
| quoted j5 hypervis_gqa_gridboxes s=1 acc 50.4 | 50.4000 | 50.4000 | yes |
| quoted j5 hypervis_gqa_gridboxes s=1 delta -8.6 | -8.5500 | -8.6000 | yes |
| quoted j5 hypervis_gqa_gridboxes s=1 CI lo -10.5 | -10.5000 | -10.5000 | yes |
| quoted j5 hypervis_gqa_gridboxes s=1 CI hi -6.6 | -6.6000 | -6.6000 | yes |
| quoted j5 hypervis_gqa_gridboxes s=0 acc 59.0 | 58.9500 | 59.0000 | yes |
| quoted j5 hypervis_gqa s=1 acc 34.7 | 34.7000 | 34.7000 | yes |
| quoted j5 rms_ungated GQA absent 61.3 | 61.2500 | 61.3000 | yes |
| E1 vanilla/rel DROP 52.94 | 52.9390 | 52.9400 | yes |
| E1 rmsPub/rel DROP 53.41 | 53.4099 | 53.4100 | yes |
| E1 rmsPub/rel ACT 37.92 | 37.9239 | 37.9200 | yes |
| E1 rmsPub/rel DROP-ACT +15.49 | 15.4861 | 15.4900 | yes |
| E1 rmsR2/rel DROP 53.28 | 53.2781 | 53.2800 | yes |
| E1 rmsR2/rel ACT 45.89 | 45.8930 | 45.8900 | yes |
| E1 rmsR2/rel DROP-ACT +7.39 | 7.3851 | 7.3900 | yes |
| E1 eucGeo/rel DROP 53.18 | 53.1839 | 53.1800 | yes |
| E1 eucGeo/rel ACT 52.35 | 52.3549 | 52.3500 | yes |
| E1 eucGeo/rel DROP-ACT +0.83 | 0.8289 | 0.8300 | yes |
| E1 soft/rel DROP 53.03 | 53.0332 | 53.0300 | yes |
| E1 soft/rel ACT 50.09 | 50.0942 | 50.0900 | yes |
| E1 soft/rel DROP-ACT +2.94 | 2.9390 | 2.9400 | yes |
| E1 euc/rel DROP 52.32 | 52.3173 | 52.3200 | yes |
| E1 euc/rel ACT 49.64 | 49.6420 | 49.6400 | yes |
| E1 euc/rel DROP-ACT +2.68 | 2.6752 | 2.6800 | yes |
| E1 g2/rel DROP 52.34 | 52.3361 | 52.3400 | yes |
| E1 g2/rel ACT 49.55 | 49.5479 | 49.5500 | yes |
| E1 g2/rel DROP-ACT +2.79 | 2.7882 | 2.7900 | yes |
| E1 smallA/rel DROP 52.43 | 52.4303 | 52.4300 | yes |
| E1 smallA/rel ACT 52.56 | 52.5622 | 52.5600 | yes |
| E1 smallA/rel DROP-ACT -0.13 | -0.1319 | -0.1300 | yes |
| E1 vanilla/attr DROP 66.12 | 66.1203 | 66.1200 | yes |
| E1 rmsPub/attr DROP 66.62 | 66.6217 | 66.6200 | yes |
| E1 rmsPub/attr ACT 59.26 | 59.2557 | 59.2600 | yes |
| E1 rmsPub/attr DROP-ACT +7.37 | 7.3660 | 7.3700 | yes |
| E1 rmsR2/attr DROP 66.24 | 66.2360 | 66.2400 | yes |
| E1 rmsR2/attr ACT 62.36 | 62.3602 | 62.3600 | yes |
| E1 rmsR2/attr DROP-ACT +3.88 | 3.8758 | 3.8800 | yes |
| E1 eucGeo/attr DROP 66.72 | 66.7181 | 66.7200 | yes |
| E1 eucGeo/attr ACT 67.95 | 67.9522 | 67.9500 | yes |
| E1 eucGeo/attr DROP-ACT -1.23 | -1.2341 | -1.2300 | yes |
| E1 soft/attr DROP 65.1 | 65.0983 | 65.1000 | yes |
| E1 soft/attr ACT 67.18 | 67.1809 | 67.1800 | yes |
| E1 soft/attr DROP-ACT -2.08 | -2.0825 | -2.0800 | yes |
| E1 euc/attr DROP 65.35 | 65.3490 | 65.3500 | yes |
| E1 euc/attr ACT 67.05 | 67.0459 | 67.0500 | yes |
| E1 euc/attr DROP-ACT -1.7 | -1.6969 | -1.7000 | yes |
| E1 g2/attr DROP 65.14 | 65.1369 | 65.1400 | yes |
| E1 g2/attr ACT 67.12 | 67.1230 | 67.1200 | yes |
| E1 g2/attr DROP-ACT -1.98 | -1.9861 | -1.9800 | **NO** |
| E1 smallA/attr DROP 67.47 | 67.4701 | 67.4700 | yes |
| E1 smallA/attr ACT 67.47 | 67.4701 | 67.4700 | yes |
| E1 smallA/attr DROP-ACT +0.0 | 0.0000 | 0.0000 | yes |
| E1 vanilla/vsr DROP 66.21 | 66.2092 | 66.2100 | yes |
| E1 rmsPub/vsr DROP 66.49 | 66.4931 | 66.4900 | yes |
| E1 rmsPub/vsr ACT 55.56 | 55.5608 | 55.5600 | yes |
| E1 rmsPub/vsr DROP-ACT +10.93 | 10.9323 | 10.9300 | yes |
| E1 rmsR2/vsr DROP 68.53 | 68.5282 | 68.5300 | yes |
| E1 rmsR2/vsr ACT 56.22 | 56.2234 | 56.2200 | yes |
| E1 rmsR2/vsr DROP-ACT +12.3 | 12.3048 | 12.3000 | yes |
| E1 eucGeo/vsr DROP 63.8 | 63.7956 | 63.8000 | yes |
| E1 eucGeo/vsr ACT 56.55 | 56.5547 | 56.5500 | yes |
| E1 eucGeo/vsr DROP-ACT +7.24 | 7.2409 | 7.2400 | yes |
| E1 soft/vsr DROP 63.61 | 63.6062 | 63.6100 | yes |
| E1 soft/vsr ACT 56.03 | 56.0341 | 56.0300 | yes |
| E1 soft/vsr DROP-ACT +7.57 | 7.5722 | 7.5700 | yes |
| E1 euc/vsr DROP 62.85 | 62.8490 | 62.8500 | yes |
| E1 euc/vsr ACT 55.99 | 55.9867 | 55.9900 | yes |
| E1 euc/vsr DROP-ACT +6.86 | 6.8623 | 6.8600 | yes |
| E1 g2/vsr DROP 63.37 | 63.3696 | 63.3700 | yes |
| E1 g2/vsr ACT 55.37 | 55.3715 | 55.3700 | yes |
| E1 g2/vsr DROP-ACT +8.0 | 7.9981 | 8.0000 | yes |
| E1 smallA/vsr DROP 54.47 | 54.4723 | 54.4700 | yes |
| E1 smallA/vsr ACT 54.52 | 54.5196 | 54.5200 | yes |
| E1 smallA/vsr DROP-ACT -0.05 | -0.0473 | -0.0500 | yes |
| E1 check(2) rmsR2_DROP-vanilla/rel +0.34 | 0.3391 | 0.3400 | yes |
| E1 check(2) eucGeo_DROP-vanilla/rel +0.24 | 0.2449 | 0.2400 | yes |
| E1 check(2) rmsR2_DROP-vanilla/attr +0.12 | 0.1157 | 0.1200 | yes |
| E1 check(2) eucGeo_DROP-vanilla/attr +0.6 | 0.5978 | 0.6000 | yes |
| E1 check(2) eucGeo_DROP-soft_DROP/attr +1.62 | 1.6197 | 1.6200 | yes |
| E1 check(2) rmsR2_DROP-vanilla/vsr +2.32 | 2.3190 | 2.3200 | yes |
| E1 check(2) rmsPub_DROP-vanilla/vsr +0.28 | 0.2840 | 0.2800 | yes |
| E1 check(2) eucGeo_DROP-vanilla/vsr -2.41 | -2.4136 | -2.4100 | yes |
