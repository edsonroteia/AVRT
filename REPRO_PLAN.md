# AVRT Reproduction Plan (Paper + Rebuttal)

Goal: reproduce the paper results and rebuttal addenda end-to-end, including data generation, training (SFT + GRPO), evaluation, ablations, and tables/figures.

## 1) Data + Preprocessing Artifacts
- `data/avqa/` fetch script + checksum list; split definitions for train/val/test used in paper.
- `data/dailyomni/`, `data/omnibench/`, `data/mmar/` loaders with consistent sample indexing.
- Rebuttal add-ons: `data/mmau/`, `data/video_mme/`, `data/rivabench/` loaders.
- Audio/visual preprocessing pipeline:
  - 10-second audio extraction for teachers.
  - 8 uniformly sampled frames for teacher vision input.
  - Standardized resizes/crops; verify 1280x720 main resolution handling.
  - Deterministic sampling seeds + metadata cache (`.jsonl` / `.parquet`).

## 2) Prompt + Teacher Trace Generation
- Teacher prompt templates:
  - `prompts/teacher_audio.md` (Audio Flamingo 3 (think)).
  - `prompts/teacher_visual.md` (Kimi-VL-Thinking).
- Teacher inference harness:
  - `scripts/gen_traces_audio.py`, `scripts/gen_traces_visual.py`.
  - Batch inference + retry/timeout logic.
- Trace filtering:
  - `scripts/filter_traces.py` to keep only samples where both teachers answer correctly.
  - Produce AVRT-20K with exact sample IDs (18,279 train / 945 val).
- Cross-modal aggregation:
  - `prompts/merger.md` for M_agg.
  - `scripts/merge_traces.py` with model backend for Qwen2.5-14B-Instruct and Gemma3-12B-It.
  - Output format enforcement: `<think>...</think><answer>...</answer>`.

## 3) Training Artifacts
- Base student model configs:
  - `configs/model_qwen2.5_omni_3b.yaml` (frozen vision/audio modules).
  - Optional `configs/model_qwen2.5_omni_7b.yaml` (for rebuttal 7B results).
- SFT stage:
  - `configs/sft.yaml` (1 epoch, effective batch size 32, LR 2e-6, cosine, AdamW, wd=0.01, warmup=100).
  - `scripts/train_sft.py` with deterministic seeds and logging.
- RL stage (GRPO):
  - `configs/grpo.yaml` (G=4, epsilon=0.2, beta=0.01, temp=1).
  - Reward functions in `rl/reward.py` with:
    - Format reward (binary).
    - Accuracy reward (string match).
    - Length reward (Gaussian). NOTE: bonus window corrected to **100-200 words** per rebuttal; reconcile tokens vs words.
  - `scripts/train_grpo.py` using full AVQA train set.
- Infrastructure:
  - `configs/deepspeed_zero2.json` (CPU offload, bf16).
  - `scripts/launch_deepspeed.sh` with 4x H100 assumptions.

## 4) Evaluation + Metrics Artifacts
- Unified eval harness: `eval/run_eval.py` with dataset-specific adapters.
- Metrics:
  - Multiple-choice accuracy.
  - Error analysis: IFA (invalid format answers) and logic error rates (per rebuttal table).
- Benchmarks:
  - DailyOmni, OmniBench, AVQA, MMAR (paper).
  - MMAU, Video-MME, RivaBench + video-SALMONN-o1 baseline (rebuttal).
- Output normalization:
  - Strict answer extraction from `<answer>` tag.
  - Consistent handling of A/B/C/D.

## 5) Baselines + Ablations (Repro Tables)
- Baselines:
  - Qwen2.5-Omni-3B zero-shot.
  - AVATAR numbers noted as non-reproducible; track in metadata.
- Ablations:
  - RL-only vs SFT+RL.
  - SFT on audio-only vs visual-only vs A+V aggregated traces.
  - Merger model swap (Qwen2.5-14B-Instruct vs Gemma3-12B-It).
  - Length reward ablation (format+acc vs +length).
  - Modality removal for single-modality performance.
  - Unfiltered trace training (matched count) per rebuttal.
- 7B scale runs for DailyOmni/OmniBench/AVQA/MMAR/MMAU/Video-MME.

## 6) Tables/Figures Reproduction
- `reports/build_tables.py` to output LaTeX-ready tables:
  - Main results, ablations, subsets, rebuttal tables.
- `reports/build_figs.py` to recreate qualitative figure inputs.
- Dataset stats table generator from AVRT-20K metadata.

## 7) Reproducibility Controls
- `configs/seeds.yaml` with fixed RNG for sampling, batching, and generation.
- `artifacts/` manifest with checksums for datasets, traces, and checkpoints.
- `README_repro.md` with exact commands to:
  1) download data,
  2) generate traces,
  3) filter + merge,
  4) train SFT,
  5) train GRPO,
  6) evaluate + produce tables/figures.

## 8) Validation Checklist
- AVRT-20K counts and format compliance match paper.
- RL reward parameters match rebuttal correction.
- DailyOmni/OmniBench/AVQA/MMAR metrics reproduce Table 1 + ablations.
- Rebuttal tables reproduce MMAU/Video-MME/RivaBench + IFA/logic error analysis.

