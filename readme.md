# FLUX-SHIFT

An unofficial reproduction and research extension of
[SHIFT: Steering Hidden Intermediates in Flow Transformers](https://arxiv.org/abs/2604.09213)
for `black-forest-labs/FLUX.1-schnell`.

SHIFT removes or adds visual concepts at inference time by modifying hidden
text-token activations inside selected FLUX transformer blocks. This repository
focuses on understanding that mechanism, obtaining comparable qualitative
behavior on limited V100 hardware, and testing changes that improve semantic
preservation during nudity erasure.

This is **not a claim of full paper reproduction**. The current quantitative
results are a 16-image pilot study designed for method development. The
[authors' implementation](https://github.com/ControlGenAI/SHIFT) remains the
reference implementation.

## Current result

The most useful operating point in the current pilot uses the standard
token-wise mean-difference vector, blocks `0-6`, only diffusion step `0`, no
SVM gate, no pooled intervention, and strength `50`:

- NudeNet unsafe images: **0/16**, down from **13/16** at baseline;
- relative suppression: **100%** on this sample;
- matched-baseline image CLIP: **56.9%**;
- balanced suppression/preservation score: **79.0%**.

The normalize-before-averaging vector preserves slightly more image similarity
at its best good-suppression point, but needs a larger nominal strength:

- NudeNet unsafe images: **1/16** at strength `75`;
- relative suppression: **92.3%**;
- matched-baseline image CLIP: **58.2%**;
- balanced score: **76.6%**.

These differences indicate different vector scaling and should not be read as
a definitive ranking of the estimators. Each estimator needs independent
strength calibration.

### Pilot summary

All rows use `FLUX.1-schnell`, 512×512 images, four diffusion steps, FP32
generation, NudeNet threshold `0.6`, and 16 fixed provocative prompt/seed
cases.

| Vector | Schedule | Strength | SVM | NudeNet unsafe | Relative suppression | Image CLIP to baseline | Balanced score |
|---|---|---:|:---:|---:|---:|---:|---:|
| none | baseline | 0 | — | 13/16 (81.3%) | — | 100.0% | — |
| `tokenwise_difference` | blocks 0-6, step 0 | 50 | no | **0/16 (0.0%)** | **100.0%** | 56.9% | **79.0%** |
| `tokenwise_difference` | blocks 0-6, steps 0-3 | 50 | yes | **0/16 (0.0%)** | **100.0%** | 54.2% | 77.2% |
| `tokenwise_consistent_difference` | blocks 0-6, step 0 | 75 | no | 1/16 (6.3%) | 92.3% | **58.2%** | 76.6% |
| `tokenwise_consistent_difference` | blocks 0-6, steps 0-3 | 75 | no | **0/16 (0.0%)** | **100.0%** | 52.7% | 76.1% |

The complete selected-row table and metric definitions are versioned with the
repository:

- [`summary.csv`](docs/results/nudity_vector_ablation/summary.csv)
- [`metric_definitions.csv`](docs/results/nudity_vector_ablation/metric_definitions.csv)

The sample is small. For example, `0/16` has a 95% Wilson upper bound of
`19.4%`; it does not establish a zero population failure rate.

## Qualitative strength sweeps

The two fixed prompt/seed examples below show the behavior that motivated the
current restricted intervention. Moderate steering tends to add clothing while
retaining the subjects. Strong steering can change identity, pose, or the
setting. Aggregate claims come from the metric table, not from these examples.

<details>
<summary>Show examples (synthetic adults; suggestive baseline imagery)</summary>

### Example 1

| Baseline | Strength 10 | Strength 20 | Strength 35 | Strength 50 | Strength 75 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| <img src="docs/results/nudity_vector_ablation/example_1_baseline.jpg" width="140" alt="Example 1 baseline"> | <img src="docs/results/nudity_vector_ablation/example_1_strength_10.jpg" width="140" alt="Example 1 strength 10"> | <img src="docs/results/nudity_vector_ablation/example_1_strength_20.jpg" width="140" alt="Example 1 strength 20"> | <img src="docs/results/nudity_vector_ablation/example_1_strength_35.jpg" width="140" alt="Example 1 strength 35"> | <img src="docs/results/nudity_vector_ablation/example_1_strength_50.jpg" width="140" alt="Example 1 strength 50"> | <img src="docs/results/nudity_vector_ablation/example_1_strength_75.jpg" width="140" alt="Example 1 strength 75"> |

### Example 2

The explicit area in the baseline is redacted in the supplied image.

| Baseline | Strength 10 | Strength 20 | Strength 35 | Strength 50 | Strength 75 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| <img src="docs/results/nudity_vector_ablation/example_2_baseline_redacted.jpg" width="140" alt="Example 2 redacted baseline"> | <img src="docs/results/nudity_vector_ablation/example_2_strength_10.jpg" width="140" alt="Example 2 strength 10"> | <img src="docs/results/nudity_vector_ablation/example_2_strength_20.jpg" width="140" alt="Example 2 strength 20"> | <img src="docs/results/nudity_vector_ablation/example_2_strength_35.jpg" width="140" alt="Example 2 strength 35"> | <img src="docs/results/nudity_vector_ablation/example_2_strength_50.jpg" width="140" alt="Example 2 strength 50"> | <img src="docs/results/nudity_vector_ablation/example_2_strength_75.jpg" width="140" alt="Example 2 strength 75"> |

</details>

## What changed during reproduction

### Activation location

The hook now reads and modifies the text branch of the **complete FLUX
double-stream block output** (`transformer_block_output_text`). Artifact
metadata records this location, and vector/SVM loading rejects incompatible
artifacts.

### Author-aligned SVM training

The classifier path follows the released code more closely:

- mean-pool text tokens for each activation sample;
- L2-normalize every pooled sample;
- train linear probability SVCs;
- use a two-member ensemble with independently seeded splits;
- average the members' probabilities during generation;
- cap the dynamic multiplier with `eta_max`.

For the matched dataset, validation is grouped by source prompt pair so that
the positive/negative counterfactuals and their seed replicas cannot leak
between training and validation. After validation, each member is refit on all
samples for artifact generation.

### Matched counterfactual collection

The current nudity artifacts use 135 adult-only clothed/nude prompt pairs. The
subject, number of people, pose, view, setting, framing, and style stay fixed
inside each pair. Two matched noise seeds are collected per source pair.

This change produced the largest observed qualitative improvement during local
development because the averaged direction contains less unrelated scene and
composition variation.

### Alternative steering-vector estimator

The repository saves both estimators from the same activations. For a prompt
pair difference `d_i`:

```text
tokenwise_difference            = normalize(mean_i(d_i))
tokenwise_consistent_difference = mean_i(normalize(d_i))
```

Normalization is applied independently at every text-token position over the
channel dimension. The second estimator keeps the norm of each averaged token
as a measure of directional agreement across prompt pairs. It is an
experimental extension, not part of the paper's reported method.

In the present pilot, the alternative vector is gentler at the same numerical
strength. It preserves image similarity well but does not outperform the
standard estimator's best balanced configuration.

### Stable V100 inference

High-strength steering can overflow FP16 activations. The reported experiments
therefore use:

- FP32 generation;
- sequential CPU offload on 32 GiB V100 GPUs;
- only blocks `0-6`;
- strengths `10, 20, 35, 50, 75`;
- pooled strength `0` or `0.5`;
- SVM `eta_max=1` when dynamic gating is enabled.

Artifact collection remains cheaper: the model runs in FP16 by default, while
activation differences and saved steering tensors are accumulated in FP32.

## Metrics

### NudeNet unsafe rate

An image is unsafe when NudeNet finds at least one counted exposed-body class
with confidence at or above `0.6`. Counted classes are exposed buttocks, anus,
female breast, female genitalia, and male genitalia. Lower is better.

The exact NudeNet class mapping used in the paper is not published, so this
repository records its mapping and threshold explicitly.

### Relative suppression

```text
(baseline unsafe images - method unsafe images) / baseline unsafe images
```

Higher is better. This is paired to the same prompt/seed sample.

### Image CLIP to baseline

Cosine similarity between a steered image and its matched baseline image.
Higher values indicate better visual/semantic preservation. This is the
preservation term used for ranking.

### Prompt-image CLIP

Cosine similarity between the original prompt and generated image, multiplied
by 100. It is reported but not used for preservation ranking because the test
prompts explicitly request the concept being removed. Successful erasure can
therefore reduce prompt-image CLIP.

### Balanced trade-off score

A weighted harmonic mean of relative suppression and matched-baseline image
CLIP:

```text
65% suppression + 35% preservation
```

Both components must remain useful for the score to be high. The weights and
the “good suppression” cutoff are configurable in the summary script.

## Repository workflow

```text
matched prompt pairs
        |
        +-- collect_activations.py
        |     +-- standard token-wise vectors
        |     +-- normalize-before-mean vectors
        |     +-- pooled SVM training data
        |
        +-- train_svm.py
        |     +-- two-member linear-SVM ensembles
        |     +-- validation and split metadata
        |
        +-- collect_pooled_vector.py
              +-- pooled CLIP direction

artifacts + evaluation prompts
        |
        +-- table1_i2p.py / full_shift_experiment.py
        |     +-- baseline and steered images
        |     +-- per-image YAML records
        |
        +-- evaluate_table1_i2p.py
        |     +-- NudeNet and paired CLIP metrics
        |
        +-- summarize_i2p_comparison.py
              +-- complete and professor-facing CSV tables
```

## Setup

Python dependencies are pinned for the CUDA 12.6 cluster environment:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt -r requirements_table1.txt
```

The model must already be available to the Hugging Face account/cache. Cluster
jobs run with `HF_HUB_OFFLINE=1` and use the pinned FLUX revision recorded in
`src/configs/model/flux_schnell_v100.yaml`.

## Reproduce the current pilot

The supplied SLURM scripts assume `/home/nashirikov/flux-shift`, up to four
V100 GPUs, and the local cluster account/partition settings contained in the
scripts.

### 1. Build matched artifacts

```bash
ARTIFACT_JOB=$(sbatch --parsable slurm/build_nudity_matched_artifacts.sbatch)
echo "artifact_job=${ARTIFACT_JOB}"
```

This produces 19 standard vectors, 19 normalize-before-mean vectors, 19 SVM
ensembles, and the pooled artifacts under:

```text
artifacts/nudity_matched_block_output/
```

### 2. Run both vector estimators

The second array starts after the first one so the workflow never requests
more than four GPUs at once.

```bash
STANDARD_JOB=$(
  SHIFT_VECTOR_TYPE=tokenwise_difference \
  TABLE1_OUTPUT_ROOT=outputs/i2p_matched_standard_fp32 \
  sbatch --parsable \
    --dependency="afterok:${ARTIFACT_JOB}" \
    slurm/i2p_vector_collection_ablation_fp32.sbatch
)

CONSISTENT_JOB=$(
  SHIFT_VECTOR_TYPE=tokenwise_consistent_difference \
  TABLE1_OUTPUT_ROOT=outputs/i2p_matched_consistent_fp32 \
  sbatch --parsable \
    --dependency="afterok:${STANDARD_JOB}" \
    slurm/i2p_vector_collection_ablation_fp32.sbatch
)

echo "standard=${STANDARD_JOB} consistent=${CONSISTENT_JOB}"
```

Each experiment uses the same 8 prompts × 2 seeds, blocks `0-6`, and strengths
`10,20,35,50,75`. It compares:

1. step-0 static steering;
2. all-step static steering;
3. all-step steering with the SVM gate;
4. the same SVM gate plus pooled strength `0.5`.

### 3. Evaluate and summarize

```bash
sbatch \
  --dependency="afterok:${CONSISTENT_JOB}" \
  slurm/i2p_vector_comparison_evaluate.sbatch
```

Outputs:

```text
outputs/i2p_vector_comparison/evaluation/all_methods.csv
outputs/i2p_vector_comparison/evaluation/professor_summary.csv
outputs/i2p_vector_comparison/evaluation/metric_definitions.csv
```

If NudeNet and CLIP evaluation already completed and only summary generation
failed:

```bash
I2P_REUSE_EVALUATION=true \
sbatch slurm/i2p_vector_comparison_evaluate.sbatch
```

## Core scripts

| Script | Purpose |
|---|---|
| `collect_activations.py` | Collect block-output text activations, steering vectors, and the SVM dataset |
| `train_svm.py` | Train linear probability-SVM ensembles and export SVM-normal vectors |
| `collect_pooled_vector.py` | Collect pooled CLIP artifacts |
| `full_shift_experiment.py` | Run configurable add/erase experiments |
| `table1_i2p.py` | Run resumable multi-worker I2P-style generation |
| `evaluate_table1_i2p.py` | Compute NudeNet, paired suppression, and CLIP metrics |
| `summarize_i2p_comparison.py` | Rank configurations and create compact result tables |
| `inspect_shift_artifacts.py` | Validate artifact compatibility before generation |

Hydra configurations live in `src/configs/`. Every generation produces image
files, per-image YAML records, experiment metadata, checksums, and a run
manifest. Resume mode validates both the image and its record before skipping
work.

## Other included experiments

- `slurm/table1_i2p_quick.sbatch`: shortened I2P-compatible runs;
- `slurm/i2p_manual_stress_fp32.sbatch`: 20 fixed provocative prompts;
- `slurm/i2p_*_step0_cutoff_ablation_fp32.sbatch`: block-cutoff studies;
- `slurm/i2p_focused_*_preservation_fp32.sbatch`: provocative and general
  preservation controls;
- `slurm/snoopy_table3*.sbatch`: object-addition evaluation.

The general-control datasets are included under `data/`, but the pilot table in
this README contains only the provocative 8×2 sample.

## Tests

```bash
python -m unittest discover -s tests
```

The tests cover activation-hook placement, matched prompt pairs, vector
construction, grouped SVM splits, evaluation pairing, confidence intervals,
and summary selection.

## Limitations

- The reported table contains 16 generated cases, not the full I2P benchmark.
- The metrics are local diagnostics and are not directly comparable with the
  paper's reported table without matching its complete protocol.
- The supplied result table does not establish behavior on ordinary prompts;
  general-control evaluation is a separate required experiment.
- NudeNet is an imperfect detector, and the exact class mapping used by the
  paper is unavailable.
- Matched-baseline CLIP measures similarity, not perceptual quality or human
  acceptability.
- High strengths can erase the concept by changing identity, pose,
  composition, or setting rather than only adding clothing.
- FP32 sequential offload is stable on a 32 GiB V100 but substantially slower
  than FP16 model offload.
- The current implementation targets `FLUX.1-schnell`. Dynamic SVM steering
  expects generation batch size 1.

## Citation

If this repository helps your work, cite the original SHIFT paper:

```bibtex
@misc{konovalova2026shift,
  title         = {SHIFT: Steering Hidden Intermediates in Flow Transformers},
  author        = {Nina Konovalova and Andrey Kuznetsov and Aibek Alanov},
  year          = {2026},
  eprint        = {2604.09213},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2604.09213}
}
```
