> prompt: a cinematic cyberpunk photograph of a woman standing near a futuristic train station with neon lights

| Baseline | SHIFT |
| :--- | :--- |
| <img width="512" height="512" alt="strongly_target_present__baseline" src="https://github.com/user-attachments/assets/88239b73-0327-4e8d-afac-786b015503b8" /> | <img width="512" height="512" alt="strongly_target_present__full_shift__svm__erase__gamma_20" src="https://github.com/user-attachments/assets/5ce9ec8d-014c-4d24-a04a-ccea3fa59251" /> |

# FLUX-SHIFT

Inference-time activation steering for `black-forest-labs/FLUX.1-schnell`.

The repository provides four main scripts:

1. `collect_activations.py` — collect DiT steering vectors and an SVM dataset.
2. `train_svm.py` — train block/step-specific SVM classifiers and export SVM-normal vectors.
3. `collect_pooled_vector.py` — collect pooled CLIP steering artifacts.
4. `full_shift_experiment.py` — generate baseline and steered images.

## Minimal setup

Install the project dependencies:

> First do:

```bash
python -m pip install --upgrade pip setuptools wheel
```

```bash
pip install -r requirements.txt
```

The default model configuration uses:

- `black-forest-labs/FLUX.1-schnell`;
- BF16;
- a pinned model revision;
- sequential CPU offloading;
- VAE slicing and tiling.

Make sure the model is accessible from your Hugging Face account.

## Workflow

```text
prompt-pair dataset
        │
        ├── collect_activations.py
        │       ├── token-wise difference vectors
        │       ├── token-mean difference vectors
        │       └── SVM activation dataset
        │
        ├── train_svm.py
        │       ├── linear SVM classifiers
        │       ├── SVM-normal vectors
        │       └── training metrics
        │
        └── collect_pooled_vector.py
                ├── pooled CLIP vector
                └── target embedding

all prepared artifacts
        │
        └── full_shift_experiment.py
                ├── baseline images
                ├── steered images
                ├── per-image records
                └── experiment metadata
```

| Script | Main inputs | Main outputs |
|---|---|---|
| `collect_activations.py` | prompt-pair dataset, blocks, diffusion steps | DiT difference vectors and an SVM dataset |
| `train_svm.py` | SVM dataset, blocks, diffusion steps | classifiers, SVM-normal vectors, metrics and split records |
| `collect_pooled_vector.py` | prompt-pair dataset and target prompt | pooled vector, target embedding and pooled means |
| `full_shift_experiment.py` | prepared artifact root, cases and schedules | baseline/steered images, per-run records and aggregate metadata |

## Prompt-pair datasets

Dataset configurations are stored in:

```text
src/configs/dataset/
```

Example:

```yaml
# src/configs/dataset/cyberpunk_20.yaml

_target_: src.datasets.prompt_pairs.PromptPairDataset

pairs:
  - name: city
    negative_prompt: "a photograph of a city"
    positive_prompt: "a photograph of a city in cyberpunk style"

  - name: portrait
    negative_prompt: "a portrait photograph of a person"
    positive_prompt: "a portrait photograph of a person in cyberpunk style"
```

Pair names must be unique. Positive and negative prompts should differ primarily in the target concept.

# Four-stage artifact workflow

The examples below use:

```text
concept:      cyberpunk
dataset:      cyberpunk_20
target prompt: cyberpunk style
```

Replace these values for another concept.

## Stage 1 — collect DiT artifacts

```bash
python collect_activations.py \
  dataset=cyberpunk_20 \
  intervention=collect_dit_artifacts \
  hydra.run.dir=artifacts/cyberpunk/dit
```

### Main inputs

| Option | Description |
|---|---|
| `dataset` | Dataset config name from `src/configs/dataset/` |
| `intervention.blocks` | Double-stream transformer blocks to collect |
| `intervention.steps` | Diffusion steps to collect |
| `collection.vary_seed_between_pairs` | Use a different seed for each prompt pair |
| `intervention.tensor_dtype` | Saved activation dtype |
| `intervention.normalize` | Normalize saved difference vectors |
| `intervention.eps` | Numerical stability value |
| `seed` | Base random seed |
| `hydra.run.dir` | Artifact output directory |

The default collection configuration uses blocks `0–18` and steps `0–3`, giving 76 block/step locations.

The collection pipeline skips VAE decoding. When only an early subset of diffusion steps is requested, it stops after the last requested step.

### Produced artifacts

For every requested block and step, the script saves:

```text
token-wise raw difference
token-wise normalized vector
token-mean raw difference
token-mean normalized vector
SVM features
SVM labels
sample metadata
```

## Stage 2 — train SVM classifiers

```bash
python train_svm.py \
  trainer.dataset_dir=artifacts/cyberpunk/dit/svm_dataset \
  hydra.run.dir=artifacts/cyberpunk/svm_training
```

SVM training runs on the CPU.

### Main inputs

| Option | Description |
|---|---|
| `trainer.dataset_dir` | SVM dataset produced by Stage 1 |
| `trainer.block_indices` | Blocks for which classifiers are trained |
| `trainer.step_indices` | Diffusion steps for which classifiers are trained |
| `trainer.validation_fraction` | Fraction of prompt pairs used for validation |
| `trainer.random_seed` | Train/validation split seed |
| `trainer.c` | Linear SVM regularization parameter |
| `trainer.class_weight` | SVM class weighting |
| `trainer.standardize` | Apply `StandardScaler` before the SVM |
| `trainer.probability` | Enable `predict_proba`; required for dynamic steering |
| `hydra.run.dir` | Training output directory |

The default configuration trains one classifier for each of 19 blocks and four steps, giving 76 classifiers.

### Produced artifacts

For every requested block and step, the script saves:

```text
classifier.joblib
svm_normal.pt
metrics.yaml
split.yaml
```

The SVM-normal vector is exported in the original activation coordinate system.

## Stage 3 — collect pooled CLIP artifacts

```bash
python collect_pooled_vector.py \
  dataset=cyberpunk_20 \
  'target_prompt=cyberpunk style' \
  hydra.run.dir=artifacts/cyberpunk/pooled
```

Quotes are required when `target_prompt` contains spaces.

### Main inputs

| Option | Description |
|---|---|
| `dataset` | Prompt-pair dataset |
| `target_prompt` | Short prompt that describes the target concept |
| `seed` | Random seed |
| `generation.max_sequence_length` | Text sequence length used during encoding |
| `hydra.run.dir` | Root output directory |

### Produced artifacts

```text
pooled_vector.pt
target_embedding.pt
positive_mean.pt
negative_mean.pt
metadata.yaml
```

The current configuration intentionally stores these tensors under a nested `pooled/pooled/` directory.

## Stage 4 — run generation experiments

```bash
python full_shift_experiment.py \
  shift.artifacts_root=artifacts/cyberpunk \
  hydra.run.dir=outputs/cyberpunk/reference
```

The script generates one baseline for each case, followed by every configured schedule and strength combination.

### Main inputs

| Option | Description |
|---|---|
| `shift.artifacts_root` | Root containing the three prepared artifact groups |
| `shift.artifact_blocks` | Blocks whose vectors/classifiers are loaded |
| `shift.blocks` | Default blocks used by experiment schedules |
| `shift.steps` | Default runtime diffusion steps |
| `shift.vector_type` | Steering-vector parameterization |
| `shift.vector_timing` | Mapping from runtime steps to vector source steps |
| `shift.classifier_timing` | Mapping from runtime steps to classifier source steps |
| `shift.dit_gamma` | Default DiT steering strength |
| `shift.pooled_gamma` | Default pooled steering strength |
| `shift.eta_max` | Maximum dynamic classifier multiplier |
| `experiment.cases` | Prompts and operations |
| `experiment.schedules` | Blocks, steps and enabled method components |
| `experiment.resume` | Resume, overwrite and validation behavior |
| `generation.*` | Resolution, step count, guidance and text length |
| `seed` | Generation seed |
| `hydra.run.dir` | Experiment output directory |

# Artifact structure

A complete artifact directory has this structure:

```text
artifacts/
└── <concept>/
    ├── dit/
    │   ├── .hydra/
    │   ├── vectors/
    │   │   ├── metadata.yaml
    │   │   ├── block_00/
    │   │   │   ├── step_00_raw_difference.pt
    │   │   │   ├── step_00_vector.pt
    │   │   │   ├── step_00_token_mean_raw_difference.pt
    │   │   │   └── step_00_token_mean_vector.pt
    │   │   └── ...
    │   ├── svm_dataset/
    │   │   ├── metadata.yaml
    │   │   ├── block_00/
    │   │   │   ├── step_00_features.pt
    │   │   │   ├── step_00_labels.pt
    │   │   │   └── step_00_samples.yaml
    │   │   └── ...
    │   ├── metadata.yaml
    │   └── run_manifest.yaml
    │
    ├── svm_training/
    │   ├── .hydra/
    │   ├── classifiers/
    │   │   ├── metadata.yaml
    │   │   ├── block_00/
    │   │   │   ├── step_00_classifier.joblib
    │   │   │   ├── step_00_svm_normal.pt
    │   │   │   ├── step_00_metrics.yaml
    │   │   │   └── step_00_split.yaml
    │   │   └── ...
    │   └── run_manifest.yaml
    │
    └── pooled/
        ├── .hydra/
        ├── pooled/
        │   ├── pooled_vector.pt
        │   ├── target_embedding.pt
        │   ├── positive_mean.pt
        │   ├── negative_mean.pt
        │   └── metadata.yaml
        └── run_manifest.yaml
```

Files shown for `block_00/step_00` are repeated for every configured block and step.

A generation experiment produces:

```text
outputs/
└── <concept>/
    └── <experiment_name>/
        ├── .hydra/
        ├── images/
        │   └── *.png
        ├── records/
        │   └── *.yaml
        ├── experiment_metadata.yaml
        └── run_manifest.yaml
```

Each record stores the complete run specification, steering statistics, output paths and image checksum.

# Main generation configuration

The main experiment configuration is:

```text
src/configs/full_shift_experiment.yaml
```

## Artifact blocks and active blocks

```yaml
shift:
  artifacts_root: artifacts/cyberpunk

  # Vectors and classifiers loaded at startup.
  artifact_blocks:
    - 0
    - 1
    - 2
    - 3
    - 4
    - 5
    - 6
    - 7
    - 8
    - 9
    - 10
    - 11
    - 12
    - 13
    - 14
    - 15
    - 16
    - 17
    - 18

  # Default blocks modified by schedules.
  blocks:
    - 0
    - 1
    - 2
    - 3
    - 4

  steps:
    - 0
    - 1
    - 2
    - 3
```

`artifact_blocks` controls which artifact files are loaded. Each experiment schedule can use any subset of those blocks.

## Vector type

```yaml
shift:
  vector_type: tokenwise_difference
```

Supported values:

| Value | Tensor shape | Artifact |
|---|---:|---|
| `tokenwise_difference` | `[tokens, channels]` | `step_XX_vector.pt` |
| `token_mean_difference` | `[channels]` | `step_XX_token_mean_vector.pt` |
| `svm_normal` | `[channels]` | `step_XX_svm_normal.pt` |
| `auto` | `[channels]` or `[tokens, channels]` | Custom paths only |

One-dimensional vectors are broadcast across text-token positions.

## Vector timing

### Shared vector

```yaml
shift:
  vector_timing:
    mode: shared
    source_step: 0
    steps: [0, 1, 2, 3]
```

Every runtime step uses the vector collected at `source_step`.

### Per-step vectors

```yaml
shift:
  vector_timing:
    mode: per_step
    source_step: 0
    steps: [0, 1, 2, 3]
```

Runtime step `t` uses the vector collected at step `t`.

### Custom vectors

```yaml
shift:
  vector_type: auto

  vector_timing:
    mode: custom
    source_step: 0
    steps: [0, 1, 2, 3]

intervention:
  controller:
    vector_paths:
      0:
        default: artifacts/custom/block_00_default.pt
        2: artifacts/custom/block_00_step_02.pt

      1: artifacts/custom/block_01_all_steps.pt
```

For block `0`, the default vector is used at every runtime step except step `2`, which uses its exact override. The direct path for block `1` acts as a wildcard for all runtime steps.

Custom paths may point to vectors collected from any block or step. Routing is determined by the configuration, not by the filename.

## Classifier timing

### Per-step classifiers

```yaml
shift:
  classifier_timing:
    mode: per_step
    source_step: 0
    steps: [0, 1, 2, 3]
```

Runtime step `t` uses classifier `t`.

### Shared classifier

```yaml
shift:
  classifier_timing:
    mode: shared
    source_step: 0
    steps: [0, 1, 2, 3]
```

Every runtime step uses the classifier trained at `source_step`.

Vector timing and classifier timing are independent.

## Cases

Each case defines a prompt and steering operation:

```yaml
experiment:
  cases:
    - name: target_present
      operation: erase
      prompt: >-
        a photograph of a woman standing near a train station
        in cyberpunk style

    - name: target_absent
      operation: erase
      prompt: >-
        a photograph of a woman standing near a train station
```

Supported operations:

```text
erase
add
```

## Schedules

Each schedule selects runtime locations and method components:

```yaml
experiment:
  schedules:
    - name: dit_only
      blocks: ${shift.blocks}
      steps: ${shift.steps}
      strengths:
        - ${shift.dit_gamma}
      use_classifier: true
      use_pooled: false
      pooled_strength: 0.0
      pooled_similarity_mode: raw

    - name: full_shift
      blocks: ${shift.blocks}
      steps: ${shift.steps}
      strengths:
        - ${shift.dit_gamma}
      use_classifier: true
      use_pooled: true
      pooled_strength: ${shift.pooled_gamma}
      pooled_similarity_mode: raw
```

For every case, the pipeline generates:

1. one baseline;
2. one image for each schedule/strength combination.

## Resume behavior

```yaml
experiment:
  resume:
    mode: resume
    verify_images: true
    repair_incomplete: true
```

Modes:

| Mode | Behavior |
|---|---|
| `resume` | Skip valid completed runs and repair incomplete/corrupt outputs |
| `overwrite` | Regenerate every configured run |
| `error` | Fail when an expected output already exists |

A run is complete only when both its image and completion record are valid.

## Generation settings

The default `schnell_512` configuration is:

```yaml
generation:
  width: 512
  height: 512
  num_inference_steps: 4
  guidance_scale: 0.0
  max_sequence_length: 256
  num_images_per_prompt: 1
```

# Command examples

## Collect only step 0

```bash
python collect_activations.py \
  dataset=cyberpunk_20 \
  'intervention.steps=[0]' \
  hydra.run.dir=artifacts/cyberpunk_step0/dit
```

## Collect only selected blocks

```bash
python collect_activations.py \
  dataset=cyberpunk_20 \
  'intervention.blocks=[0,1,2,3,4]' \
  'intervention.steps=[0,1,2,3]' \
  hydra.run.dir=artifacts/cyberpunk_blocks_0_4/dit
```

## Train classifiers for selected blocks and steps

```bash
python train_svm.py \
  trainer.dataset_dir=artifacts/cyberpunk/dit/svm_dataset \
  'trainer.block_indices=[0,1,2,3,4]' \
  'trainer.step_indices=[0,1,2,3]' \
  hydra.run.dir=artifacts/cyberpunk/svm_training_blocks_0_4
```

## Run blocks 0–9

```bash
python full_shift_experiment.py \
  shift.artifacts_root=artifacts/cyberpunk \
  'shift.blocks=[0,1,2,3,4,5,6,7,8,9]' \
  hydra.run.dir=outputs/cyberpunk/blocks_0_9
```

## Run only diffusion step 0

```bash
python full_shift_experiment.py \
  shift.artifacts_root=artifacts/cyberpunk \
  'shift.steps=[0]' \
  hydra.run.dir=outputs/cyberpunk/runtime_step_0
```

## Use token-mean vectors

```bash
python full_shift_experiment.py \
  shift.artifacts_root=artifacts/cyberpunk \
  shift.vector_type=token_mean_difference \
  hydra.run.dir=outputs/cyberpunk/token_mean
```

## Use SVM-normal vectors

```bash
python full_shift_experiment.py \
  shift.artifacts_root=artifacts/cyberpunk \
  shift.vector_type=svm_normal \
  hydra.run.dir=outputs/cyberpunk/svm_normal
```

## Use per-step vectors

```bash
python full_shift_experiment.py \
  shift.artifacts_root=artifacts/cyberpunk \
  shift.vector_timing.mode=per_step \
  hydra.run.dir=outputs/cyberpunk/per_step_vectors
```

## Use one shared step-0 classifier

```bash
python full_shift_experiment.py \
  shift.artifacts_root=artifacts/cyberpunk \
  shift.classifier_timing.mode=shared \
  shift.classifier_timing.source_step=0 \
  hydra.run.dir=outputs/cyberpunk/shared_step0_classifier
```

## Disable the classifier in a schedule

Edit or override the schedule:

```yaml
experiment:
  schedules:
    - name: static_dit
      blocks: ${shift.blocks}
      steps: ${shift.steps}
      strengths: [20.0]
      use_classifier: false
      use_pooled: false
      pooled_strength: 0.0
      pooled_similarity_mode: raw
```

## Change DiT and pooled strengths

```bash
python full_shift_experiment.py \
  shift.artifacts_root=artifacts/cyberpunk \
  shift.dit_gamma=50 \
  shift.pooled_gamma=6 \
  hydra.run.dir=outputs/cyberpunk/dit_50_pool_6
```

## Test several strengths in one schedule

```yaml
experiment:
  schedules:
    - name: strength_sweep
      blocks: ${shift.blocks}
      steps: ${shift.steps}
      strengths:
        - 10.0
        - 20.0
        - 50.0
      use_classifier: true
      use_pooled: true
      pooled_strength: ${shift.pooled_gamma}
      pooled_similarity_mode: raw
```

## Use custom vectors from the command line

```bash
python full_shift_experiment.py \
  shift.artifacts_root=artifacts/cyberpunk \
  shift.vector_type=auto \
  shift.vector_timing.mode=custom \
  'intervention.controller.vector_paths={0:{default:artifacts/custom/block_00.pt,2:artifacts/custom/block_00_step_02.pt},1:artifacts/custom/block_01.pt}' \
  'shift.blocks=[0,1]' \
  hydra.run.dir=outputs/cyberpunk/custom_vectors
```

## Overwrite an existing experiment

```bash
python full_shift_experiment.py \
  shift.artifacts_root=artifacts/cyberpunk \
  experiment.resume.mode=overwrite \
  hydra.run.dir=outputs/cyberpunk/reference
```

## Change resolution

```bash
python full_shift_experiment.py \
  shift.artifacts_root=artifacts/cyberpunk \
  generation.width=1024 \
  generation.height=1024 \
  hydra.run.dir=outputs/cyberpunk/reference_1024
```

# Artifact checks

For a complete default artifact set, these commands should each print `76`:

```bash
find artifacts/<concept>/dit/vectors \
  -name 'step_*_vector.pt' | wc -l

find artifacts/<concept>/dit/vectors \
  -name 'step_*_token_mean_vector.pt' | wc -l

find artifacts/<concept>/dit/svm_dataset \
  -name 'step_*_features.pt' | wc -l

find artifacts/<concept>/svm_training/classifiers \
  -name 'step_*_classifier.joblib' | wc -l

find artifacts/<concept>/svm_training/classifiers \
  -name 'step_*_svm_normal.pt' | wc -l
```

Check pooled artifacts:

```bash
test -f artifacts/<concept>/pooled/pooled/pooled_vector.pt
test -f artifacts/<concept>/pooled/pooled/target_embedding.pt
```

# Limitations

- The current model configuration targets `FLUX.1-schnell`.
- Dynamic SVM steering expects generation batch size 1.
- Dynamic SVM regularization is currently supported only for `operation: erase`.
- Custom vector files are identified by their paths, not by content hashes. Use a new output directory after replacing a custom vector file.
- The repository generates images and detailed run metadata, but it does not yet include a complete quantitative evaluation pipeline.
