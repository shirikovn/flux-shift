# SHIFT: подготовка артефактов и запуск экспериментов

Этот файл описывает минимальный workflow запуска эксперементов.

Общая схема выглядит так:

```text
prompt pairs
    |
    +-- один проход FLUX
    |       |
    |       +-- token-wise DiT activations
    |               |
    |               +-- steering vectors
    |               |
    |               +-- mean over tokens -> SVM dataset
    |
    +-- отдельный проход text encoders
            |
            +-- pooled CLIP steering vector

SVM dataset -> classifiers

steering vectors + classifiers + pooled vector
    -> Full SHIFT generation
```
---

## 1. Основные скрипты

```text
collect_activations.py
train_svm.py
collect_pooled_vector.py
full_shift_experiment.py
```

| Скрипт | Назначение |
|---|---|
| `collect_activations.py` | Один проход FLUX: одновременно собирает DiT vectors и SVM dataset |
| `train_svm.py` | Обучает block-specific linear SVM classifiers |
| `collect_pooled_vector.py` | Собирает pooled CLIP vector и target embedding |
| `full_shift_experiment.py` | Запускает итоговую генерацию с pooled + DiT steering |

---

## 2. Минимальный набор конфигов

```text
src/configs/intervention/collect_dit_artifacts.yaml
src/configs/intervention/steer_full_shift.yaml

src/configs/trainer/linear_svm.yaml

src/configs/pipeline/activation_collection.yaml
src/configs/pipeline/pooled_vector_collection.yaml
src/configs/pipeline/steering_experiment.yaml

src/configs/collect_pooled_vector.yaml
src/configs/train_svm.yaml
src/configs/full_shift_experiment.yaml
```

**Для каждой новой концепции пользователь создаёт один dataset-конфиг:**

```text
src/configs/dataset/<dataset_name>.yaml
```

Например:

```text
src/configs/dataset/cyberpunk_20.yaml
src/configs/dataset/glasses_20.yaml
src/configs/dataset/van_gogh_20.yaml
```

---

## 3. Структура артефактов

```text
artifacts/
└── <concept>/
    ├── dit/
    │   ├── vectors/
    │   ├── svm_dataset/
    │   └── metadata.yaml
    ├── svm_training/
    │   └── classifiers/
    └── pooled/
        └── pooled/

outputs/
└── <concept>/
    └── <experiment_name>/
```

`artifacts/` содержит переиспользуемые vectors, features и classifiers.  
`outputs/` содержит изображения и metadata конкретных запусков.

---

# Часть I. Новая концепция

## 4. Dataset с prompt pairs

Создай:

```text
src/configs/dataset/<dataset_name>.yaml
```

Пример для стиля:

```yaml
_target_: src.datasets.prompt_pairs.PromptPairDataset

pairs:
  - name: city
    negative_prompt: "a photograph of a city"
    positive_prompt: "a photograph of a city in cyberpunk style"

  - name: portrait
    negative_prompt: "a portrait photograph of a person"
    positive_prompt: "a portrait photograph of a person in cyberpunk style"

  - name: car
    negative_prompt: "a photograph of a car"
    positive_prompt: "a photograph of a car in cyberpunk style"
```

Пример для объекта:

```yaml
_target_: src.datasets.prompt_pairs.PromptPairDataset

pairs:
  - name: portrait
    negative_prompt: "a portrait photograph of a person"
    positive_prompt: "a portrait photograph of a person wearing glasses"

  - name: office
    negative_prompt: "a photograph of a person working in an office"
    positive_prompt: "a photograph of a person wearing glasses and working in an office"
```

Главное правило:

```text
positive_prompt = negative_prompt + только целевая концепция
```

Хорошо:

```text
a photograph of a car
a photograph of a car in cyberpunk style
```

Плохо:

```text
a photograph of a car
a futuristic neon car at night in cyberpunk style
```

Во втором примере direction будет описывать одновременно:

```text
cyberpunk + futuristic + neon + night
```

Рекомендации:

1. Использовать уникальный `name` для каждой пары.
2. Сохранять одинаковую структуру positive и negative prompts.
3. Менять только target concept.
4. Использовать разные объекты и сцены.
5. Для smoke test достаточно примерно 5 пар.
6. Для основного style/object эксперимента удобно начинать с 20 пар.
7. Для широкой концепции может понадобиться больше примеров.

Короткий `target_prompt` для pooled steering должен описывать только концепцию:

```text
cyberpunk style
glasses
Van Gogh style
red lipstick
a hat
```

---

# Часть II. Подготовка артефактов

Используем placeholders:

```text
<concept>        имя каталога, например cyberpunk
<dataset_name>   имя dataset-конфига без .yaml, например cyberpunk_20
<target_prompt>  короткое описание концепции
```

## 5. Этап 1 — Combined DiT collection

### Команда

```bash
python collect_activations.py \
  dataset=<dataset_name> \
  intervention=collect_dit_artifacts \
  hydra.run.dir=artifacts/<concept>/dit
```

Пример:

```bash
python collect_activations.py \
  dataset=cyberpunk_20 \
  intervention=collect_dit_artifacts \
  hydra.run.dir=artifacts/cyberpunk/dit
```

### Что происходит

Для каждой positive/negative prompt pair выполняется один обычный запуск FLUX.

В hook поступает одна text-token activation:

```text
[1, tokens, channels]
```

Она передаётся двум дочерним collectors.

`MeanDifferenceCollector` вычисляет token-wise mean difference и сохраняет steering vectors.

`PooledSVMDatasetCollector` использует тот же tensor:

```text
pooled_feature = activation.mean(dim=1)
```

Метки:

```text
negative prompt -> 0
positive prompt -> 1
```

При 20 prompt pairs выполняется:

```text
20 pairs x 2 prompts = 40 генераций
```

До объединения выполнялось 80 генераций: 40 для vectors и ещё 40 для SVM dataset.

Количество diffusion steps пока остаётся прежним. Для FLUX.1-schnell это четыре шага на генерацию. Collectors принимают только activation шага `0`, но denoising loop пока выполняется полностью.

### Создаваемые артефакты

```text
artifacts/<concept>/dit/
├── vectors/
│   ├── block_00/
│   │   ├── step_00_raw_difference.pt
│   │   └── step_00_vector.pt
│   ├── ...
│   ├── block_18/
│   │   └── ...
│   └── metadata.yaml
│
├── svm_dataset/
│   ├── block_00/
│   │   ├── step_00_features.pt
│   │   ├── step_00_labels.pt
│   │   └── step_00_samples.yaml
│   ├── ...
│   ├── block_18/
│   │   └── ...
│   └── metadata.yaml
│
└── metadata.yaml
```

Основные outputs:

```text
artifacts/<concept>/dit/vectors
artifacts/<concept>/dit/svm_dataset
```

---

## 6. Этап 2 — обучение SVM classifiers

### Команда

```bash
python train_svm.py \
  trainer.dataset_dir=artifacts/<concept>/dit/svm_dataset \
  hydra.run.dir=artifacts/<concept>/svm_training
```

Пример:

```bash
python train_svm.py \
  trainer.dataset_dir=artifacts/cyberpunk/dit/svm_dataset \
  hydra.run.dir=artifacts/cyberpunk/svm_training
```

### Что происходит

Для каждого блока обучается:

```text
StandardScaler
    ->
SVC(kernel="linear", probability=True)
```

Train/validation split выполняется по `pair_name`.

### Создаваемые артефакты

```text
artifacts/<concept>/svm_training/
└── classifiers/
    ├── block_00/
    │   ├── step_00_classifier.joblib
    │   ├── step_00_metrics.yaml
    │   └── step_00_split.yaml
    ├── ...
    ├── block_18/
    │   └── ...
    └── metadata.yaml
```

Каталог для финальной генерации:

```text
artifacts/<concept>/svm_training/classifiers
```

---

## 7. Этап 3 — сбор pooled CLIP vector

### Команда

```bash
python collect_pooled_vector.py \
  dataset=<dataset_name> \
  'target_prompt=<target_prompt>' \
  hydra.run.dir=artifacts/<concept>/pooled
```

Пример:

```bash
python collect_pooled_vector.py \
  dataset=cyberpunk_20 \
  'target_prompt=cyberpunk style' \
  hydra.run.dir=artifacts/cyberpunk/pooled
```

Кавычки обязательны, когда `target_prompt` содержит пробелы.

Этот этап использует pooled CLIP embeddings и остаётся отдельным, потому что не использует DiT activations.

### Создаваемые артефакты

```text
artifacts/<concept>/pooled/
└── pooled/
    ├── pooled_vector.pt
    ├── target_embedding.pt
    ├── positive_mean.pt
    ├── negative_mean.pt
    └── metadata.yaml
```

---

# Часть III. Финальная генерация

## 8. Пути в `steer_full_shift.yaml`

```yaml
controller:
  vector_directory: ${shift.artifacts_root}/dit/vectors

  regularizer:
    classifier_directory: ${shift.artifacts_root}/svm_training/classifiers

pooled_controller:
  vector_path: ${shift.artifacts_root}/pooled/pooled/pooled_vector.pt
  target_embedding_path: ${shift.artifacts_root}/pooled/pooled/target_embedding.pt
```

При смене концепции достаточно изменить `shift.artifacts_root`.

## 9. Этап 4 — Full SHIFT experiment

```bash
python full_shift_experiment.py \
  shift.artifacts_root=artifacts/<concept> \
  hydra.run.dir=outputs/<concept>/<experiment_name>
```

Пример с параметрами:

```bash
python full_shift_experiment.py \
  shift.artifacts_root=artifacts/cyberpunk \
  shift.dit_gamma=20 \
  shift.pooled_gamma=6 \
  shift.eta_max=4 \
  seed=123 \
  hydra.run.dir=outputs/cyberpunk/gamma_20_pool_6_seed_123
```

Создаётся:

```text
outputs/<concept>/<experiment_name>/
├── .hydra/
├── <case>__baseline.png
├── <case>__dit_only__....png
├── <case>__full_shift__....png
└── experiment_metadata.yaml
```

---

# Часть IV. Параметры

## 10. Combined collection config

Файл:

```text
src/configs/intervention/collect_dit_artifacts.yaml
```

| Параметр | Типичное значение | Назначение |
|---|---:|---|
| `blocks` | `0..18` | Блоки, для которых вызывается collector |
| `steps` | `[0]` | Шаги, активации которых принимает collector |
| `collector.save_dir` | `${hydra:runtime.output_dir}` | Корень combined metadata |
| `vector_collector.save_dir` | `.../vectors` | Каталог steering vectors |
| `vector_collector.tensor_dtype` | `float32` | dtype активаций |
| `vector_collector.normalize` | `true` | Token-wise channel normalization |
| `vector_collector.eps` | `1e-8` | Защита от деления на ноль |
| `svm_collector.save_dir` | `.../svm_dataset` | Каталог SVM features |
| `svm_collector.tensor_dtype` | `float32` | dtype pooled features |

`steps: [0]` фильтрует hook calls, но не останавливает генерацию после первого шага.

## 11. SVM training

| Параметр | Типичное значение | Назначение |
|---|---:|---|
| `dataset_dir` | путь | Combined collection SVM dataset |
| `output_dir` | `${hydra:runtime.output_dir}/classifiers` | Выход classifiers |
| `block_indices` | `0..18` | Обучаемые блоки |
| `source_step` | `0` | Шаг features |
| `validation_fraction` | `0.2` | Доля prompt pairs в validation |
| `random_seed` | `123` | Повторяемость split |
| `c` | `1.0` | SVM regularization parameter |
| `class_weight` | `balanced` | Балансировка классов |
| `standardize` | `true` | StandardScaler |
| `probability` | `true` | Требуется для `predict_proba` |

## 12. Full SHIFT

```yaml
shift:
  artifacts_root: artifacts/cyberpunk
  blocks: [0, 1, 2, 3, 4]
  steps: [0, 1, 2, 3]
  dit_gamma: 20.0
  pooled_gamma: 6.0
  eta_max: 4.0
```

Начальные сетки для новой концепции:

```text
dit_gamma:    10, 20, 30, 50
pooled_gamma: 3, 6, 9
eta_max:      4
```

## 13. Evaluation prompts

Редактируются в:

```text
src/configs/full_shift_experiment.yaml
```

Рекомендуется минимум три типа:

```text
target_present
strongly_target_present
target_absent
```

Пример:

```yaml
experiment:
  cases:
    - name: target_present
      operation: erase
      prompt: "a photograph of a woman in cyberpunk style"

    - name: strongly_target_present
      operation: erase
      prompt: "a cinematic cyberpunk portrait under neon lights"

    - name: target_absent
      operation: erase
      prompt: "a photograph of a woman"
```

---

# Часть V. Полный пример для `glasses`

Предполагается файл:

```text
src/configs/dataset/glasses_20.yaml
```

### 1. Combined DiT collection

```bash
python collect_activations.py \
  dataset=glasses_20 \
  intervention=collect_dit_artifacts \
  hydra.run.dir=artifacts/glasses/dit
```

### 2. SVM training

```bash
python train_svm.py \
  trainer.dataset_dir=artifacts/glasses/dit/svm_dataset \
  hydra.run.dir=artifacts/glasses/svm_training
```

### 3. Pooled vector

```bash
python collect_pooled_vector.py \
  dataset=glasses_20 \
  'target_prompt=glasses' \
  hydra.run.dir=artifacts/glasses/pooled
```

### 4. Full SHIFT

После обновления `experiment.cases`:

```bash
python full_shift_experiment.py \
  shift.artifacts_root=artifacts/glasses \
  shift.dit_gamma=20 \
  shift.pooled_gamma=6 \
  shift.eta_max=4 \
  seed=123 \
  hydra.run.dir=outputs/glasses/reference
```

---

## 14. Проверка артефактов

```bash
find artifacts/<concept>/dit/vectors \
  -name 'step_00_vector.pt' | wc -l

find artifacts/<concept>/dit/svm_dataset \
  -name 'step_00_features.pt' | wc -l

find artifacts/<concept>/svm_training/classifiers \
  -name 'step_00_classifier.joblib' | wc -l

test -f artifacts/<concept>/pooled/pooled/pooled_vector.pt
test -f artifacts/<concept>/pooled/pooled/target_embedding.pt
```

Для полного набора блоков первые три команды должны вывести `19`.
