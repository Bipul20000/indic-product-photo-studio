# Kaggle Experiment Context — Indic Product Photo Studio

> **Purpose of this file:** Give Claude Code the missing context from my Kaggle notebook run. Claude Code cannot see my Kaggle session or notebook, so use this document as the record of what I actually ran, what succeeded, what failed, and what the logs showed.
>
> **Evidence source:** `genai.ipynb` (16 cells) and its exported PDF `vertopal.com_genai.pdf` (63 pages). This file records notebook evidence, not assumptions about code that is not visible in those files.

## 1. Project identity and goal

- **Repository:** https://github.com/Bipul20000/indic-product-photo-studio
- **Project:** Indic Product Photo Studio
- **High-level goal:** Generate Indian-context/festive product photographs from product photos, using scene-specific prompts and a Stable Diffusion pipeline; then train and compare a custom festive LoRA against the base model.
- **Main themes/scenes visible in the run:** `diwali`, `holi`, `indian_home`, `raksha_bandhan`, `wedding`, plus the `minimal` scene seen in the initial generation run.
- **Notebook environment:** Kaggle, Python 3.13.15, Linux x86_64, CUDA enabled, Tesla T4 GPU, OpenCV 4.14.0.
- **Repository was cloned repeatedly during the notebook.** Some cells delete and reclone `/kaggle/working/indic-product-photo-studio`, so do not assume files generated in an earlier clone automatically survived a later clean clone unless they were committed, downloaded, or explicitly persisted.

The notebook is an experiment log rather than a clean, linear, reproducible notebook. It includes initial environment checks, base image generation, a sweep, launching the Gradio app, LoRA dataset preparation/training, comparison, and another app launch.

## 2. Kaggle environment setup

The first setup cell:
1. Cloned the repository.
2. Installed it in editable mode with `rembg` and `opencv-python-headless`.
3. Ran `scripts/check_env.py`.

The environment check reported:

| Item | Reported value |
|---|---|
| Python | 3.13.15 |
| OS | Linux 6.18.55+ (x86_64) |
| Kaggle | True |
| Device | cuda |
| OpenCV | 4.14.0 |
| GPU | Tesla T4 |

A later setup installed both `requirements-gpu.txt` and the editable package.

### Dependency warnings

Pip printed dependency-conflict warnings involving packages already present in Kaggle, including `bigframes`, `ydata-profiling`, `dopamine-rl`, `moviepy`, `cuml-cu12`, `cudf-cu12`, `pytensor`, and `cucim-cu12`. The logs specifically mention version conflicts around `numba`, `numpy`, `gym`, `decorator`, and `scikit-image`.

These were warnings from Kaggle's preinstalled environment. The notebook continued into image generation and training; do not treat the warnings alone as proof that the project run failed.

## 3. Wikimedia product-photo collection and Shoot Coach gate

I ran:

```bash
python scripts/fetch_coach_photos.py
```

The script searched Wikimedia Commons using 12 search terms:

- brass diya
- oil lamp diya
- rakhi bracelet
- ceramic coffee mug
- leather handbag
- backpack
- wristwatch
- sneakers shoes
- stainless steel water bottle
- table lamp
- scented candle
- ceramic vase

The output reports:
- **121 total photos collected**.
- Attribution data saved to `docs/coach_eval_attribution.csv`.
- The script then gated **55 originals** with Shoot Coach.
- **37 passed** and **18 were skipped**.
- Gate decisions were saved to `outputs/base_gen/coach_gate.csv`.
- The passing images represented **36 unique product families**.

The run selected 10 products across diverse families for the initial scene generation:

1. `2026.06.05_USSR_Milk_Glass_Table_Lamp_03.jpg`
2. `Ceramic_vase_with_uranium_glaze.jpg`
3. `Cup_of_coffee_on_black_background.jpg`
4. `Diya_set_in_a_rangoli.jpg`
5. `Junghans_Funkuhr_um_2005_2022-01-30_Sp.jpg`
6. `Khalili_Collection_Islamic_Art_sci_0053.3.jpg`
7. `Military_backpack_01.jpg`
8. `Paar_sneakers_Onitsuka_Tiger_objectnr_91283-1-2.jpg`
9. `Red_and_blue_champion_backpack.jpg`
10. `Waterbottle.jpg`

The gate downloaded the `isnet-general-use.onnx` model for background/person-related processing through `rembg`.

**Important distinction:** 121 is the total collected photo count; 55 is the subset evaluated by the coach gate; 37 is the number that passed. These are not interchangeable counts.

## 4. Initial base-model scene generation

I ran:

```bash
python scripts/run_scenes.py --n-products 10 --seeds 0,1
```

The notebook output shows the script loading the base diffusion model:

```text
stable-diffusion-v1-5/stable-diffusion-v1-5
```

The first model load downloaded the Stable Diffusion v1.5 weights and initialized the Diffusers pipeline. The output shows 20/20 inference steps for each generated result.

The run iterated over 10 selected products and scene names including:
- `diwali`
- `raksha_bandhan`
- `holi`
- `wedding`
- `indian_home`
- `minimal`

Two seeds were requested: `0` and `1`.

Thus, the requested configuration corresponds to 10 products × 2 seeds across the scene set used by the script. The logs visibly show per-product/per-scene generation messages, but the notebook does not provide a clean final aggregate count of unique saved files. Some output filenames shown in the logs repeat across scene names, so do not claim a verified total of unique generated files from the log alone.

The initial generated outputs were under `outputs/base_gen/`. The notebook attempted to zip this directory:
- First command `!zip -qr base_gen.zip outputs/base_gen` returned `Nothing to do!`.
- A following command used `zip -qr base_gen.zip . -i outputs/base_gen`; the notebook does not show a verified download or successful archive transfer.

## 5. Sweep experiment

I later ran:

```bash
python scripts/sweep_gen.py
```

and displayed:

```python
from IPython.display import Image
Image("outputs/sweep/sweep.jpg")
```

The sweep visualization was rendered in the notebook. The notebook/PDF does not expose a readable numeric sweep summary in the text output, so do not invent sweep scores, chosen parameters, or a winning configuration from this record.

## 6. Gradio app launches

I launched the app using:

```bash
python app/app.py --share
```

The notebook shows two temporary Gradio share URLs at different points:
- `https://eefc73b74dd15a2915.gradio.live`
- `https://1844cc2d3d31f6ed4c.gradio.live`

These are temporary session URLs and should not be treated as permanent deployment links.

One app run was interrupted with `^C`; the logs show the server/tunnel closing. A later launch also shows the base diffusion pipeline loading successfully before the notebook execution was interrupted. Do not describe the Kaggle share URL as a deployed production service.

## 7. LoRA dataset preparation

I next attempted to prepare and train a festive LoRA.

The notebook first ran:

```bash
python scripts/build_lora_dataset.py --from-metadata docs/lora_dataset_metadata.csv
python scripts/train_lora.py
python scripts/lora_samples.py
```

The dataset rebuild reported:

```text
Repro mode: rebuilding dataset from docs/lora_dataset_metadata.csv (249 images)...
Repro mode complete.
```

The first training attempt failed with:

```text
ModuleNotFoundError: No module named 'scripts'
```

The traceback occurred inside `scripts/train_lora.py` when it tried:

```python
from scripts.prepare_lora_training import prepare_dataset
```

Because training did not create the LoRA weights in that attempt, `scripts/lora_samples.py` reported that `models/festive_lora` was not found and skipped sample generation.

### Fix applied in the notebook

I set the repository root as `PYTHONPATH`:

```bash
%cd /kaggle/working/indic-product-photo-studio
%env PYTHONPATH=/kaggle/working/indic-product-photo-studio
```

Then I reran the training script. The notebook also tried a normal `accelerate launch` invocation that detected more than one GPU/process by default and produced a distributed-launch error. I then used an explicit single-process launch.

The dataset preparation output reported these theme counts:

| Theme | Images |
|---|---:|
| `diwali` | 60 |
| `holi` | 50 |
| `indian_home` | 60 |
| `raksha_bandhan` | 42 |
| `wedding` | 37 |
| **Total** | **249** |

The dataset was copied to:

```text
data/lora_train_hf/train
```

and the image/caption index was written to:

```text
data/lora_train_hf/train/metadata.jsonl
```

The training helper downloaded the official Diffusers training script to:

```text
scripts/vendor/train_text_to_image_lora.py
```

The notebook reports version `0.40.0`.

## 8. Successful LoRA training configuration

The successful explicit training command was:

```bash
accelerate launch --num_processes=1 --mixed_precision fp16 \
  scripts/vendor/train_text_to_image_lora.py \
  --pretrained_model_name_or_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --train_data_dir data/lora_train_hf \
  --image_column image \
  --caption_column text \
  --resolution 512 \
  --random_flip \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --max_train_steps 1500 \
  --learning_rate 1e-4 \
  --lr_scheduler cosine \
  --lr_warmup_steps 100 \
  --rank 16 \
  --mixed_precision fp16 \
  --seed 42 \
  --checkpointing_steps 250 \
  --checkpoints_total_limit 2 \
  --dataloader_num_workers 2 \
  --report_to tensorboard \
  --output_dir models/festive_lora
```

The training run reported:
- Single process (`--num_processes=1`).
- CUDA device.
- Mixed precision `fp16`.
- 512 × 512 training resolution.
- Batch size 1 with gradient accumulation 4.
- LoRA rank 16.
- 1,500 maximum training steps.
- Learning rate `1e-4`, cosine scheduler, 100 warmup steps.
- Seed 42.
- Checkpoint every 250 steps, with at most 2 checkpoints retained.

The log shows checkpoints at 250, 500, 750, 1000, 1250, and 1500 steps. Older checkpoints were removed as the retention limit was reached.

The run reached **1500/1500 steps** in approximately **51 minutes** (the log reports 51:00 at about 2.04 seconds/step). The final displayed step loss was `0.00691`. This is a training step loss from the log, not a validation loss or proof of generalization.

The final LoRA weights were saved at:

```text
models/festive_lora/pytorch_lora_weights.safetensors
```

The notebook then ran:

```bash
python scripts/lora_samples.py
```

and reported:

```text
Sample contact sheet saved to: outputs/lora_samples.jpg
```

### Additional compatibility action

In a later training cell I ran:

```bash
pip uninstall -y torchao
```

The output confirms that `torchao 0.10.0` was uninstalled. This was part of the notebook's compatibility troubleshooting. Do not assume `torchao` is required by the project itself.

## 9. Base vs LoRA comparison

I later recloned the repository, installed the GPU requirements, ran the photo-fetch script again, and executed:

```bash
python scripts/compare_lora.py --lora models/festive_lora
```

The comparison run reports:
- Shoot Coach evaluated 55 originals again.
- 37 passed and 18 were skipped.
- 36 unique product families were found among passing originals.
- **6 products** were selected for the comparison, across diverse families.
- Comparison output was saved to:

```text
outputs/compare/compare.jpg
```

The text output contains this comparison table:

| Scene | Variant | CLIP score | Mean attempts | Person flagged | Mean seconds |
|---|---|---:|---:|---:|---:|
| diwali | base | 0.284 | 1.00 | 0/12 | 9.10 |
| diwali | lora | 0.300 | 1.00 | 0/12 | 7.43 |
| holi | base | 0.260 | 1.00 | 0/12 | 7.69 |
| holi | lora | 0.260 | 1.00 | 0/12 | 7.77 |
| indian_home | base | 0.232 | 1.00 | 0/12 | 7.85 |
| indian_home | lora | 0.246 | 1.00 | 0/12 | 7.83 |
| raksha_bandhan | base | 0.245 | 1.00 | 0/12 | 7.49 |
| raksha_bandhan | lora | 0.267 | 1.00 | 0/12 | 7.59 |
| wedding | base | 0.289 | 1.00 | 0/12 | 7.72 |
| wedding | lora | 0.293 | 1.00 | 0/12 | 7.80 |

The comparison script's sanity check reported:

```text
Sanity check (scale 0.0 scenes):
holi: mean absolute pixel difference = 0.0000
```

The script itself printed an important interpretation note:

> CLIP score measures text alignment only; it does not measure authenticity.

### What these results support

- The LoRA variant had higher CLIP scores for `diwali`, `indian_home`, `raksha_bandhan`, and `wedding`.
- `holi` had the same displayed CLIP score for base and LoRA (`0.260`).
- All rows show mean attempts of `1.00`.
- All rows show `0/12` person-flagged.
- Mean generation time was similar, with some small per-scene differences.

### What these results do NOT establish

- They do not establish that LoRA outputs are more authentic or culturally correct.
- They do not establish statistically significant improvement.
- They do not establish better performance on a broad independent test set.
- The comparison table does not report confidence intervals, standard deviations, or a significance test.
- The displayed CLIP metric is text alignment, not a complete image-quality metric.

Keep these limitations explicit in any report or discussion.

## 10. Repository revisions visible in the notebook

The notebook's later `git log --oneline -3` showed:

```text
90eb10b Phase 7a: per-scene LoRA scale and paired comparison
9be6b8b Add trained LoRA
8a48749 Phase 5: LoRA training scripts
```

An earlier app launch showed:

```text
b2a56d6 Phase 6: Gradio app
2e138eb Phase 4: update person detector import and add category generator support
021e7e8 Phase 3c: sweep-chosen defaults
```

These are the commit hashes/messages recorded by the notebook at those points. The repository's `main` branch advanced during the experiment.

## 11. Notebook cell-by-cell run history

This is the practical sequence represented by the 16 notebook cells:

| Cell | What I did | Result recorded |
|---:|---|---|
| 0 | Kaggle imports and input-directory listing boilerplate | No project result |
| 1 | Clone repo, editable install, install `rembg`/OpenCV, run environment check | Kaggle/T4/CUDA environment confirmed |
| 2 | Install GPU requirements, fetch Wikimedia coach photos, run `run_scenes.py --n-products 10 --seeds 0,1` | 121 photos collected; 37/55 passed gate; initial scene generation ran |
| 3 | Zip `outputs/base_gen` | Failed with “Nothing to do!” |
| 4 | Retry zip with include path | Command issued; no verified archive download in notebook |
| 5 | Run `sweep_gen.py` after fetching coach photos | Sweep completed far enough to produce `outputs/sweep/sweep.jpg` |
| 6 | Display sweep image | Image displayed |
| 7 | Launch Gradio app with share link | Temporary share URL generated |
| 8 | Delete/reclone repo, inspect recent commits, install GPU requirements, launch app | New temporary share URL; app started |
| 9 | Reclone, install training requirements, rebuild LoRA dataset, call `train_lora.py` and samples | Dataset rebuilt; training failed with `No module named 'scripts'` |
| 10 | Set `PYTHONPATH`; rerun training wrapper and sample script | Dataset prepared; initial Accelerate launch hit a distributed-launch issue |
| 11 | Run vendored Diffusers training script directly with `--num_processes=1` | Training run completed to 1500 steps; LoRA weights and sample sheet saved |
| 12 | Uninstall `torchao`; rerun explicit single-process training command | Notebook records the successful 1500-step training and final weights/sample output here as well |
| 13 | Reclone repo, set `PYTHONPATH`, install GPU requirements, uninstall `torchao`, refetch photos, run `compare_lora.py` | Base-vs-LoRA table and comparison image produced |
| 14 | Inspect recent commits and launch Gradio app again | App loaded; final run was interrupted with Ctrl+C |
| 15 | Empty cell | No action |

**Note on cells 11 and 12:** The notebook contains two very similar explicit single-process training commands. The exported output records a complete 1500-step run and final weights. The second cell also begins with `pip uninstall -y torchao` and repeats the training command. When reasoning about exact execution provenance, inspect the full cell outputs rather than assuming these represent two distinct independent trained models.

## 12. Known issues and caveats to preserve

1. **Python import path:** The wrapper `scripts/train_lora.py` initially failed because `scripts` was not importable. Setting `PYTHONPATH` to the repository root was the workaround shown in the notebook.
2. **Accelerate process count:** Accelerate initially detected multiple GPUs/processes by default. The successful command explicitly used `--num_processes=1`.
3. **TorchAO:** `torchao` was uninstalled during troubleshooting.
4. **Kaggle dependency conflicts:** Pip warned about incompatibilities with preinstalled packages. Avoid making broad environment changes without understanding the Kaggle image.
5. **Temporary Gradio links:** They are session links, not permanent hosting.
6. **Notebook persistence:** Cells delete and reclone the repo. Kaggle `/kaggle/working` data may be lost when the session resets unless saved as notebook output or otherwise persisted.
7. **Base generation archive:** The first zip command failed; the notebook does not prove that the generated base images were successfully downloaded from Kaggle.
8. **LoRA comparison interpretation:** CLIP measures text alignment only. Do not call it an authenticity score.
9. **No independent validation result is shown:** The notebook shows training loss and a small comparison table, but no independent validation dataset, confidence intervals, or significance test.
10. **Sweep result details:** The sweep image was displayed, but its numerical findings are not legible in the extracted notebook text. Do not invent sweep outcomes.

## 13. Instructions for Claude Code

When I ask you to continue working on this repository:

- First inspect the current repository state, branch, recent commits, and actual source files. Do not assume the local checkout exactly matches the Kaggle clone at every notebook cell.
- Use this context file as the history of what I ran in Kaggle, but treat the repository code as the source of truth for current implementation details.
- Do not claim to have accessed Kaggle. You only have this written record unless I provide notebook/output files.
- Preserve the experiment details and the caveats above when writing documentation, reports, or methodology.
- Do not silently turn training loss into validation performance or CLIP score into authenticity/quality.
- If a detail is absent here, inspect the code or ask me. Do not fabricate hyperparameters, data splits, evaluation results, or implementation details.
- When changing training or evaluation code, explain which part of this recorded pipeline the change affects and whether it changes comparability with the Kaggle experiment.
- If reproducing training, use the logged settings as the starting point, but verify current package/API compatibility and current repository scripts before running.

## 14. Quick reference

```text
Repository:
https://github.com/Bipul20000/indic-product-photo-studio

Base model:
stable-diffusion-v1-5/stable-diffusion-v1-5

Kaggle GPU:
Tesla T4

Coach photos:
121 collected
55 gated
37 passed
18 skipped

Initial selected products:
10

LoRA dataset:
249 images
diwali: 60
holi: 50
indian_home: 60
raksha_bandhan: 42
wedding: 37

LoRA:
rank 16
resolution 512
max steps 1500
learning rate 1e-4
cosine schedule
warmup 100
fp16
batch size 1
gradient accumulation 4
seed 42
single process

Final weights:
models/festive_lora/pytorch_lora_weights.safetensors

LoRA sample sheet:
outputs/lora_samples.jpg

Base-vs-LoRA comparison:
outputs/compare/compare.jpg

Comparison:
CLIP higher for LoRA in 4/5 scenes
same score for Holi
person flagged: 0/12 in every row
mean attempts: 1.00 in every row
```

---
*Generated from the uploaded Kaggle notebook and its PDF export. This is intended as project handoff context for antigravity, not as a replacement for the source notebook or repository.*
