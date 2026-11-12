# Indic Product Photo Studio — The Complete Technical Study Guide & Project Blueprint

> **About this Document:**  
> This guide is a complete, step-by-step technical blueprint and study document of the **Indic Product Photo Studio** project. It documents every phase from scratch: why each component was designed, the mathematics and algorithms behind it, the errors and bottlenecks encountered, how they were solved, and the exact Kaggle GPU experimentation results. 
> Written in clear, accessible English for study, presentation, and team onboarding.

---

## Table of Contents

1. [Executive Summary & The Core Problem](#1-executive-summary--the-core-problem)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [Phase 1: The Shoot Coach (Automated Quality Gate)](#3-phase-1-the-shoot-coach-automated-quality-gate)
   - 3.1 The "Garbage In, Garbage Out" Dilemma
   - 3.2 The Four Quality Checks Explained
   - 3.3 Evolution: Phases 1b, 1c, and 1d
   - 3.4 The Joint False-Alarm Budget
4. [Phase 2: Segmentation, Canvas Placement & Compositing](#4-phase-2-segmentation-canvas-placement--compositing)
   - 4.1 Product Cutout (`rembg` with ISNet)
   - 4.2 Mask-Based Quality Refinements (Phases 2a, 2b, 2c)
   - 4.3 Canvas Geometry & Marketplace Presets
   - 4.4 Soft Shadow Synthesis and Realistic Compositing
5. [Phase 3: Generative AI Backgrounds with ControlNet Inpaint](#5-phase-3-generative-ai-backgrounds-with-controlnet-inpaint)
   - 5.1 Why ControlNet Inpainting Beats Simple Image Generation
   - 5.2 Scene Definitions & Prompt Engineering
   - 5.3 Phase 3b: People Filtering & Ghost Object Removal
   - 5.4 Phase 3c: The 6-Configuration Parameter Sweep
6. [Phase 4 & 4b: Custom LoRA Dataset & CLIP Relevance Filter](#6-phase-4--4b-custom-lora-dataset--clip-relevance-filter)
   - 6.1 The Western Bias of Diffusion Models
   - 6.2 Scraping Wikimedia Commons
   - 6.3 Phase 4b: The Multi-Prompt CLIP Filter
   - 6.4 The Final 249-Image Dataset Breakdown
7. [Phase 5: LoRA Training & The Real-World Kaggle Run](#7-phase-5-lora-training--the-real-world-kaggle-run)
   - 7.1 Dataset Hugging Face Formatting (`metadata.jsonl`)
   - 7.2 Vendoring the Official Diffusers Training Script
   - 7.3 Training Hyperparameters on Kaggle Tesla T4
   - 7.4 Kaggle Debugging Log: Errors Encountered & Fixed
   - 7.5 Final Model Weights & Sample Generation
8. [Phase 6: The Interactive Gradio Web Application](#8-phase-6-the-interactive-gradio-web-application)
   - 8.1 Dual-Mode Engine (Real GPU vs CPU Gradient Fallback)
   - 8.2 User Experience & UI Components
9. [Phase 7a: Per-Scene LoRA Scales & Paired Comparative Study](#9-phase-7a-per-scene-lora-scales--paired-comparative-study)
   - 9.1 The Need for Per-Scene Scale Tuning
   - 9.2 The Zero-Scale Fallback (`holi` Scale = 0.0)
   - 9.3 The Paired Comparison Experiment (`compare_lora.py`)
   - 9.4 Empirical Results & Crucial Interpretation Caveats
10. [Repository File Map & How-To Command Reference](#10-repository-file-map--how-to-command-reference)

---

## 1. Executive Summary & The Core Problem

### The Problem
In emerging e-commerce marketplaces across India (such as Amazon India, Flipkart, Meesho, and Etsy India), millions of small and medium enterprise (SME) artisans, crafters, and local shopkeepers sell products online. 

Professional studio photography is expensive (often costing ₹500 to ₹2,000 per product photo) and requires professional equipment, lighting rigs, and studio space. As a result, small sellers take product photos with budget smartphones in suboptimal home environments:
- Uneven or dark room lighting (harsh shadows, high grain).
- Cluttered backgrounds (living room floors, kitchen counters, family members in the frame).
- Poor framing (products chopped off at the borders, or placed too far away).
- Inability to showcase products in authentic, festive Indian settings (Diwali diyas, Holi colors, wedding celebrations, traditional wooden decor).

### The Solution: Indic Product Photo Studio
We designed an automated, AI-driven photography studio tailored specifically to Indian sellers:
1. **Pre-flight Shoot Coach**: Gently checks the seller's photo for basic photography flaws (resolution, blur, lighting, cutoff) and gives simple, actionable advice. If the photo is too flawed for AI, it prevents wasted GPU time.
2. **Precision Cutout & Framing**: Automatically extracts the product with clean edges, places it at the optimal marketplace canvas ratio, and adds natural soft contact shadows.
3. **Culturally Grounded Inpainting**: Uses Stable Diffusion 1.5 guided by ControlNet inpainting and a custom-trained **Festive LoRA** to generate realistic backgrounds matching iconic Indian themes (`diwali`, `raksha_bandhan`, `holi`, `wedding`, `indian_home`, plus clean `white` and `minimal`).
4. **Safety & Quality Guardrails**: Detects and rejects accidental human faces/limbs generated in backgrounds, cleans ghost outlines around products, and allows per-scene fine-tuning.

---

## 2. End-to-End System Architecture

The following diagram illustrates how an image moves through the entire pipeline:

```mermaid
flowchart TD
    A["Raw Smartphone Photo (Seller Upload)"] --> B["Shoot Coach Quality Gate"]
    
    B -->|Fails Quality Checks| C["Show Actionable Advice (e.g., 'Turn on lights', 'Hold steady')"]
    C -->|Optional User Force Override| D["Foreground Segmentation (rembg / ISNet)"]
    B -->|Passes Checks| D
    
    D --> E["Product Mask & BGR Cutout"]
    E --> F["Canvas Placement & Resizing (Preset: 1000x1000, fill: 0.65)"]
    
    F --> G{"Scene Selected?"}
    
    G -->|White Background| H["Pure White Canvas + Soft Ambient Shadow"]
    G -->|Festive Scene (Diwali, Holi, etc.)| I["ControlNet Inpainting Preprocessing"]
    
    I --> J["Apply Prompt Trigger (if effective LoRA scale > 0)"]
    J --> K["ControlNet Inpainting + SD 1.5 + Festive LoRA"]
    
    K --> L{"YOLOv8n Person Detector: Person in Background?"}
    L -->|Yes, and Attempts <= 3| M["Regenerate Background with New Seed (seed + 1000*attempt)"]
    M --> K
    L -->|No Person / Max Attempts Reached| N["Bicubic Upscale to Preset Canvas"]
    
    N --> O["Alpha Blend & Soft Contact Shadow (opacity 0.25)"]
    H --> P["Final Marketplace-Ready Product Image"]
    O --> P
```

---

## 3. Phase 1: The Shoot Coach (Automated Quality Gate)

### 3.1 The "Garbage In, Garbage Out" Dilemma
Early experiments revealed a fundamental truth: **Generative diffusion models cannot fix an image that lacks basic visual data.**
If an input photo is severely out of focus, taken in total darkness, or clipped off at the lens edge, Stable Diffusion and ControlNet hallucinate warped geometries or blurry cutouts. 

Rather than letting sellers wait 10 seconds only to receive a broken image, we built **Shoot Coach**: a lightweight, CPU-based computer vision analyzer that verifies the input photo in under 150 milliseconds.

### 3.2 The Four Quality Checks Explained

#### 1. Resolution Check
- **Goal**: Ensure the photo has enough pixels for e-commerce magnification and marketplace standards.
- **Rule**: Minimum shorter dimension $\ge 480\text{ px}$, minimum total megapixels $\ge 0.5\text{ MP}$.
- **Tip if failed**: *"Photo resolution is too low. Move closer or use a higher camera setting."*

#### 2. Blur Detection (Why standard Laplacian fails)
- **Standard Approach**: Compute the variance of the Laplacian operator over the whole image (`cv2.Laplacian(gray, cv2.CV_64F).var()`).
- **Why it failed in practice**: If a seller took a photo of a blurry leather bag resting on a sharply patterned tablecloth, the high frequencies of the tablecloth falsely fooled the algorithm into thinking the photo was sharp!
- **Our Solution (Phase 1c Tile-Based Blur)**:
  1. Resize the image so its longest dimension is 1024 pixels (standardizing blur scale across different camera sensors).
  2. Split the image into an $8 \times 8$ grid of tiles (or inside the detected product bounding box).
  3. Calculate the Laplacian variance per tile.
  4. Use the 75th percentile of tile variances rather than the global average. This focuses on the sharpest region of the product itself.
- **Tip if failed**: *"Photo looks blurry. Hold your phone steady or tap on the product to focus."*

#### 3. Exposure Checks (Too Dark & Too Bright)
- **Dark Check**:
  - We convert the image to the LAB color space and extract the Lightness ($L$) channel ($0 \le L \le 255$).
  - We calculate the 10th percentile lightness ($L_{10}$). If $L_{10} < 25$, shadow details are completely lost to black crush.
  - Tip: *"The photo is too dark. Turn on lights or face a window."*
- **Bright Check**:
  - We measure the fraction of pixels with lightness $L > 250$.
  - If more than $15\%$ of pixels are blown out to pure white (specular blowouts), texture data is permanently clipped.
  - Tip: *"The photo has harsh glare or overexposed areas. Avoid direct flash or harsh sunlight."*

#### 4. Cut-off / Framing Check
- **Goal**: Make sure the camera didn't slice off a corner of the product.
- If the detected product mask touches the image outer boundary on any side, it is flagged as cut off.
- Tip: *"The product is cut off at the edge. Step back so the entire product is in the frame."*

### 3.3 Evolution: Phases 1b, 1c, and 1d
- **Phase 1b (Honest Dataset Evaluation)**: We gathered 121 diverse Wikimedia images representing real consumer products (bags, mugs, diyas, shoes) and labeled them as ground truth `GOOD` or `BAD`.
- **Phase 1c (Tuned Thresholds & Honest Split)**: We split the evaluation data into separate 50% calibration and 50% validation splits. Thresholds were optimized on calibration only, ensuring zero data leakage.
- **Phase 1d (Robust Edge Handling)**: Replaced raw RGB thresholds with luminance-normalized color variance to eliminate false alarms caused by dark-colored products (like black shoes or dark wood).

### 3.4 The Joint False-Alarm Budget
A common mistake in quality filters is making thresholds so strict that 40% of decent photos are rejected. Sellers get frustrated and abandon the app.
In **Phase 2c**, we implemented a **Joint False-Alarm Budget**:
- Target: False alarm rate on clean photos must be $\le 5\%$.
- The thresholds across blur, dark, bright, and clutter were solved jointly as a constrained optimization problem, maximizing the true defect rejection rate while guaranteeing that $95\%+$ of acceptable seller photos pass on the first click.

---

## 4. Phase 2: Segmentation, Canvas Placement & Compositing

### 4.1 Product Cutout (`rembg` with ISNet)
To place a product in any scene, we must create a clean foreground alpha matte:
- We utilize the `isnet-general-use` neural network through the `rembg` library.
- It returns an RGBA image where the RGB channels contain the original product pixels and the Alpha channel ($0 \text{ to } 255$) represents pixel opacity.
- **Offline / Failure Fallback (Phase 2a)**: If the neural segmentation model encounters corrupt input or is executed in an isolated environment without weights, the code safely falls back to a central bounding-box rectangular mask rather than crashing.

### 4.2 Mask-Based Quality Refinements (Phases 2a, 2b, 2c)
Early in the project, the Shoot Coach analyzed the entire camera rectangle. If the seller had a dark room but a bright brass lamp, the room tripped the "Too Dark" check.
In **Phase 2b**, we bridged Shoot Coach and Segmentation:
- **`assess_with_segmentation(image)`**: Runs segmentation *first*, then calculates blur, luminance, and contrast **strictly inside the product alpha mask**.
- The background room is completely ignored during coach assessment!

### 4.3 Canvas Geometry & Marketplace Presets
Marketplaces require strict aspect ratios:
- `square_1000`: $1000 \times 1000\text{ px}$ (Universal marketplace standard, Amazon & Flipkart default).
- `square_1200`: $1200 \times 1200\text{ px}$.
- `portrait_amazon`: $1500 \times 2000\text{ px}$ ($3:4$ aspect ratio).
- `landscape_banner`: $1600 \times 900\text{ px}$ ($16:9$ promotional banner).

The product is placed on the canvas using a configurable `fill` factor:
$$\text{Target Maximum Dimension} = \min(\text{Canvas Width}, \text{Canvas Height}) \times \text{fill}$$
The product is scaled proportionally and centered on the canvas coordinates:
$$x_{\text{offset}} = \frac{\text{Canvas Width} - \text{Scaled Width}}{2}, \quad y_{\text{offset}} = \frac{\text{Canvas Height} - \text{Scaled Height}}{2}$$

### 4.4 Soft Shadow Synthesis and Realistic Compositing
Placing a cutout directly on a background looks fake—the product appears to float in mid-air like a cheap sticker.
In `studio.compose.composite`, we synthesize natural contact drop shadows:
1. Extract the product alpha mask.
2. Offset the mask slightly downwards and rightwards ($y_{\text{shift}} = +12\text{ px}$, $x_{\text{shift}} = +4\text{ px}$).
3. Apply a heavy Gaussian blur ($\sigma = 21$).
4. Multiply this blurred shadow onto the background with a gentle opacity factor ($0.25$ to $0.30$).
5. Alpha-composite the sharp product cutout over both the background and its new shadow.

---

## 5. Phase 3: Generative AI Backgrounds with ControlNet Inpaint

### 5.1 Why ControlNet Inpainting Beats Simple Image Generation
If you type *"a leather handbag on a table with Diwali diyas"* into standard Midjourney or Stable Diffusion, it will generate a brand new handbag that looks nothing like the seller's product.
If you use standard Inpainting, the boundary between the product and the background is often smeared or eroded.

**The Solution: ControlNet 1.1 Inpaint (`lllyasviel/control_v11p_sd15_inpaint`)**:
1. We keep the seller's original product pixels identical.
2. We provide an inpainting mask that covers only the empty background canvas.
3. We dilate/erode the mask edge by 6 pixels (`erode_px = 6`). This small seam allows Stable Diffusion to draw realistic shadows, lighting reflections, and table contact points right along the product's edge, seamlessly grounding it into the generated scene.

```
+-------------------------------------------------------------+
|                 EMPTY BACKGROUND CANVAS                     |
|                 (ControlNet generates here)                 |
|                                                             |
|                   +---------------------+                   |
|                   |  6px blending seam  |                   |
|                   |  +---------------+  |                   |
|                   |  | ORIGINAL      |  |                   |
|                   |  | PRODUCT       |  |                   |
|                   |  | PIXELS LOCKED |  |                   |
|                   |  +---------------+  |                   |
|                   +---------------------+                   |
|                                                             |
+-------------------------------------------------------------+
```

### 5.2 Scene Definitions & Prompt Engineering (`configs/scenes.yaml`)
We created six core scene environments designed for Indian commerce:
1. **`diwali`**: Warm golden festive lights, clay diyas with burning wicks, fresh marigold flower garlands, colorful rangoli on polished teak wood.
2. **`raksha_bandhan`**: Ornate silk rakhi threads, decorated brass puja thalis, traditional sweets, festive silk fabrics.
3. **`holi`**: Vibrant mounds of dry gulal powders (gulal pink, turmeric yellow, vibrant teal), festive celebration surface.
4. **`wedding`**: Grand Indian wedding mandap backdrop, velvet and chiffon drapes, exotic floral arrangements, warm ambient lighting.
5. **`indian_home`**: Traditional Indian living room interior with warm teak furniture, brass showpieces, and natural warm window illumination.
6. **`minimal`**: Clean studio surface, soft neutral off-white tones, subtle architectural shadows.

### 5.3 Phase 3b: People Filtering & Ghost Object Removal
During initial generations, two recurring artifacts emerged:
1. **Ghost Products**: When the prompt mentioned "shoes" or "vase", the diffusion model occasionally drew a second, duplicate floating shoe or ghostly ring next to the product.
2. **Hallucinated People**: In festival scenes (especially Diwali and Holi), Stable Diffusion often rendered human faces, hands, or distant people standing in the background.

#### How we fixed both in Phase 3b:
- **Negative Prompt Hardening**:
  Every prompt is combined with a strict negative prompt:
  ```
  "person, people, woman, man, girl, boy, child, face, portrait, hands, duplicated product, extra object, ring around the product, frame around the product, text, letters, watermark, logo, distorted, deformed, low quality, blurry, oversaturated"
  ```
- **Foreground Isolation**:
  We appended `, the product stands alone in the foreground` to every scene prompt.
- **YOLOv8n Automated Person Detector (`studio.vision.people`)**:
  - After generation, the background is scanned with Ultralytics YOLOv8n (`conf=0.35`, `min_box_area=0.01`).
  - **The `ignore_mask` Rule**: If more than 50% of a detected box lies inside the product mask, it is ignored (so products like shoes, mannequins, or leather bags cannot accidentally trigger a false alarm).
  - **Automatic Retry Loop**: If a human is detected in the generated background, the pipeline automatically discards the image and regenerates with a new seed:
    $$\text{seed}_{\text{retry}} = \text{seed}_{\text{initial}} + 1000 \times \text{attempt}$$
    Retries up to 3 times (1 to 4 attempts total).

### 5.4 Phase 3c: The 6-Configuration Parameter Sweep
To find the ideal visual balance, we executed a parameter grid sweep across 4 diverse products and 2 scenes:
- `controlnet_scale` $\in \{1.0, 0.7, 0.5\}$
- `fill` $\in \{0.80, 0.65\}$

**Findings & Decision**:
- `fill = 0.80` occupied too much of the frame. Round products (like vases and mugs) created a circular mask that forced ControlNet into generating tight, halo-like "ghost shells" around the edges.
- `fill = 0.65` gave the model sufficient breathing room to render rich, convincing perspective depth.
- `controlnet_scale = 0.5` allowed the background context to render naturally while firmly locking the product geometry.
- **Defaults adopted**: `controlnet_scale = 0.5` and `fill = 0.65` for all generative scenes (`fill = 0.80` retained only for pure white background e-commerce cutouts).

---

## 6. Phase 4 & 4b: Custom LoRA Dataset & CLIP Relevance Filter

### 6.1 The Western Bias of Diffusion Models
Stable Diffusion 1.5 was trained primarily on western internet data (LAION-5B). When asked for festive settings, it frequently rendered:
- Christmas fairy lights instead of traditional earthen diyas.
- Modern western wedding banquet halls instead of Indian mandaps with marigold drapes.
- Generic powder paint instead of authentic Holi gulal.

To solve this without retraining a massive model from scratch, we trained a **Low-Rank Adaptation (LoRA)** adapter.

### 6.2 Scraping Wikimedia Commons (`scripts/build_lora_dataset.py`)
We gathered thousands of candidate images from Wikimedia Commons under Creative Commons licenses using targeted search terms and category members:
- `diwali`: "Diwali", "Diyas", "brass diya oil lamp", "rangoli Diwali".
- `raksha_bandhan`: "Rakhi", "Raksha Bandhan", "rakhi thali".
- `holi`: "Holi", "gulal", "Holi powder bowls".
- `wedding`: "Weddings in India", "wedding stage decoration India", "wedding mandap".
- `indian_home`: "Interior of houses in India", "traditional Indian home interior".

**Quality Heuristics Applied**:
- Removed corrupted images.
- Minimum short dimension $\ge 600\text{ px}$.
- Minimum mean luminance $\ge 40$ (discarding pitch-black photos).
- Per-author cap $\le 8$ images (preventing one photographer's camera style from dominating).
- Perceptual Hash (`pHash`) deduplication: Discarded any image with Hamming distance $\le 6$ to an existing image.

### 6.3 Phase 4b: The Multi-Prompt CLIP Filter
Raw scraping pulled in unwanted clutter (tourist selfies, stone temple ruins, museum sketches, street maps).
In **Phase 4b**, we added an automated semantic relevance filter using `openai/clip-vit-base-patch32`:

1. For each candidate image, we construct positive prompts and negative prompts.
   - Shared Negatives: *"a photo of a person or crowd", "a close-up portrait of a face", "a scanned document", "a sketch", "a street with cars"*.
   - Theme Positives: Distinct descriptive phrases for diyas, rangolis, rakhis, mandaps, etc.
2. We pass the image and prompt embeddings through CLIP and compute softmax probabilities across the positive and negative prompt set:
   $$p_{\text{positive}} = \sum_{i \in \text{Positives}} \text{Softmax}(\text{Logits})_i$$
3. **Threshold & Ranking**:
   - Only images with $p_{\text{positive}} \ge 0.60$ were accepted.
   - The images were ranked by $p_{\text{positive}}$, and the top 60 were selected.
   - **Threshold Relaxation**: If a theme yielded $< 40$ images, the threshold automatically relaxed from $0.60 \rightarrow 0.50 \rightarrow 0.40$.

### 6.4 The Final 249-Image Dataset Breakdown

| Theme | Raw Pool | Passed CLIP | Kept Count | Final Threshold | Mean $p_{\text{pos}}$ | Trigger Word |
|---|---:|---:|---:|---:|---:|---|
| `diwali` | 250 | 236 | **60** | 0.60 | 1.000 | `indiandiwali` |
| `raksha_bandhan` | 186 | 42 | **42** | 0.60 | 0.990 | `indianrakhi` |
| `holi` | 106 | 50 | **50** | 0.60 | 0.936 | `indianholi` |
| `wedding` | 79 | 37 | **37** | 0.40 (relaxed) | 0.862 | `indianwedding` |
| `indian_home` | 114 | 64 | **60** | 0.60 | 0.973 | `indianhome` |
| **Total** | **735** | **429** | **249** | — | **0.952** | — |

Every image was paired with a descriptive caption:
$$\text{Caption} = \text{"<trigger>, <best positive CLIP caption>, <theme description>"}$$

---

## 7. Phase 5: LoRA Training & The Real-World Kaggle Run

### 7.1 Dataset Hugging Face Formatting (`metadata.jsonl`)
In `scripts/prepare_lora_training.py`, the dataset is formatted into the Hugging Face `imagefolder` standard:
```
data/lora_train_hf/
└── train/
    ├── diwali_001.jpg
    ├── diwali_002.jpg
    ...
    ├── wedding_037.jpg
    └── metadata.jsonl
```
Each line of `metadata.jsonl` contains:
```json
{"file_name": "diwali_001.jpg", "text": "indiandiwali, Diwali decorations with lit diyas, festive Diwali decoration with diyas, marigold flowers and warm lights"}
```

### 7.2 Vendoring the Official Diffusers Training Script
Rather than writing an unstable custom PyTorch training loop, `scripts/train_lora.py` vendors the official, tested Hugging Face Diffusers script:
`examples/text_to_image/train_text_to_image_lora.py`.
- It dynamically checks the installed `diffusers` version and downloads the matching script into `scripts/vendor/`.
- It executes training via Hugging Face `accelerate`.

### 7.3 Training Hyperparameters on Kaggle Tesla T4

| Hyperparameter | Value | Rationale |
|---|---|---|
| **Base Model** | `stable-diffusion-v1-5/stable-diffusion-v1-5` | SD 1.5 architecture matching ControlNet inpaint |
| **Hardware** | Nvidia Tesla T4 GPU (16 GB VRAM) | Cloud GPU environment |
| **Precision** | `fp16` (Mixed Precision) | Fits within 16 GB VRAM without OOM |
| **Resolution** | $512 \times 512$ | Native resolution of SD 1.5 |
| **LoRA Rank ($r$)** | 16 | Sufficient capacity for textures without overfitting |
| **Batch Size** | 1 | Maximizes memory headroom |
| **Gradient Accumulation** | 4 steps | Simulates effective batch size of 4 |
| **Learning Rate** | $1 \times 10^{-4}$ | Standard stable AdamW learning rate for LoRA |
| **LR Scheduler** | Cosine with 100 warmup steps | Smooth decay to prevent catastrophic forgetting |
| **Max Training Steps** | 1,500 steps | ~24 epochs over 249 images |
| **Checkpoint Interval** | Every 250 steps (keep last 2) | Safeguard against session disconnection |

### 7.4 Kaggle Debugging Log: Errors Encountered & Fixed

#### Error 1: `ModuleNotFoundError: No module named 'scripts'`
- **What happened**: When running `python scripts/train_lora.py`, Python could not resolve internal imports like `from scripts.prepare_lora_training import prepare_dataset` because the Kaggle working directory was not automatically on the Python module search path.
- **Kaggle Fix**: Ran `%env PYTHONPATH=/kaggle/working/indic-product-photo-studio`.
- **Permanent Codebase Fix (Phase 7a)**: We added automatic path resolution to the top of every script in `scripts/`:
  ```python
  _REPO_ROOT = Path(__file__).resolve().parent.parent
  if str(_REPO_ROOT) not in sys.path:
      sys.path.insert(0, str(_REPO_ROOT))
  _SRC_DIR = _REPO_ROOT / "src"
  if str(_SRC_DIR) not in sys.path:
      sys.path.insert(0, str(_SRC_DIR))
  ```
  Now every script runs seamlessly from any working directory!

#### Error 2: Accelerate Multi-Process Detection Error
- **What happened**: Running plain `accelerate launch` caused Accelerate to erroneously attempt a multi-GPU distributed launch on Kaggle's single GPU container, causing an immediate crash.
- **Fix**: Updated command to explicitly specify `--num_processes=1`:
  ```bash
  accelerate launch --num_processes=1 --mixed_precision fp16 scripts/vendor/train_text_to_image_lora.py ...
  ```

#### Error 3: TorchAO Dependency Conflicts
- **What happened**: Kaggle's pre-installed environment had conflicting versions of `torchao 0.10.0` that threw warnings with PyTorch 2.5.
- **Fix**: Executed `pip uninstall -y torchao` in the notebook before training.

### 7.5 Final Model Weights & Sample Generation
The training ran for **1,500 steps** in **51 minutes** (~2.04 seconds per step).
- The final step loss was **0.00691**.
- The resulting weights were compiled into:
  `models/festive_lora/pytorch_lora_weights.safetensors` (**12.1 MB**).
- We ran `scripts/lora_samples.py` to generate `outputs/lora_samples.jpg`, visually verifying that the trigger words correctly produced authentic Indian decor.

---

## 8. Phase 6: The Interactive Gradio Web Application

### 8.1 Dual-Mode Engine (Real GPU vs CPU Gradient Fallback)
To ensure the application can be developed and demonstrated on laptops without an expensive GPU, `app/app.py` implements a smart dual-mode engine:

```python
cuda_available = torch.cuda.is_available()
is_demo_mode = fake_generator or not cuda_available

if is_demo_mode:
    # CPU Mode: Generates smooth deterministic gradients matching the theme palette
    generator = FakeBackgroundGenerator()
else:
    # GPU Mode: Loads SD 1.5, ControlNet Inpaint, and Festive LoRA on CUDA
    generator = BackgroundGenerator()
```

### 8.2 User Experience & UI Components
The web app (`app/app.py`) provides:
1. **Product Upload**: Drag-and-drop any smartphone photo.
2. **"Check My Photo" Button**: 
   - Instant CPU evaluation (150ms).
   - Displays verdict (`OK` or `RETAKE`).
   - Shows human-friendly photography tips.
   - Renders interactive data table of all 4 quality checks.
   - Displays visual **Product Mask Overlay** (alpha mask highlighted in translucent red at 40% opacity).
3. **"Generate" Button**:
   - Runs full segmentation and generative inpainting for the chosen scene.
   - Gated by Shoot Coach (won't generate if photo fails, unless user checks *"Generate even if Shoot Coach says retake"*).
4. **"Generate All Scenes" Button**:
   - Generates all 6 scenes in batch using the same seed.
   - Displays results in an interactive gallery with one-click full-resolution downloads.

---

## 9. Phase 7a: Per-Scene LoRA Scales & Paired Comparative Study

### 9.1 The Need for Per-Scene Scale Tuning
During testing of the trained LoRA, an important visual nuance was observed:
- On `diwali`, `raksha_bandhan`, `wedding`, and `indian_home`, the LoRA produced dramatically superior Indian aesthetic details compared to base SD 1.5.
- However, on `holi`, base SD 1.5 was already proficient at generating colorful chalk powders. Applying the festive LoRA at full 0.8 scale sometimes caused color oversaturation.

### 9.2 The Zero-Scale Fallback (`configs/lora_scales.yaml`)
In **Phase 7a**, we introduced granular, per-scene LoRA scale control:

```yaml
diwali: 0.8
raksha_bandhan: 0.8
holi: 0.0          # 0.0 automatically disables LoRA and falls back to pure base model
wedding: 0.8
indian_home: 0.8
```

- **Supported Adapter API**: We integrated Diffusers' modern adapter control:
  ```python
  if scale <= 0.0:
      pipe.disable_lora()
  else:
      pipe.enable_lora()
      pipe.set_adapters(["default"], [float(scale)])
  ```
  *(Replacing deprecated `cross_attention_kwargs={"scale": ...}`)*.
- **Trigger Gating (`apply_trigger`)**:
  The trigger word (e.g. `indiandiwali`) is prepended to the scene prompt **only if the effective scale for that scene is $> 0$**. When scale is $0.0$, the trigger word is omitted, ensuring a pure base-model run.

### 9.3 The Paired Comparison Experiment (`scripts/compare_lora.py`)
To rigorously evaluate the LoRA against the base model, we created `scripts/compare_lora.py`:
- 6 diverse products selected through Shoot Coach gating across 36 families.
- Evaluated across 5 scenes (`diwali`, `raksha_bandhan`, `holi`, `wedding`, `indian_home`) at seeds `0` and `1`.
- Every product $\times$ scene $\times$ seed was generated **twice** with the exact same seed:
  1. Once with pure Base SD 1.5 (LoRA disabled, trigger omitted).
  2. Once with Festive LoRA at the configured per-scene scale.
- Saved paired contact sheet to `outputs/compare/compare.jpg`.

### 9.4 Empirical Results & Crucial Interpretation Caveats

#### The Empirical Results from the Kaggle Run:

| Scene | Variant | Effective Scale | CLIP Alignment Score | Mean Attempts | Person Flagged | Mean Latency |
|---|---|---:|---:|---:|---:|---:|
| `diwali` | Base | 0.0 | 0.284 | 1.00 | 0 / 12 | 9.10s |
| `diwali` | **LoRA** | **0.8** | **0.300 (+0.016)** | 1.00 | 0 / 12 | 7.43s |
| `holi` | Base | 0.0 | 0.260 | 1.00 | 0 / 12 | 7.69s |
| `holi` | **LoRA** | **0.0** | **0.260 (Identical)** | 1.00 | 0 / 12 | 7.77s |
| `indian_home`| Base | 0.0 | 0.232 | 1.00 | 0 / 12 | 7.85s |
| `indian_home`| **LoRA** | **0.8** | **0.246 (+0.014)** | 1.00 | 0 / 12 | 7.83s |
| `raksha_bandhan`| Base | 0.0 | 0.245 | 1.00 | 0 / 12 | 7.49s |
| `raksha_bandhan`| **LoRA** | **0.8** | **0.267 (+0.022)** | 1.00 | 0 / 12 | 7.59s |
| `wedding` | Base | 0.0 | 0.289 | 1.00 | 0 / 12 | 7.72s |
| `wedding` | **LoRA** | **0.8** | **0.293 (+0.004)** | 1.00 | 0 / 12 | 7.80s |

#### Scale 0.0 Sanity Check:
$$\text{Mean Absolute Pixel Difference for Holi (Base vs LoRA)} = \mathbf{0.0000}$$
*(Confirms that setting scale to 0.0 completely turns off the LoRA and produces an exact pixel-perfect match to the base model!)*

#### Safety & Stability Metrics:
- **Person Flag Rate**: **0 / 12** across all scenes (the YOLO filter + negative prompts prevented all accidental humans).
- **Mean Attempts**: **1.00** (clean generations on the first seed pass).

#### ⚠️ Crucial Interpretation Caveats to Remember:
1. **CLIP score measures text-image alignment, NOT aesthetic authenticity**: A higher CLIP score indicates the generated image correlates more strongly with the text description (*"festive Diwali decoration with diyas and warm lights"*). It does not objectively prove cultural authenticity.
2. **No independent statistical significance claim**: The sample of 6 products across 2 seeds provides directional validation; it is not a large-scale randomized trial.
3. **Training loss is not validation accuracy**: The final training loss of $0.00691$ indicates the model learned the 249 training samples effectively; real-world quality must always be evaluated visually.

---

## 10. Repository File Map & How-To Command Reference

### Repository Architecture

```
indic-product-photo-studio/
├── configs/
│   ├── coach_eval_exclude.txt   # File exclusions for test evaluation sets
│   ├── lora_scales.yaml         # Per-scene LoRA scale factors (holi: 0.0, others: 0.8)
│   ├── lora_themes.yaml         # Theme definitions, prompts, triggers, CLIP keywords
│   └── scenes.yaml              # Prompt templates & shared negative prompt
├── data/
│   ├── coach_eval/              # Benchmark evaluation images (good vs bad)
│   ├── lora_train/              # Raw scraped images & text captions (249 images)
│   └── lora_train_hf/           # Prepared HuggingFace imagefolder dataset
├── docs/
│   ├── coach_eval_attribution.csv
│   └── lora_dataset_metadata.csv # Full metadata index of all 249 LoRA images
├── models/
│   └── festive_lora/
│       └── pytorch_lora_weights.safetensors # Trained LoRA adapter (12 MB)
├── outputs/
│   ├── base_gen/                # Base generation outputs & overview.jpg
│   ├── compare/                 # Paired comparison outputs & compare.jpg
│   └── sweep/                   # 6-configuration parameter sweep outputs
├── src/studio/
│   ├── coach/                   # Shoot Coach diagnostics, blur, exposure algorithms
│   ├── compose/                 # Preset canvas dimensions, resizing, shadow synthesis
│   ├── generate/                # ControlNet inpainting, LoRA loading, background engine
│   ├── segment/                 # rembg wrapper with offline fallback
│   ├── vision/                  # Ultralytics YOLOv8n person detector & ignore mask
│   ├── app_logic.py             # Pure functional backend for Gradio handlers
│   └── pipeline.py              # Main unified orchestrator (run_pipeline)
├── scripts/
│   ├── build_lora_dataset.py    # Wikimedia scraper & CLIP relevance filter
│   ├── check_env.py             # CUDA/GPU/Platform verification
│   ├── compare_lora.py          # Paired Base vs LoRA evaluation & metrics
│   ├── fetch_coach_photos.py    # Downloads sample product photos from Wikimedia
│   ├── lora_samples.py          # Text-to-image contact sheet generator
│   ├── prepare_lora_training.py # Formats data into Hugging Face imagefolder
│   ├── run_scenes.py            # Generates scenes for diverse products
│   ├── sweep_gen.py             # Hyperparameter grid sweep
│   ├── train_lora.py            # Accelerate launch wrapper for Diffusers
│   └── vendor/                  # Vendored train_text_to_image_lora.py (v0.40.0)
├── app/
│   └── app.py                   # Gradio web application
├── tests/                       # Complete pytest suite (77 unit & integration tests)
├── kaggle_context.md            # Experimental log and execution trace from Kaggle T4
└── PROJECT_STUDY_GUIDE.md       # This comprehensive project blueprint
```

---

### Command Cheat Sheet

#### 1. Run All Tests Locally (CPU, Fast)
```bash
pytest
```
*Executes all 77 unit tests in ~30 seconds without requiring a GPU or network.*

#### 2. Launch the Gradio Web App Locally (Demo CPU Mode)
```bash
python app/app.py --fake-generator
```
*Opens interactive UI at `http://127.0.0.1:7860` with instant gradient backgrounds.*

#### 3. Launch Gradio App on GPU with Public Share Link
```bash
python app/app.py --share
```

#### 4. Run the Paired Base vs. LoRA Comparison
```bash
python scripts/compare_lora.py --lora models/festive_lora --n-products 6 --seeds 0,1
```

#### 5. Re-run Training Dry-Run
```bash
python scripts/train_lora.py --dry-run
```
*Outputs the exact Accelerate training command without executing training.*

---
*Created as the definitive technical blueprint and study reference for the Indic Product Photo Studio project.*
