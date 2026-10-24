# indic-product-photo-studio

An AI-powered product photography tool designed for Indic marketplace sellers to create high-quality studio images through automated segmentation, composition, and generation workflows.

## Setup

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies and package in editable mode:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Verify installation:**
   ```bash
   python scripts/check_env.py
   pytest
   ```
