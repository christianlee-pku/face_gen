# Pipeline Step 1: Training

## Overview
The training step fine-tunes a StyleGAN3 model on the CelebA dataset. This process involves loading the pre-trained weights (if available) and updating them based on the new dataset to generate realistic face images.

## Prerequisites
-   **Environment**: Conda environment `face-gen-pipeline` activated.
-   **Data**: CelebA dataset prepared in `data/celeba/`.
-   **Config**: Configuration file at `configs/stylegan3_celeba.py`.

## Execution
Run the training script:
```bash
bash scripts/step1_train.sh
```
Or manually:
```bash
python src/tools/train.py configs/stylegan3_celeba.py
```

## Key Components
-   `src/tools/train.py`: Main entry point.
-   `src/models/generator.py`: StyleGAN3 Generator architecture.
-   `src/models/discriminator.py`: StyleGAN3 Discriminator architecture.
-   `src/datasets/celeba.py`: Dataset loader.

## Output
-   Checkpoints saved to `work_dirs/stylegan3_celeba/`.
-   Logs tracked via MLflow (if configured).
