# Pipeline Step 2: Model Export

## Overview
This step converts the trained PyTorch model (`.pth`) into the ONNX format (`.onnx`). ONNX (Open Neural Network Exchange) provides a portable and optimized format for inference across different platforms and hardware.

## Prerequisites
-   **Trained Checkpoint**: A `.pth` file from Step 1 (e.g., `work_dirs/stylegan3_celeba/epoch_5.pth`).
-   **Config**: Same configuration used for training.

## Execution
Run the export script:
```bash
bash scripts/step2_export.sh
```
Or manually:
```bash
python src/tools/export.py configs/stylegan3_celeba.py work_dirs/stylegan3_celeba/epoch_5.pth --out work_dirs/stylegan3.onnx
```

## Key Components
-   `src/tools/export.py`: Loads the PyTorch model and runs `torch.onnx.export`.
-   **Dynamic Axes**: The export script configures the ONNX model to accept variable batch sizes.

## Output
-   `work_dirs/stylegan3.onnx`: The exported model file.
