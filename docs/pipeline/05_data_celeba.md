# Data Documentation: CelebA

## Dataset Overview
**CelebA (CelebFaces Attributes Dataset)** is a large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations.

-   **Total Images**: 202,599
-   **Identities**: 10,177
-   **Attributes**: 40 binary labels per image (e.g., Male, Smiling, Young).
-   **Landmarks**: 5 landmark locations per image.

In this project, we primarily use the **Aligned & Cropped** version (`img_align_celeba`), where faces are already centered and resized.

## Data Structure
The `data/celeba/` directory is structured as follows:

```text
data/celeba/
├── img_align_celeba/         # Folder containing 202,599 JPEG images
│   ├── 000001.jpg
│   └── ...
├── list_eval_partition.csv   # Train/Val/Test split definitions
├── list_attr_celeba.csv      # Attribute labels
├── list_bbox_celeba.csv      # Bounding box coordinates (unused for aligned data)
└── list_landmarks_align_celeba.csv # 5-point landmarks
```

## Data Split
The dataset is partitioned according to `list_eval_partition.csv`:
-   **Train**: 162,770 images (Partition 0)
-   **Validation**: 19,867 images (Partition 1)
-   **Test**: 19,962 images (Partition 2)

## Data Processing Pipeline
The data loading logic is implemented in `src/datasets/celeba.py` and `src/datasets/pipelines.py`.

### 1. Loading
-   **Input**: JPEG image path from `img_align_celeba`.
-   **Metadata**: Attributes and split info are loaded into a Pandas DataFrame (`self.data_df`) during initialization.

### 2. Preprocessing
We apply a standard pipeline (`src/datasets/pipelines.py`) to prepare images for StyleGAN3 training:

1.  **Resize**: Images are resized to the target resolution (default: 256x256).
2.  **Random Horizontal Flip**: Applied with probability 0.5 for data augmentation.
3.  **ToTensor**: Converts PIL images to PyTorch tensors `(C, H, W)`.
4.  **Normalization**: Scales pixel values from `[0, 1]` to `[-1, 1]` to match the Tanh output of the generator.
    -   Mean: `[0.5, 0.5, 0.5]`
    -   Std: `[0.5, 0.5, 0.5]`

### 3. Output
The `DataLoader` yields batches of tensors with shape `(Batch_Size, 3, 256, 256)`.
