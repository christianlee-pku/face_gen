# Config for fine-tuning StyleGAN3 on CelebA
# This config is python-based and loaded by src.core.config

# --- Experiment Settings ---
work_dir = "work_dirs/stylegan3_celeba"
total_epochs = 5  # Train for 5 epochs
batch_size = 8
num_workers = 4
lr = 0.002
log_interval = 10 # Log every 10 batches
checkpoint_interval = 1 # Save every epoch
eval_interval = 1 # Evaluate every 1 epoch

# --- Loss Settings ---
r1_gamma = 10.0 # R1 regularization weight

# --- MLflow ---
mlflow = dict(
    tracking_uri = "work_dirs/mlruns",
    experiment_name = "stylegan3_celeba_finetune"
)

# --- Model ---
model = dict(
    generator = dict(
        type = "StyleGAN3Generator",
        z_dim = 512,
        c_dim = 0,
        w_dim = 512,
        img_resolution = 256,
        img_channels = 3
    ),
    discriminator = dict(
        type = "StyleGAN3Discriminator",
        img_resolution = 256,
        img_channels = 3
    )
)

# --- Data ---
dataset = dict(
    type = "CelebADataset",
    data_root = "data/celeba", # Placeholder path
    split = "train"
)

train_pipeline = [
    dict(type="Resize", size=256),
    dict(type="CenterCrop", size=256),
    dict(type="ToTensor"),
    dict(type="Normalize", mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
]

# --- Resume / Fine-tune ---
resume_from = "work_dirs/stylegan3_celeba/epoch_3.pth" # Optional
# load_from = "pretrained_models/stylegan3-r-ffhq-1024x1024.pkl" # Optional