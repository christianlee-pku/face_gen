import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image

def create_dummy_celeba(data_root="data/celeba", num_images=10, img_size=(218, 178)):
    """
    Creates a dummy CelebA dataset structure for testing the pipeline.
    
    Args:
        data_root (str): The root directory where the dummy dataset will be created.
        num_images (int): Number of dummy images and corresponding entries to create.
        img_size (tuple): (height, width) of dummy images.
    """
    if os.path.exists(data_root):
        print(f"Removing existing dummy data at {data_root}...")
        shutil.rmtree(data_root)
    
    os.makedirs(data_root, exist_ok=True)
    
    img_align_celeba_dir = os.path.join(data_root, "img_align_celeba")
    os.makedirs(img_align_celeba_dir, exist_ok=True)
    
    print(f"Creating {num_images} dummy images in {img_align_celeba_dir}...")
    filenames = []
    for i in range(1, num_images + 1):
        fname = f"{i:06d}.jpg"
        filenames.append(fname)
        
        # Create random image (RGB)
        img_array = np.random.randint(0, 255, (img_size[0], img_size[1], 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(os.path.join(img_align_celeba_dir, fname))
        
    print("Creating dummy annotation files...")
    
    # --- list_attr_celeba.csv ---
    # Header from actual CelebA: 40 attributes
    # Example: '5_o_Clock_Shadow', 'Arched_Eyebrows', ..., 'Young'
    attr_names = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
        'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry',
        'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
        'Gala_Attire', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
        'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
        'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
        'Wearing_Necktie', 'Young'
    ]
    attr_cols = ['image_id'] + attr_names
    attr_data = []
    for fname in filenames:
        row = [fname] + list(np.random.choice([-1, 1], len(attr_names))) # -1 or 1 for attributes
        attr_data.append(row)
    df_attr = pd.DataFrame(attr_data, columns=attr_cols)
    df_attr.to_csv(os.path.join(data_root, "list_attr_celeba.csv"), index=False)
    
    # --- list_eval_partition.csv ---
    # image_id, partition (0: train, 1: val, 2: test)
    part_cols = ['image_id', 'partition']
    part_data = []
    for i, fname in enumerate(filenames):
        # Roughly 80% train, 10% val, 10% test
        if i < num_images * 0.8:
            part = 0
        elif i < num_images * 0.9:
            part = 1
        else:
            part = 2
        part_data.append([fname, part])
    df_part = pd.DataFrame(part_data, columns=part_cols)
    # CelebA eval partition file is space separated, header=False
    df_part.to_csv(os.path.join(data_root, "list_eval_partition.csv"), index=False, header=False, sep=' ')

    # --- list_bbox_celeba.csv ---
    # image_id x_1 y_1 width height (top-left x, top-left y, width, height)
    bbox_cols = ['image_id', 'x_1', 'y_1', 'width', 'height']
    bbox_data = []
    for fname in filenames:
        # Generate plausible random bounding boxes within image dimensions
        x1 = np.random.randint(0, img_size[1] // 4)
        y1 = np.random.randint(0, img_size[0] // 4)
        w = np.random.randint(img_size[1] // 2, img_size[1] * 3 // 4)
        h = np.random.randint(img_size[0] // 2, img_size[0] * 3 // 4)
        row = [fname, x1, y1, w, h]
        bbox_data.append(row)
    df_bbox = pd.DataFrame(bbox_data, columns=bbox_cols)
    df_bbox.to_csv(os.path.join(data_root, "list_bbox_celeba.csv"), index=False)

    # --- list_landmarks_align_celeba.csv ---
    # image_id lefteye_x lefteye_y righteye_x righteye_y nose_x nose_y leftmouth_x leftmouth_y rightmouth_x rightmouth_y
    lm_cols = [
        'image_id', 'lefteye_x', 'lefteye_y', 'righteye_x', 'righteye_y', 
        'nose_x', 'nose_y', 'leftmouth_x', 'leftmouth_y', 'rightmouth_x', 'rightmouth_y'
    ]
    lm_data = []
    for fname in filenames:
        # Generate random landmark coordinates within plausible ranges
        row = [fname] + list(np.random.randint(img_size[0] // 5, img_size[0] * 4 // 5, 10))
        lm_data.append(row)
    df_lm = pd.DataFrame(lm_data, columns=lm_cols)
    df_lm.to_csv(os.path.join(data_root, "list_landmarks_align_celeba.csv"), index=False)

    print(f"Dummy CelebA dataset with {num_images} images created successfully at {data_root}.")

if __name__ == "__main__":
    create_dummy_celeba(num_images=10)
