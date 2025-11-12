import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml
import uuid
from pathlib import Path


def get_augmentation_config(model_version: str) -> Dict: 
    """Load augmentation configuration from YAML file based on model version. 
    Args: model_version: Model version identifier (e.g., 'UAV_RGB', 'UGV_IR') 
    Returns: Dictionary containing augmentation parameters for the model """ 

# Get the directory where this file is located 
config_path = "workspace/autotrain/augmentations" + f"{model_version}.yaml" 
if not config_path.exists():
    logger.warning(
        f"!!!Augmentation config for '{model_version}' not found at {config_path}. !!!"
        f"Using default UAV_RGB config."
    )
    config_path = Path("workspace/autotrain/augmentations/UAV_RGB.yaml")
    
try: 
    with open(config_path, 'r') as f: 
        config = yaml.safe_load(f) 
        logger.info(f"Loaded augmentation config from {config_path}") 
        return config 
    
except Exception as e: 
    logger.error(f"Failed to load augmentation config: {str(e)}") 
    return {}


def generate_unique_dataset_dirs(base_temp_dir="workspace/autotrain/temp"):
    """
    Generate unique subfolders for YOLO labels and frames.

    Args:
        base_temp_dir (str, optional): Base directory to create the dataset folders. Default is 'workspace/autotrain/temp'.

    Returns:
        tuple[str, str]: Paths to (labels_dir, frames_dir)
    """
    unique_id = str(uuid.uuid4())[:8]  # short unique ID
    base_dir = os.path.join(base_temp_dir, unique_id)
    labels_dir = os.path.join(base_dir, "labels")
    frames_dir = os.path.join(base_dir, "frames")

    # Create directories
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    return labels_dir, frames_dir


def combine_yolo_datasets(
    source_dirs,
    output_dir,
    train_ratio=0.8,
    val_ratio=0.18,
    test_ratio=0.02,
    skip_empty_labels=True,
    class_names=None
):
    """
    Combines multiple YOLO-format datasets, splits into train/val/test, and generates a YOLO dataset YAML.

    Args:
        source_dirs (list[str]): List of source directories containing YOLO datasets.
        output_dir (str): Output directory to save combined YOLO dataset.
        train_ratio (float): Train split ratio.
        val_ratio (float): Validation split ratio.
        test_ratio (float): Test split ratio.
        skip_empty_labels (bool): Whether to skip empty label files.
        class_names (list[str], optional): List of class names. Default ['Person'].

    Returns:
        str: Path to the generated YAML file.
    """

    if class_names is None:
        class_names = ['Person']

    # Validate split ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not abs(total_ratio - 1.0) < 1e-6:
        raise ValueError("Split ratios must sum to 1.0.")

    # --- Helper functions ---
    def find_yolo_paths(source_dir):
        """Detects YOLO dataset structure inside `source_dir`."""
        nested_images = os.path.join(source_dir, "images")
        nested_labels = os.path.join(source_dir, "labels")
        if os.path.isdir(nested_images) and os.path.isdir(nested_labels):
            return nested_images, nested_labels
        return source_dir, source_dir  # flat structure

    def collect_file_pairs(source_dirs):
        """Collects valid (image, label) pairs from all source dirs."""
        image_extensions = ('.png', '.jpg', '.jpeg')
        pairs = []

        for idx, source_dir in enumerate(source_dirs):
            if not os.path.isdir(source_dir):
                print(f"⚠️ Skipping missing directory: {source_dir}")
                continue

            print(f"\n📁 Scanning {source_dir}")
            images_dir, labels_dir = find_yolo_paths(source_dir)
            image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(image_extensions)]
            base_names = [os.path.splitext(f)[0] for f in image_files]

            folder_tag = os.path.basename(os.path.normpath(source_dir))
            valid_pairs = 0

            for name in base_names:
                img_path = next((os.path.join(images_dir, f"{name}{ext}") for ext in image_extensions if os.path.exists(os.path.join(images_dir, f"{name}{ext}"))), None)
                if not img_path:
                    continue

                label_path = os.path.join(labels_dir, f"{name}.txt")
                if not os.path.exists(label_path):
                    continue

                if skip_empty_labels and os.path.getsize(label_path) == 0:
                    continue

                unique_name = f"{folder_tag}_{idx}_{name}"
                pairs.append((unique_name, img_path, label_path))
                valid_pairs += 1

            print(f"→ Found {valid_pairs} valid pairs in {source_dir}")

        if not pairs:
            raise RuntimeError("No valid image/label pairs found in any source directory.")

        print(f"\n✅ Total valid pairs collected: {len(pairs)}")
        return pairs

    def create_split_directories(output_dir, splits):
        """Creates the output directory structure for YOLO format based on requested splits."""
        for dtype in ['images', 'labels']:
            for split in splits:
                os.makedirs(os.path.join(output_dir, dtype, split), exist_ok=True)

    def copy_split_files(pairs, split_name, output_dir):
        """Copies files for a given split directly to the output directory."""
        img_target = os.path.join(output_dir, 'images', split_name)
        lbl_target = os.path.join(output_dir, 'labels', split_name)
        print(f"\n📦 Copying {len(pairs)} items to '{split_name}' split...")

        for unique_name, img_path, label_path in tqdm(pairs, desc=f"Copying {split_name}", unit="file"):
            img_ext = os.path.splitext(img_path)[1]
            shutil.copy2(img_path, os.path.join(img_target, f"{unique_name}{img_ext}"))
            shutil.copy2(label_path, os.path.join(lbl_target, f"{unique_name}.txt"))

        print(f"✅ Finished copying to '{split_name}'.")

    # --- Main processing ---
    pairs = collect_file_pairs(source_dirs)

    # Determine which splits are needed
    splits = []
    if train_ratio > 0: splits.append('train')
    if val_ratio > 0: splits.append('val')
    if test_ratio > 0: splits.append('test')
    create_split_directories(output_dir, splits)

    # Split data
    remaining_pairs = pairs
    train_pairs = val_pairs = test_pairs = []

    if 'train' in splits:
        train_size = train_ratio / total_ratio
        train_pairs, remaining_pairs = train_test_split(
            remaining_pairs, test_size=(1 - train_size), random_state=42, shuffle=True
        )

    if 'val' in splits and 'test' in splits:
        test_size = test_ratio / (val_ratio + test_ratio)
        val_pairs, test_pairs = train_test_split(
            remaining_pairs, test_size=test_size, random_state=42, shuffle=True
        )
    elif 'val' in splits:
        val_pairs = remaining_pairs
    elif 'test' in splits:
        test_pairs = remaining_pairs

    # Copy files
    if train_pairs: copy_split_files(train_pairs, 'train', output_dir)
    if val_pairs: copy_split_files(val_pairs, 'val', output_dir)
    if test_pairs: copy_split_files(test_pairs, 'test', output_dir)

    # --- Generate YAML ---
    yaml_dict = {}
    for split in splits:
        yaml_dict[split] = [f"images/{split}"]
    yaml_dict['nc'] = len(class_names)
    yaml_dict['names'] = class_names

    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_dict, f, sort_keys=False)

    print(f"\n🎯 YAML file created at: {yaml_path}")
    return yaml_path
