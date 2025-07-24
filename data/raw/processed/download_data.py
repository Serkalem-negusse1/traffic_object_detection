#This script will download and prepare the COCO dataset (or subset traffic data) via TensorFlow Datasets.
import tensorflow_datasets as tfds
import os

def download_coco_data(split="train", target_dir="data/raw/coco"):
    """
    Download COCO 2017 dataset using TensorFlow Datasets and save images & annotations locally.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Load COCO dataset with annotations
    dataset = tfds.load('coco/2017', split=split, shuffle_files=False, download=True)

    print(f"Downloading {split} split of COCO dataset...")
    for i, sample in enumerate(tfds.as_numpy(dataset)):
        image = sample['image']
        image_id = sample['image/id']
        annotations = sample['objects']

        # Save images
        image_path = os.path.join(target_dir, f"{image_id}.jpg")
        with open(image_path, 'wb') as f:
            f.write(image)

        # Save annotations in your preferred format (e.g., JSON, XML, TXT)
        # For now, just print first 5 to verify
        if i < 5:
            print(f"Sample {i} annotations:", annotations)
    
    print("Download and saving completed.")

if __name__ == "__main__":
    download_coco_data()
#Note:
#TensorFlow Datasets loads data in memory as tf.Tensor. To save images as JPEG files, you might need to use PIL or cv2 to write the image arrays. This code is simplified for demonstration; we will improve it next.