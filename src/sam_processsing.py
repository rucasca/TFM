import cv2
import os
import numpy as np
import time
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Define the function to get the biggest mask (assuming this is part of your logic)
def get_biggest_mask(masks):
    # Assuming the largest mask is the one with the most pixels in the segmentation mask
    biggest_mask = max(masks, key=lambda x: np.sum(x['segmentation']))
    return biggest_mask

# Set the parameters
device = "cpu"  # Use "cuda:0" for GPU if available, otherwise "cpu"
input_folders = ["bulbasaur"]  # Folders of images
output_folder = "new_images"  # Destination folder for processed images
models_route = "../model_checkpoints/"

# Load the SAM model
sam_checkpoint = models_route + "sam_vit_b_01ec64.pth"
model_type = "vit_b"

# Initialize the SAM model and mask generator
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

# Loop through the input folders and process each image
for folder in tqdm(input_folders, desc="Processing folders"):
    # Get all image files in the folder
    file_names = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Wrap file processing in tqdm to show progress
    for file_name in tqdm(file_names, desc=f"Processing images in {folder}", leave=False):
        # Load the image
        image_path = os.path.join(folder, file_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate masks
        start = time.time()
        masks = mask_generator.generate(image)
        end = time.time()
        print(f'Elapsed time for {file_name} = {str((end - start)*1000)} ms')

        # Get the biggest mask (background)
        background = get_biggest_mask(masks)
        segmentation = background['segmentation']

        # Invert the mask to keep only the object (not the background)
        object_mask = np.invert(segmentation)

        # Create an output image where the background is transparent
        output_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        output_image[:, :, :3] = cv2.imread(image_path)  # Copy the BGR values from the original image
        output_image[:, :, 3] = object_mask.astype(np.uint8) * 255  # Alpha channel for transparency

        # Define the output path
        relative_path = os.path.relpath(folder, start=os.path.commonpath(input_folders))
        new_folder_path = os.path.join(output_folder, relative_path)
        os.makedirs(new_folder_path, exist_ok=True)  # Create folder structure if needed

        output_path = os.path.join(new_folder_path, file_name.replace('.jpg', '.png'))  # Save as PNG
        cv2.imwrite(output_path, output_image)
        print(f"Processed and saved image at {output_path}")
