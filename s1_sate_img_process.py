import os
import pickle

import geopandas as gpd
import numpy as np

from generate_od.sateimgs import area_SateImgs, extract_imgfeats_RemoteCLIP
import open_clip
import torch
from constants import *
from utils import show_regional_image

# Load your regional boundaries
area_shp = gpd.read_file(out_path)

# Set your Esri token (apply at https://developers.arcgis.com/)
esri_token = os.getenv("ESRI_TOKEN", "public_endpoint_without_token")  # seems no need for my use case?
# print(esri_token)

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        regional_images = pickle.load(f)
    print("Loaded regional_images from cache:", len(regional_images))
    show_regional_image(regional_images, 0)  # there should be only 1 region
else:
    print("computing regional_images via area_SateImgs...")
    regional_images = area_SateImgs(area_shp, token=esri_token, num_proc=10)

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(regional_images, f)
    print("Saved regional_images to cache:", len(regional_images))
    show_regional_image(regional_images, 0)

# Load RemoteCLIP model for feature extraction
model_name = "RN50"  # or "ViT-B-32", "ViT-L-14"
vision_model, _, _ = open_clip.create_model_and_transforms(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extract image features using RemoteCLIP
img_features = extract_imgfeats_RemoteCLIP(
    vision_model, model_name, regional_images, device
)

print(f"Image features shape: {img_features.shape}")
print(f"Features per region: {img_features.shape[1]} dimensions")
