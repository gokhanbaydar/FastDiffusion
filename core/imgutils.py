from diffusers.utils import load_image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from PIL import Image
import numpy as np
import cv2
import torch


def canny_image(image):
    if isinstance(image, str):
        image = load_image(image)
    assert isinstance(image, Image.Image)
    # TODO Resize Image
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image).resize((1024, 1024))
    return image


def get_depth_map(image, feature_extractor=None, depth_estimator=None):
    if isinstance(image, str):
        image = load_image(image)
    assert isinstance(image, Image.Image)
    if feature_extractor is None:
        feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
    if depth_estimator is None:
        depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")

    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image
