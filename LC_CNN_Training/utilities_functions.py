# -*- coding: utf-8 -*-

import numpy as np 
#import nemaktis as nm 
from PIL import Image
from pathlib import Path
#import json

# save an image from an np array
def save_np_img(img,path,img_name):
    # If last dim is 1 (grayscale), remove the singleton dimension
    if img.shape[2] == 1:
        img_to_save = img[:, :, 0]
    else:
        img_to_save = img
    # Convert to uint8 if necessary (common image dtype)
    if img_to_save.dtype != np.uint8:
        img_to_save = (img_to_save * 255).astype(np.uint8)  # if img is float in [0,1]
    # Create PIL image
    if img_to_save.ndim == 2:  # grayscale
        pil_img = Image.fromarray(img_to_save, mode='L')
    else:  # color
        pil_img = Image.fromarray(img_to_save, mode='RGB')
    # Define path
    output_path = Path(path) / img_name
    # Create folder if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save image
    pil_img.save(output_path)

# return an np array representing the simulated images from a given polariser and analyser angle
def get_img_np(viewer,polariser_angle=0,analyser_angle=90,grayscale=False):
    viewer.analyser_angle = analyser_angle;
    viewer.polariser_angle = polariser_angle;
    viewer.grayscale = grayscale;
    viewer.update_image()
    img = viewer.get_image()
    return img
# return a dictionary contain all properties from a given viewer, it aso allows you to add custom properties to the dictionary e.g. ImgName, FieldType
# example: utilities_functions.get_properties(viewer,ImgName = "90twist", FieldType="twist")
def get_properties(viewer, **kwargs):
    dic = {}

    # Add all kwargs directly
    dic.update(kwargs)

    # Add fixed properties from viewer, with defaults if attribute missing
    dic.update({
        "polariser": getattr(viewer, "polariser", None),
        "analyser": getattr(viewer, "analyser", None),
        "upper_waveplate": getattr(viewer, "upper_waveplate", None),
        "lower_waveplate": getattr(viewer, "lower_waveplate", None),
        "polariser_angle": getattr(viewer, "polariser_angle", None),
        "analyser_angle": getattr(viewer, "analyser_angle", None),
        "upper_waveplate_angle": getattr(viewer, "upper_waveplate_angle", None),
        "lower_waveplate_angle": getattr(viewer, "lower_waveplate_angle", None),
        "angle_lock": getattr(viewer, "angle_lock", None),
        "intensity": getattr(viewer, "intensity", None),
        "NA_condenser": getattr(viewer, "NA_condenser", None),
        "n_tiles_x": getattr(viewer, "n_tiles_x", None),
        "n_tiles_y": getattr(viewer, "n_tiles_y", None),
        "grayscale": getattr(viewer, "grayscale", None),
        "z_focus": getattr(viewer, "z_focus", None),
        "NA_objective": getattr(viewer, "NA_objective", None),
    })

    return dic
    