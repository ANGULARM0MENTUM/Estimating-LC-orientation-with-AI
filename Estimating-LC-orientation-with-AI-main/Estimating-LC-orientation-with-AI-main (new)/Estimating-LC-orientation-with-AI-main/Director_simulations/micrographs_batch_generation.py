# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 13:12:15 2025

@author: JohnC

Batch Generator for Liquid Crystal Micrographs
Reads recipes from 'generation_recipes.json'
"""

import sys
import numpy as np 
import nemaktis as nm 
from pathlib import Path
import json
import utilities_functions 
import DirectorField_NEW as DirectorField

# --- Configuration ---
RECIPE_FILE = "recipes.json"
DATABASE_DIR = "Samples"
MESH_LENGTHS = (20, 20, 5)
MESH_DIMENSIONS = (40, 40, 10)
PROPAGATION_METHOD = "dtmm(1)"

# --- Helper: Function Builder ---
def build_combined_functions(components, unique_name_suffix=""):
    """
    Parses a list of component dictionaries and returns summed (nx, ny, nz) functions.
    components: list of dicts {class, params, weight, shift}
    """
    
    # Store the individual function triplets and their weights/shifts
    func_list = []
    
    for idx, comp in enumerate(components):
        cls_name = comp["class"]
        params = comp["params"]
        weight = comp.get("weight", 1.0)
        shift = comp.get("shift", [0.0, 0.0])
        
        # Instantiate the class dynamically
        if hasattr(DirectorField, cls_name):
            Cls = getattr(DirectorField, cls_name)
            # Give it a unique name to avoid internal caching collisions if any
            instance = Cls(name=f"{cls_name}_{unique_name_suffix}_{idx}", **params)
            func_list.append({
                "funcs": instance.as_init_funcs(), # (nx, ny, nz)
                "weight": weight,
                "shift": shift
            })
        else:
            print(f"Warning: Class {cls_name} not found in DirectorField.")

    # Define the summed lambda functions
    # Note: We must be careful with closures in loops.
    # We define a master function that iterates over our prepared list.
    
    def master_nx(x, y, z):
        res = 0.0
        for item in func_list:
            fn = item["funcs"][0]
            sx, sy = item["shift"]
            w = item["weight"]
            res += w * fn(x - sx, y - sy, z)
        return res

    def master_ny(x, y, z):
        res = 0.0
        for item in func_list:
            fn = item["funcs"][1]
            sx, sy = item["shift"]
            w = item["weight"]
            res += w * fn(x - sx, y - sy, z)
        return res

    def master_nz(x, y, z):
        res = 0.0
        for item in func_list:
            fn = item["funcs"][2]
            sx, sy = item["shift"]
            w = item["weight"]
            res += w * fn(x - sx, y - sy, z)
        return res

    return master_nx, master_ny, master_nz

# --- Main Execution ---

# 1. Setup Directories
Path(DATABASE_DIR).mkdir(exist_ok=True)
json_path = Path(DATABASE_DIR) / "info.json"
metadata_json = Path(DATABASE_DIR) / "Samples_metadata.json"

# 2. Load Recipes
if not Path(RECIPE_FILE).exists():
    print(f"Error: {RECIPE_FILE} not found. Run generate_recipes.py first.")
    sys.exit()

with open(RECIPE_FILE, 'r') as f:
    recipes = json.load(f)

print(f"Loaded {len(recipes)} recipes. Starting batch generation...")

# 3. Iterate through recipes
for recipe_idx, recipe in enumerate(recipes):
    
    # --- A. Handle Sample Indexing (Atomic per loop to be safe) ---
    if json_path.exists():
        with open(json_path, 'r') as file:
            try:
                database_data = json.load(file)
            except json.JSONDecodeError:
                database_data = {"number_of_samples": 0}
    else:
        database_data = {"number_of_samples": 0}

    current_id = str(database_data["number_of_samples"])
    print(f"Processing Sample ID: {current_id} (Recipe {recipe_idx+1}/{len(recipes)})")
    
    # Update counter immediately
    database_data["number_of_samples"] += 1
    with open(json_path, 'w') as file:
        json.dump(database_data, file, indent=4)

    # --- B. Director Field Generation ---
    nfield = nm.DirectorField(
        mesh_lengths=MESH_LENGTHS, 
        mesh_dimensions=MESH_DIMENSIONS
    )
    
    try:
        if recipe["type"] == "2D":
            # Linear combination of 2D fields
            nx, ny, nz = build_combined_functions(recipe["components"], unique_name_suffix=current_id)
            nfield.init_from_funcs(nx, ny, nz)
            
        elif recipe["type"] == "interpolation":
            # Bend director (Top -> Bottom)
            # 1. Build Top Composite
            tx, ty, tz = build_combined_functions(recipe["top"], unique_name_suffix=f"{current_id}_top")
            # 2. Build Bottom Composite
            bx, by, bz = build_combined_functions(recipe["bottom"], unique_name_suffix=f"{current_id}_bot")
            
            h_val = MESH_LENGTHS[2] * recipe.get("height_factor", 1.0)
            
            # Initialize BendDirector
            # Note: BendDirector expects tuples of functions for directions
            Interpolation_gen = DirectorField.InterpolationDirector(
                name=current_id, 
                height=h_val,
                Top_direction=(tx, ty, tz),
                Bottom_direction=(bx, by, bz)
            )
            
            nx, ny, nz = Interpolation_gen.as_init_funcs()
            nfield.init_from_funcs(nx, ny, nz)
            
        else:
            print(f"Unknown recipe type: {recipe['type']}, skipping.")
            continue

        nfield.normalize()
        
        # Save VTI
        nfield_vti_path = str(Path(DATABASE_DIR) / (current_id + '.vti'))
        nfield.save_to_vti(nfield_vti_path)

        # --- C. Optical Simulation ---
        mat = nm.LCMaterial(
            lc_field=nfield, ne=1.750, no=1.526, nhost=1.0003, nin=1.51, nout=1.0003
        )
        mat.add_isotropic_layer(nlayer=1.51, thickness=1000)

        wavelengths = np.linspace(0.4, 0.6, 20)
        sim = nm.LightPropagator(
            material=mat, 
            wavelengths=wavelengths, 
            max_NA_objective=0.4, 
            max_NA_condenser=0.4, 
            N_radial_wavevectors=1
        )

        output_fields = sim.propagate_fields(method=PROPAGATION_METHOD)
        viewer = nm.FieldViewer(output_fields)

        # --- D. Image Generation ---
        polariser_angles = [0, 30, 60, 90]
        analyser_angles = [0, 30, 60, 90]
        img_grid = []

        for j in analyser_angles:
            row_imgs = []
            for i in polariser_angles:
                img = utilities_functions.get_img_np(viewer, polariser_angle=i, analyser_angle=j)
                row_imgs.append(img)
            row_concat = np.concatenate(row_imgs, axis=1)
            img_grid.append(row_concat)

        final_img = np.concatenate(img_grid, axis=0)
        final_filename = current_id + ".bmp"
        utilities_functions.save_np_img(final_img, DATABASE_DIR, final_filename)

        # --- E. Update Metadata ---
        if metadata_json.exists():
            with open(metadata_json, 'r') as mf:
                try:
                    metadata = json.load(mf)
                except json.JSONDecodeError:
                    metadata = []
        else:
            metadata = []

        metadata.append({
            "ImgName": current_id,
            "MeshLengths": MESH_LENGTHS,
            "MeshDimensions": MESH_DIMENSIONS,
            "Method": PROPAGATION_METHOD,
            "RecipeType": recipe["type"],
            "RecipeDetails": recipe, # Store the full recipe used for this sample
            "PolariserAngles": polariser_angles,
            "AnalyserAngles": analyser_angles
        })

        with open(metadata_json, 'w') as mf:
            json.dump(metadata, mf, indent=4)
            
    except Exception as e:
        print(f"FAILED on sample {current_id}: {e}")
        import traceback
        traceback.print_exc()

print("Batch processing complete.")