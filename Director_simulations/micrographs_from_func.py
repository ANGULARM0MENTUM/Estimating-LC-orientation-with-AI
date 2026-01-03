# -*- coding: utf-8 -*-
"""
Created on Fri May  2 17:03:58 2025

@author: manzoni
"""

import sys
import numpy as np 
import nemaktis as nm 
from pathlib import Path
import json
import utilities_functions # self made script containing some useful functions
import DirectorField_NEW as DirectorField


mesh_lengths=(20,20,5)
mesh_dimensions=(40,40,10)
propagation_method="dtmm(1)"

database = "Samples_test"
json_path = Path(database) / "info.json"
metadata_json = Path(database) / "Samples_metadata.json"
img_name = "0"

Path(database).mkdir(exist_ok=True)

if json_path.exists():
    # Open in read mode first to get the data
    with open(json_path, 'r') as file:
        try:
            database_data = json.load(file)
        except json.JSONDecodeError:
            # Handle case where file exists but is empty/corrupt
            database_data = {"number_of_samples": 0}
else:
    # Initialize data if file doesn't exist
    database_data = {"number_of_samples": 0}

# 4. Update the data
img_name = str(database_data["number_of_samples"])
database_data["number_of_samples"] += 1

# 5. Save the updated data back to the file (using 'w' here is correct)
with open(json_path, 'w') as file:
    json.dump(database_data, file, indent=4)

nfield_vti = str(Path(database)/(img_name+'.vti')) #file name for nfield to save

# set dimensions of director field
nfield = nm.DirectorField(
    mesh_lengths=mesh_lengths, # (Lx, Ly, Lz)
    mesh_dimensions=mesh_dimensions) # (Nx, Ny, Nz) # fix dimensions 40 40 10 


# --- Initialize instances of available Director Classes ---
DirectorFieldInitializer1 = DirectorField.UniformDirector(name=img_name, direction=(0, 0, 1), normalize=True)  # 1. Uniform: Constant direction
DirectorFieldInitializer2 = DirectorField.TwistDirector(name=img_name, q=2*np.pi/5, axis='z')                 # 2. Twist: Rotates along an axis
DirectorFieldInitializer3 = DirectorField.InterpolationDirector(name=img_name, height=10.0)                             # 3. Bend: Interpolates between Top and Bottom alignment
DirectorFieldInitializer4 = DirectorField.LineDirector(name=img_name, direction=(1,0))                       # 4. Line: Aligns with a specific line projection
DirectorFieldInitializer5 = DirectorField.CircularDirector(name=img_name, center=(0,0),normalize=True)            # 5. Circular: Tangent to circles around center
DirectorFieldInitializer6 = DirectorField.RadialDirector(name=img_name, center=(0,0))     # 6. Radial: Points outward from center
DirectorFieldInitializer7 = DirectorField.ParabolarDirector(name=img_name, center=(0,0), angle=180)                     # 7. Parabolar: Follows parabolic tangents
DirectorFieldInitializer8 = DirectorField.RandomDirector(name=img_name, n_modes=6, amplitude=1.0, k_max=5)              # 8. Random: Smooth random field (Fourier series)
DirectorFieldInitializer9 = DirectorField.NoiseDirector(name=img_name)                                # 9. Noise: White noise (grainy)
DirectorFieldInitializer10 = DirectorField.LineDirector(name=img_name, direction=(-1,1))  

DirectorFieldInitializer11 = DirectorField.SinusoidalDirector(name=img_name)

DirectorFieldInitializer12 = DirectorField.DomainLineDirector(name=img_name,height=20.0,number_of_lines=16)

nx1, ny1, nz1 = DirectorFieldInitializer1.as_init_funcs()
nx2, ny2, nz2 = DirectorFieldInitializer2.as_init_funcs()
nx3, ny3, nz3 = DirectorFieldInitializer3.as_init_funcs()
nx4, ny4, nz4 = DirectorFieldInitializer4.as_init_funcs()
nx5, ny5, nz5 = DirectorFieldInitializer5.as_init_funcs()
nx6, ny6, nz6 = DirectorFieldInitializer6.as_init_funcs()
nx7, ny7, nz7 = DirectorFieldInitializer7.as_init_funcs()
nx8, ny8, nz8 = DirectorFieldInitializer8.as_init_funcs()
nx9, ny9, nz9 = DirectorFieldInitializer9.as_init_funcs()
nx10, ny10, nz10 = DirectorFieldInitializer10.as_init_funcs()
nx11, ny11, nz11 = DirectorFieldInitializer11.as_init_funcs()
nx12, ny12, nz12 = DirectorFieldInitializer12.as_init_funcs()

nx = lambda x,y,z: nx8(x+4,y,z)+nx4(x,y,z)
ny = lambda x,y,z: ny8(x+4,y,z)+ny4(x,y,z)
nz = lambda x,y,z: nz8(x+4,y,z)+nz4(x,y,z)

BendDirectorField = DirectorField.InterpolationDirector( name = img_name, height = 2*mesh_lengths[2],Top_direction=(nx1, ny1, nz1),Bottom_direction=(nx5, ny5, nz5))
nxB ,nyB, nzB = BendDirectorField.as_init_funcs()

# initialize the director field with the functions nx, ny, nz
nfield.init_from_funcs(nx7,ny7,nz7)
nfield.normalize()

nfield.save_to_vti(nfield_vti)

#create the set up for the liquid cristal 
mat = nm.LCMaterial(
    lc_field=nfield,ne=1.750,no=1.526,nhost=1.0003,nin=1.51,nout=1.0003)
# add 1 mm-thick glass plate
mat.add_isotropic_layer(nlayer=1.51, thickness=1000)


# create the array of wavelength of the light 
wavelengths = np.linspace(0.4,0.6,20)

# create a light propagator object
sim = nm.LightPropagator(material=mat, 
                         wavelengths=wavelengths, 
                         max_NA_objective=0.4, 
                         max_NA_condenser=0.4, 
                         N_radial_wavevectors=1)


# make the light propagate
#output_fields=sim.propagate_fields(method="bpm") #bpm often breaks
output_fields=sim.propagate_fields(method=propagation_method) 


img_property_list = []
# Use Nemaktis viewer to see the output
viewer = nm.FieldViewer(output_fields)

viewer.plot()
#sys.exit()

polariser_angles = [0,30,60,90]
analyser_angles = [0,30,60,90]

# Initialize a 2D list to hold images
img_grid = []

for j in analyser_angles:
    row_imgs = []
    for i in polariser_angles:
        img = utilities_functions.get_img_np(viewer,polariser_angle=i,analyser_angle=j)
        row_imgs.append(img)
    row_concat = np.concatenate(row_imgs, axis=1)  # axis=1 is horizontal concatenation
    img_grid.append(row_concat)

# Concatenate all rows along y-axis (height)
final_img = np.concatenate(img_grid, axis=0)  # axis=0 is vertical concatenation
final_filename = img_name + ".bmp"
utilities_functions.save_np_img(final_img, database, final_filename)

#output_path = Path(database) / (img_name+".json")
if metadata_json.exists():
    with open(metadata_json, 'r') as metadata_file:
        try:
            metadata = json.load(metadata_file)
        except json.JSONDecodeError:
            metadata = []
else:
    metadata = []

metadata.append({"ImgName":img_name,
                 "MeshLengths":mesh_lengths,
                 "MeshDimensions":mesh_dimensions,
                 "Method":propagation_method,
                 "FieldType":"None",
                 "PolariserAngles":polariser_angles,
                 "AnalyserAngles":analyser_angles})

with open(metadata_json, 'w') as metadata_file:
    json.dump(metadata, metadata_file, indent=4)
