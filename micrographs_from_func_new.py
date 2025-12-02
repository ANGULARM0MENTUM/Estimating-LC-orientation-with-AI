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
import DirectorField


mesh_lengths=(20,20,5)
mesh_dimensions=(40,40,10)

img_name = "combined1"

#nfield_vti = 'POUYA_90twist_dir.vti' #file name for nfield to save
#output_fields_vti = "OUTPUTFIELD_90twist_POUYA" #file name for output field to save

nfield_vti = str(Path(img_name) / (img_name+'_dir.vti')) #file name for nfield to save
output_fields_vti =  str(Path(img_name) / ("OUTPUTFIELD_"+img_name)) #file name for output field to save
print(nfield_vti)
print(output_fields_vti)

Path(img_name).mkdir(exist_ok=True)

# set dimensions of director field
nfield = nm.DirectorField(
    mesh_lengths=mesh_lengths, # (Lx, Ly, Lz)
    mesh_dimensions=mesh_dimensions) # (Nx, Ny, Nz) # fix dimensions 40 40 10 


DirectorFieldInitializer1 = DirectorField.RadialDirector(name = img_name,center=(0,0,0),axis_mask=(1,1,0),normalize = False)
DirectorFieldInitializer2 = DirectorField.NoiseDirector(name = img_name)
DirectorFieldInitializer3 = DirectorField.CircularDirector(name = img_name,center=(0,0,0),normalize = False)
DirectorFieldInitializer4 = DirectorField.ParabolarDirector(name = img_name,center=(0,3,0))
DirectorFieldInitializer5 = DirectorField.RandomDirector(name = img_name,n_modes=2,amplitude=0.3)
DirectorFieldInitializer6 = DirectorField.LineDirector(name = img_name,direction=(-1,-1,0))
DirectorFieldInitializer7 = DirectorField.LineDirector(name = img_name,direction=(-1,1,0))
DirectorFieldInitializer8 = DirectorField.LineDirector(name = img_name,direction=(1,1,0))

nx1, ny1, nz1 = DirectorFieldInitializer1.as_init_funcs()
nx2, ny2, nz2 = DirectorFieldInitializer2.as_init_funcs()
nx3, ny3, nz3 = DirectorFieldInitializer3.as_init_funcs()
nx4, ny4, nz4 = DirectorFieldInitializer4.as_init_funcs()
nx5, ny5, nz5 = DirectorFieldInitializer5.as_init_funcs()
nx6, ny6, nz6 = DirectorFieldInitializer6.as_init_funcs()
nx7, ny7, nz7 = DirectorFieldInitializer7.as_init_funcs()
nx8, ny8, nz8 = DirectorFieldInitializer8.as_init_funcs()

nx = lambda x,y,z: 10*nx3(x,y,z)+nx5(x,y,z)#+nx6(x-4,y,z)+nx7(x-4,y,z)+nx8(x+4,y,z)
ny = lambda x,y,z: 10*ny3(x,y,z)+ny5(x,y,z)#+ny6(x-4,y,z)+ny7(x-4,y,z)+ny8(x+4,y,z)
nz = lambda x,y,z: 10*nz3(x,y,z)+nz5(x,y,z)#+nz6(x-4,y,z)+nz7(x-4,y,z)+nz8(x+4,y,z)

# initialize the director field with the functions nx, ny, nz
nfield.init_from_funcs(nx,ny,nz)
nfield.normalize()


# save the director in vti format (optional)
#nfield.save_to_vti('C:/Users/manzoni/Desktop/NEMAKTIS/POUYA_90twist_dir.vti')
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

#print(sim.material)
#sys.exit()


# make the light propagate
#output_fields=sim.propagate_fields(method="bpm") #bpm often breaks
output_fields=sim.propagate_fields(method="dtmm(1)") 

#save the results of the simulation
output_fields.save_to_vti(output_fields_vti)

img_property_list = []
# Use Nemaktis viewer to see the output
viewer = nm.FieldViewer(output_fields)

# viewer.plot()
# sys.exit()

polariser_angles = [0,15,30,45,60,75,90]
analyser_angles = [0,15,30,45,60,75,90]

# Initialize a 2D list to hold images
img_grid = []

for j in analyser_angles:
    row_imgs = []
    for i in polariser_angles:
        img = utilities_functions.get_img_np(viewer,polariser_angle=i,analyser_angle=j)
        FileName = img_name+"_polariser="+str(viewer.polariser_angle)+"_analyser="+str(viewer.analyser_angle)+".bmp"
        
        img_property_list.append(utilities_functions.get_properties(viewer,ImgName = img_name, FieldType="None")) #if hasattr(viewer, 'NA_objective') else None,})
        
        utilities_functions.save_np_img(img,img_name,FileName)
        
        row_imgs.append(img)
    row_concat = np.concatenate(row_imgs, axis=1)  # axis=1 is horizontal concatenation
    img_grid.append(row_concat)

# Concatenate all rows along y-axis (height)
final_img = np.concatenate(img_grid, axis=0)  # axis=0 is vertical concatenation
final_filename = img_name + ".bmp"
utilities_functions.save_np_img(final_img, img_name, final_filename)

output_path = Path(img_name) / "list.txt"
with open(output_path, 'w') as f:
    json.dump(img_property_list, f, indent=4)
