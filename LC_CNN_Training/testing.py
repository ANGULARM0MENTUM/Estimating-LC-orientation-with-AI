# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 16:39:38 2026

@author: tomla
"""

# script.py
# Tutorial example - Double Twist Droplet with nemaktis

import nemaktis as nm
import numpy as np

# ───────────────────────────────────────────────
# 1. Define the director field (optical axes)
# ───────────────────────────────────────────────

# Create empty director field object
# mesh_lengths  → physical size in µm (Lx, Ly, Lz)
# mesh_dimensions → number of grid points (Nz, Ny, Nx)
nfield = nm.DirectorField(
    mesh_lengths=(10, 10, 10),
    mesh_dimensions=(40, 40, 40)
)

# Analytical functions for double-twist cylinder director field
q = 2 * np.pi / 20   # twist rate (rad/µm)

def nx(x, y, z):
    r = np.sqrt(x**2 + y**2)
    return -q * y * np.sinc(q * r)

def ny(x, y, z):
    r = np.sqrt(x**2 + y**2)
    return q * x * np.sinc(q * r)

def nz(x, y, z):
    r = np.sqrt(x**2 + y**2)
    return np.cos(q * r)

# Fill the field using the analytical expressions
nfield.init_from_funcs(nx, ny, nz)

# Make sure director is normalized (usually recommended)
nfield.normalize()

# Geometric transformations & mask
nfield.rotate_90deg("x")       # rotate 90° around x-axis
nfield.extend(2, 2)            # scale xy plane by factor 2 (important for mask)
nfield.set_mask(mask_type="droplet")   # spherical droplet mask

# Optional: save director field for later use / visualization in Paraview
nfield.save_to_vti("double_twist_droplet")

# You can also load it later like this:
# nfield = nm.DirectorField(vti_file="double_twist_droplet.vti")


# ───────────────────────────────────────────────
# 2. Define materials and isotropic layers
# ───────────────────────────────────────────────

mat = nm.LCMaterial(
    lc_field = nfield,
    ne     = 1.7,     # extraordinary index of LC
    no     = 1.5,     # ordinary index of LC
    nhost  = 1.55,    # refractive index of surrounding fluid
    nin    = 1.51,    # index below LC layer (for interface correction)
    nout   = 1.0      # index after objective / air
)

# Add isotropic layers above the LC droplet
mat.add_isotropic_layer(nlayer=1.55, thickness=5.0)     # 5 µm host fluid
mat.add_isotropic_layer(nlayer=1.51, thickness=1000.0)  # 1 mm glass plate


# ───────────────────────────────────────────────
# 3. Propagate optical fields through the sample
# ───────────────────────────────────────────────

wavelengths = np.linspace(0.4, 0.8, 11)   # 400–800 nm, 11 points

sim = nm.LightPropagator(
    material              = mat,
    wavelengths           = wavelengths,
    max_NA_objective      = 0.4,
    max_NA_condenser      = 0.0,          # almost closed condenser → coherent
    N_radial_wavevectors  = 1             # → only normal incidence
)

# Run the propagation (using Beam Propagation Method)
print("Propagating fields ... (may take a few minutes)")    
output_fields = sim.propagate_fields(method="dtmm")

# Optional: save optical fields (can be reloaded later)
# output_fields.save_to_vti("optical_fields")

# You can reload later with:
# output_fields = nm.OpticalFields(vti_file="optical_fields.vti")


# ───────────────────────────────────────────────
# 4. Open interactive viewer to explore micrographs
# ───────────────────────────────────────────────

print("Starting interactive viewer...")
viewer = nm.FieldViewer(output_fields)
viewer.plot()

print("You can now interactively change:")
print(" • polarizer / analyzer angle")
print(" • compensators")
print(" • focus position (z-focus)")
print(" • NA of objective & condenser")
print(" • wavelength (or use white light)")