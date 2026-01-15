# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 12:50:25 2025

@author: JohnC
"""

"""
director_fields.py

Common director-field classes for liquid-crystal initialization.
Optimized for 2D director structures (z-invariant) for CNN training generation.

Classes included:
- UniformDirector (2D direction)
- TwistDirector (Exception: Keeps 3D structure)
- InterpolationDirector (Exception: Keeps 3D structure)
- NoiseDirector (Exception: Keeps 3D structure)
- LineDirector (2D projection)
- CircularDirector (2D circles)
- RadialDirector (2D splay)
- ParabolarDirector (2D parabolic tangents)
- RandomDirector (Fourier series smooth random)
- DefectDirector (New: Topological defects k = +/- 0.5, 1, etc.)
- SinusoidalDirector (New: Wavy director field)
- DomainDirector (New: Checkerboard of uniform domains)
- DomainLineDirector (New: Random grid of uniform domains)
"""

import numpy as np
from typing import Callable, Tuple, Optional, Union

# Utility functions
def _ensure_array(a, template):
    """Return array with same shape as template (broadcasting scalars)."""
    return np.full_like(template, a) if np.isscalar(a) else np.asarray(a)

def _normalize_components(nx, ny, nz, eps=1e-12):
    mag = np.sqrt(nx*nx + ny*ny + nz*nz)
    mag = np.maximum(mag, eps)
    return nx/mag, ny/mag, nz/mag

class BaseDirector:
    """Base class interface for director fields."""
    def __call__(self, x, y, z):
        """Return (nx, ny, nz) arrays with same shape as x,y,z."""
        raise NotImplementedError

    def as_init_funcs(self) -> Tuple[Callable, Callable, Callable]:
        """Return three functions fn(x,y,z), gn(x,y,z), hn(x,y,z)."""
        return (lambda x,y,z: self(x,y,z)[0],
                lambda x,y,z: self(x,y,z)[1],
                lambda x,y,z: self(x,y,z)[2])

# ---------------------------------------------------------------------
# 3D / Z-Dependent Fields (Exceptions allowed by user)
# ---------------------------------------------------------------------

class TwistDirector(BaseDirector):
    """
    Twist director field (Allowed Exception: Z-dependent).
    Rotates around a given axis.
    """
    Type = "Twist"
    def __init__(self, name, q, axis='z', phi0=0.0, out_of_plane=0.0, normalize=True):
        self.name = name
        self.q = float(q)
        if axis not in ('x', 'y', 'z'):
            raise ValueError("axis must be 'x', 'y', or 'z'")
        self.axis = axis
        self.phi0 = float(phi0)
        self.out_of_plane = float(out_of_plane)
        self.normalize = bool(normalize)

    def __call__(self, x, y, z):
        x, y, z = map(np.asarray, (x, y, z))
        # phase varies along the same axis that the director rotates around
        if self.axis == 'z':
            phi = self.q * z + self.phi0
            nx, ny, nz = np.cos(phi), np.sin(phi), np.full_like(z, self.out_of_plane)
        elif self.axis == 'y':
            phi = self.q * y + self.phi0
            nx, ny, nz = np.cos(phi), np.full_like(y, self.out_of_plane), np.sin(phi)
        elif self.axis == 'x':
            phi = self.q * x + self.phi0
            nx, ny, nz = np.full_like(x, self.out_of_plane), np.cos(phi), np.sin(phi)

        if self.normalize:
            nx, ny, nz = _normalize_components(nx, ny, nz)
        return nx, ny, nz

class InterpolationDirector(BaseDirector):
    """
    interpolation director field (Allowed Exception: Z-dependent).
    Linearly interpolates between Top and Bottom alignment along Z.
    """
    Type = "interpolation"
    def __init__(self, name, height, Top_direction=(1,0,0), Bottom_direction=(0,0,1), normalize=True):
        self.name = name
        self.height = height
        self.normalize = normalize
        self.top_funcs = self._parse_direction(Top_direction)
        self.bot_funcs = self._parse_direction(Bottom_direction)

    def _parse_direction(self, direction):
        if isinstance(direction, (tuple, list)) and len(direction) == 3:
            if all(callable(d) for d in direction):
                return direction
            else:
                try:
                    d = np.array(direction, dtype=float)
                    mag = np.sqrt(np.sum(d**2))
                    if mag > 1e-12: d = d / mag
                    return (lambda x,y,z: np.full_like(x, d[0]),
                            lambda x,y,z: np.full_like(x, d[1]),
                            lambda x,y,z: np.full_like(x, d[2]))
                except (ValueError, TypeError):
                    pass
        raise ValueError("Direction must be a tuple of 3 callables or 3 numbers.")

    def __call__(self, x, y, z):
        x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
        t_func_x, t_func_y, t_func_z = self.top_funcs
        b_func_x, b_func_y, b_func_z = self.bot_funcs
        
        tx = t_func_x(x,y,z); ty = t_func_y(x,y,z); tz = t_func_z(x,y,z)
        bx = b_func_x(x,y,z); by = b_func_y(x,y,z); bz = b_func_z(x,y,z)
        
        w_top = (self.height - z - self.height/2)
        w_bot = (z + self.height/2)
        
        nx = w_top * tx + w_bot * bx
        ny = w_top * ty + w_bot * by
        nz = w_top * tz + w_bot * bz
        
        if self.normalize:
            nx, ny, nz = _normalize_components(nx, ny, nz)
        return nx, ny, nz

class NoiseDirector(BaseDirector):
    """
    Noise director field (Allowed Exception: Random 3D).
    """
    Type = "Noise"
    def __init__(self, name, seed: Optional[int]=None, mask=(1,1,0), normalize=True, distribution="normal"):
        self.name = name
        self.seed = seed
        if self.seed is None:
            self.seed = np.random.SeedSequence().generate_state(1)[0]
        self.normalize = bool(normalize)
        self.mask = mask
        self.distribution = distribution

    def __call__(self, x, y, z):
        x = np.asarray(x)
        shape = x.shape
        rng = np.random.default_rng(self.seed)
        if self.distribution == "normal":
            nx = rng.normal(size=shape)*self.mask[0]
            ny = rng.normal(size=shape)*self.mask[1]
            nz = rng.normal(size=shape)*self.mask[2]
        else:
            raise ValueError("No such distribution")
        if self.normalize:
            nx, ny, nz = _normalize_components(nx, ny, nz)
        return nx, ny, nz

# ---------------------------------------------------------------------
# Strictly 2D / Z-Invariant Fields
# ---------------------------------------------------------------------

class UniformDirector(BaseDirector):
    """
    Uniform director field. 
    Strictly 2D planar if direction is 2D, or constant 3D vector.
    
    Parameters:
    - direction: angle (float, radians) OR tuple (dx, dy) OR tuple (dx, dy, dz).
    """
    Type = "Uniform"
    def __init__(self, name, direction: Union[float, Tuple]=0.0, normalize=True):
        self.name = name
        self.normalize = bool(normalize)
        
        # Handle input types
        if np.isscalar(direction):
            # Input is an angle in radians
            theta = float(direction)
            self._dir = (np.cos(theta), np.sin(theta), 0.0)
        else:
            d = np.asarray(direction, dtype=float)
            if d.size == 2:
                # Input is 2D vector
                self._dir = (d[0], d[1], 0.0)
            elif d.size == 3:
                # Input is 3D vector
                self._dir = (d[0], d[1], d[2])
            else:
                raise ValueError("Direction must be angle, 2-tuple, or 3-tuple.")

    def __call__(self, x, y, z):
        x = np.asarray(x)
        # Broadcast constant value to shape of x/y/z
        nx = np.full_like(x, self._dir[0], dtype=float)
        ny = np.full_like(x, self._dir[1], dtype=float)
        nz = np.full_like(x, self._dir[2], dtype=float)
        
        if self.normalize:
            nx, ny, nz = _normalize_components(nx, ny, nz)
        return nx, ny, nz

class LineDirector(BaseDirector):
    """
    2D Director aligning with a line projection on XY plane.
    Z-invariant.
    """
    def __init__(self, name, direction=(1,0), normalize=True, eps=1e-12):
        self.name = name
        self.normalize = normalize
        self.eps = eps

        d = np.asarray(direction, dtype=float)
        if d.size == 3: d = d[:2] # Ignore z input if provided
        
        dx, dy = d[0], d[1]
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            dx, dy = 1.0, 0.0
            
        mag = np.sqrt(dx*dx + dy*dy)
        self.dx = dx / mag
        self.dy = dy / mag

    def __call__(self, x, y, z):
        x = np.asarray(x); y = np.asarray(y)
        # Distance to line passing through origin with direction (dx, dy)
        dist = np.abs(x*self.dy - y*self.dx)
        dist = np.maximum(dist, self.eps)
        intensity = 1.0 / dist

        nx = np.full_like(x, self.dx) * intensity
        ny = np.full_like(x, self.dy) * intensity
        nz = np.zeros_like(x)

        if self.normalize:
            nx, ny, nz = _normalize_components(nx, ny, nz)
        return nx, ny, nz

class CircularDirector(BaseDirector):
    """
    2D Circular director field around a center (xc, yc).
    Director is tangent to circles. Z-invariant.
    """
    Type = "Circular"
    def __init__(self, name, center=(0.0,0.0), normalize=True):
        self.name = name
        self.center = center[:2] # Ensure only x,y used
        self.normalize = bool(normalize)

    def __call__(self, x, y, z):
        x = np.asarray(x); y = np.asarray(y)
        xc, yc = self.center
        X = x - xc
        Y = y - yc
        
        intensity = 1/np.sqrt(X**2+Y**2)
        theta = np.arctan2(Y, X)
        # Tangent direction: (-sin, cos)
        nx = -np.sin(theta)*intensity
        ny =  np.cos(theta)*intensity
        nz =  np.zeros_like(nx)

        if self.normalize:
            nx, ny, nz = _normalize_components(nx, ny, nz)
        return nx, ny, nz

class RadialDirector(BaseDirector):
    """
    2D Radial director field around a center (xc, yc).
    Director points outwards. Z-invariant.
    """
    Type = "Radial"
    def __init__(self, name, center=(0.0,0.0), normalize=True):
        self.name = name
        self.center = center[:2]
        self.normalize = bool(normalize)

    def __call__(self, x, y, z):
        x = np.asarray(x); y = np.asarray(y)
        xc, yc = self.center
        X = x - xc
        Y = y - yc
        
        intensity = 1/np.sqrt(X**2+Y**2)
        theta = np.arctan2(Y, X)
        nx = np.cos(theta)*intensity
        ny = np.sin(theta)*intensity
        nz = np.zeros_like(nx)
        
        if self.normalize:
            nx, ny, nz = _normalize_components(nx, ny, nz)
        return nx, ny, nz

class ParabolarDirector(BaseDirector):
    """
    2D Director tangent to parabolas y = a*(x-xc)^2 + c.
    Z-invariant.
    """
    Type = "Parabolar"
    def __init__(self, name, center=(0.0, 0.0), angle=0.0, normalize=True):
        self.name = name
        self.normalize = bool(normalize)
        self.center = center[:2]
        self.angle = angle*np.pi/180

    def __call__(self, x, y, z=None):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        xc, yc = self.center

        X = x - xc
        Y = y - yc

        #rotation
        X = X*np.cos(self.angle)-Y*np.sin(self.angle)
        Y = X*np.sin(self.angle)+Y*np.cos(self.angle)
        # Solving the geometry for the parabola passing through (X,Y) with vertex at (0,0) relative
        # To handle y=0 cleanly, we use arctan logic on the slope.
        # General slope logic preserved from original, simplified for 2D
        t = (Y + np.sqrt(Y**2 + 4*X**2 + 1e-12)) / 2
        a_map = -1 / (t + 1e-12)
        slope = 2.0 * a_map * X
        theta = np.arctan(slope)

        nx = np.cos(theta)
        ny = np.sin(theta)
        nz = np.zeros_like(nx)

        if self.normalize:
            nx, ny, nz = _normalize_components(nx, ny, nz)
        return nx, ny, nz

class RandomDirector(BaseDirector):
    """
    Smooth pseudo-random 2D director field using truncated Fourier series.
    Z-invariant.
    """
    def __init__(self, name, seed=None, normalize=True,
                 n_modes=8, k_min=0.1, k_max=5.0, amplitude=1.0):
        self.name = name
        self.normalize = normalize
        self.amplitude = amplitude
        self.n_modes = n_modes

        rng = np.random.default_rng(seed)
        self.kx = rng.uniform(k_min, k_max, size=n_modes)
        self.ky = rng.uniform(k_min, k_max, size=n_modes)
        self.Ax = rng.normal(size=n_modes)
        self.Ay = rng.normal(size=n_modes)
        self.phix = rng.uniform(0, 2*np.pi, size=n_modes)
        self.phiy = rng.uniform(0, 2*np.pi, size=n_modes)

    def __call__(self, x, y, z):
        x = np.asarray(x); y = np.asarray(y)
        nx = np.zeros_like(x, dtype=float)
        ny = np.zeros_like(x, dtype=float)

        for i in range(self.n_modes):
            phase = 2 * np.pi * (self.kx[i] * x + self.ky[i] * y)
            nx += self.Ax[i] * np.sin(phase + self.phix[i])
            ny += self.Ay[i] * np.sin(phase + self.phiy[i])

        nz = np.zeros_like(nx)
        if self.normalize:
            nx, ny, nz = _normalize_components(nx, ny, nz)
        return nx, ny, nz

# ---------------------------------------------------------------------
# NEW CLASSES: Suitable for CNN Training
# ---------------------------------------------------------------------

class DefectDirector(BaseDirector):
    """
    [NEW] Generates a 2D topological defect.
    Excellent for training CNNs to recognize singularities (+1/2, -1/2, +1).
    
    Formula: alpha = k * atan2(y-yc, x-xc) + phi0
    
    Parameters:
    - center: (xc, yc)
    - k: charge strength (e.g., 0.5, -0.5, 1.0).
         k=1.0, phi0=0 -> Radial
         k=1.0, phi0=pi/2 -> Circular
    - phi0: constant phase offset
    """
    Type = "Defect"
    def __init__(self, name, center=(0.0, 0.0), k=1.0, phi0=0.0, normalize=True):
        self.name = name
        self.center = center[:2]
        self.k = float(k)
        self.phi0 = float(phi0)
        self.normalize = normalize

    def __call__(self, x, y, z):
        x = np.asarray(x); y = np.asarray(y)
        xc, yc = self.center
        
        # Calculate angle of spatial coordinate
        theta_space = np.arctan2(y - yc, x - xc)
        
        # Calculate director angle
        alpha = self.k * theta_space + self.phi0
        
        nx = np.cos(alpha)
        ny = np.sin(alpha)
        nz = np.zeros_like(nx)
        
        if self.normalize:
            nx, ny, nz = _normalize_components(nx, ny, nz)
        return nx, ny, nz

class SinusoidalDirector(BaseDirector):
    """
    [NEW] Director angle oscillates spatially: n = (cos(alpha), sin(alpha), 0)
    where alpha = alpha0 + A * sin(qx * x + qy * y).
    
    Good for training gradient detection.
    
    Parameters:
    - q: (qx, qy) wavevectors controlling frequency
    - amplitude: A (magnitude of oscillation in radians)
    - alpha0: base angle
    """
    Type = "Sinusoidal"
    def __init__(self, name, q=(1.0, 0.0), amplitude=1.0, alpha0=0.0, normalize=True):
        self.name = name
        self.qx = q[0]
        self.qy = q[1]
        self.amplitude = amplitude
        self.alpha0 = alpha0
        self.normalize = normalize

    def __call__(self, x, y, z):
        x = np.asarray(x); y = np.asarray(y)
        
        phase = self.qx * x + self.qy * y
        alpha = self.alpha0 + self.amplitude * np.sin(phase)
        
        nx = np.cos(alpha)
        ny = np.sin(alpha)
        nz = np.zeros_like(nx)
        
        if self.normalize:
            nx, ny, nz = _normalize_components(nx, ny, nz)
        return nx, ny, nz

class DomainDirector(BaseDirector):
    """
    [NEW] Divides the XY plane into a checkerboard-like grid.
    Each cell has a random uniform orientation.
    Good for training edge detection and segmentation.
    
    Parameters:
    - cell_size: tuple (size_x, size_y)
    - seed: random seed for reproducibility
    """
    Type = "Domain"
    def __init__(self, name, cell_size=(10.0, 10.0), seed=None, normalize=True):
        self.name = name
        self.sx = cell_size[0]
        self.sy = cell_size[1]
        self.seed = seed
        self.normalize = normalize
        # Note: We rely on hashing coordinates on the fly to support arbitrary grid sizes 
        # without pre-calculating a fixed array.

    def __call__(self, x, y, z):
        x = np.asarray(x); y = np.asarray(y)
        
        # Determine grid indices
        ix = np.floor(x / self.sx).astype(int)
        iy = np.floor(y / self.sy).astype(int)
        
        # Generate reproducible random angles based on indices
        # We use a simple pseudo-hash
        rng = np.random.default_rng(self.seed)
        
        # We need a way to map (ix, iy) to a random angle efficiently for arrays
        # A simple hashing trick for visualization purposes:
        # (Using sine to scramble integers into pseudo-random numbers)
        np.random.seed(self.seed)
        # 1. Create unique ID per cell
        cell_id = ix * 31337 + iy * 50021 
        # 2. Scramble to 0..2pi
        angles = (np.sin(cell_id) * 43758.5453) % (2*np.pi)
        
        nx = np.cos(angles)
        ny = np.sin(angles)
        nz = np.zeros_like(nx)
        
        if self.normalize:
            nx, ny, nz = _normalize_components(nx, ny, nz)
        return nx, ny, nz
    
class DomainLineDirector(BaseDirector):
    """
    [NEW] Divides the XY plane using random lines. 
    Each line splits the space into two half-planes with different orientations.
    The final director is the vector sum of these contributions.
    
    Parameters:
    - height: The physical size of the mesh (used to place random lines within view).
    - number_of_lines: How many splitting lines to generate.
    - seed: Random seed.
    """
    Type = "DomainLine"

    def __init__(self, name, height=20.0, number_of_lines=3, seed=None, normalize=True):
        self.name = name
        self.seed = seed
        self.number_of_lines = int(number_of_lines)
        self.normalize = normalize
        
        rng = np.random.default_rng(seed)
        
        # 1. Define the splitting lines using Normal Vectors to avoid infinite slope issues
        # Line orientation angle (0 to 180 degrees)
        line_angles = rng.uniform(0, np.pi, size=self.number_of_lines)
        
        # Calculate normal vector (nx, ny) perpendicular to the line
        # If line is at angle theta, normal is at theta + 90 deg
        self.norms_x = -np.sin(line_angles)
        self.norms_y = np.cos(line_angles)
        
        # 2. Define a point each line passes through
        self.px = rng.uniform(-height/2, height/2, size=self.number_of_lines)
        self.py = rng.uniform(-height/2, height/2, size=self.number_of_lines)

        # 3. Define the director orientation angles for the "Left" and "Right" sides of each line
        # (0 to 360 degrees in radians)
        self.angle_side_A = rng.uniform(0, 2*np.pi, size=self.number_of_lines)
        self.angle_side_B = rng.uniform(0, 2*np.pi, size=self.number_of_lines)

    def __call__(self, x, y, z):
        x = np.asarray(x)
        y = np.asarray(y)
        
        # Initialize accumulator fields
        total_nx = np.zeros_like(x, dtype=float)
        total_ny = np.zeros_like(y, dtype=float)
        
        for i in range(self.number_of_lines):
            # Calculate signed distance from the line: d = (P - P0) . Normal
            # d > 0 is one side, d < 0 is the other
            val = (x - self.px[i]) * self.norms_x[i] + (y - self.py[i]) * self.norms_y[i]
            
            # Vectorized selection:
            # If val >= 0 use angle A, else use angle B
            chosen_angle = np.where(val >= 0, self.angle_side_A[i], self.angle_side_B[i])
            
            # Accumulate vector components
            total_nx += np.cos(chosen_angle)
            total_ny += np.sin(chosen_angle)
        
        nz = np.zeros_like(total_nx)
        
        if self.normalize:
            total_nx, total_ny, nz = _normalize_components(total_nx, total_ny, nz)
            
        return total_nx, total_ny, nz