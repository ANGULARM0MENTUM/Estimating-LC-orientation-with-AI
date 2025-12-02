"""
director_fields.py

Common director-field classes for liquid-crystal initialization.

Each director field is implemented as an independent class. Instances are callable:
    nx, ny, nz = my_field(x, y, z)

or you can get functions suitable for nfield.init_from_funcs by:
    nx_func, ny_func, nz_func = my_field.as_init_funcs()
    nfield.init_from_funcs(nx_func, ny_func, nz_func)
    
Example:
    DirectorFieldInitializer = DirectorField.RadialDirector(center=(10,10,0))  

    nx, ny, nz = DirectorFieldInitializer.as_init_funcs()

    nfield.init_from_funcs(nx,ny,nz)

Classes included:
- UniformDirector
- TwistDirector (planar twist along x or any axis via 'axis' parameter)
- CircularDirector (circular/bent director around an axis)
- RadialDirector (radial from a center)
- NoiseDirector (random noise)

All classes include a 'normalize' boolean (default True) to ensure unit length.
"""

import numpy as np
from typing import Callable, Tuple, Optional
#import utilities_functions

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

class UniformDirector(BaseDirector):
    """Uniform director field pointing along provided vector.

    Parameters:
    - direction: tuple or list or array (dx,dy,dz) or angle for in-plane if use_angle=True
    - normalize: True to return unit vectors
    - use_angle: if True, 'direction' is interpreted as angle (radians) in xy-plane
    """
    Type = "Uniform"
    def __init__(self, name, direction=(1.0,0.0,0.0), normalize=True, use_angle=False):
        self.name = name
        self.normalize = bool(normalize)
        if use_angle:
            theta = float(direction)
            self._dir = (np.cos(theta), np.sin(theta), 0.0)
        else:
            d = np.asarray(direction, dtype=float)
            if d.size == 2:
                d = np.array([d[0], d[1], 0.0])
            self._dir = tuple(d.tolist())

    def __call__(self, x, y, z):
        x = np.asarray(x)
        nx = np.full_like(x, self._dir[0], dtype=float)
        ny = np.full_like(x, self._dir[1], dtype=float)
        nz = np.full_like(x, self._dir[2], dtype=float)
        if self.normalize:
            nx, ny, nz = _normalize_components(nx, ny, nz)
        return nx, ny, nz

class TwistDirector(BaseDirector):
    """
    Twist director field that rotates around a given axis (default 'z').

    Example: a 90° twist around z-axis along z (planar twist cell)
      n = (cos(q*z + phi0), sin(q*z + phi0), 0)

    Parameters
    ----------
    q : float
        Wavevector (2*pi / pitch)
    axis : {'x', 'y', 'z'}, default 'z'
        Rotation axis (around which director rotates)
    phi0 : float, default 0.0
        Initial phase offset (radians)
    length : float, optional
        Physical thickness (used to normalize q if desired)
    out_of_plane : float, default 0.0
        Constant component along rotation axis
    normalize : bool, default True
        Normalize result to unit length
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

class BendDirector(BaseDirector):
    """Bend director field
        given a direction on top and bottom, generate a bend struture where the director of
        liquid crystal slows truns from the top direction to the bottom direction.
        
        Parameters:
        - height: height of the sample, twice of the z component of mesh length
        - Top_direction: tuple of size 3 e.g. (1,0,0) 
        - Bottom_direction: tuple of size 3 e.g. (1,0,0)
    """
    Type = "Bend"
    def __init__(self, name, height, Top_direction=(1,0,0), Bottom_direction=(0,0,1), normalize=True):
        self.name = name
        self.height = height
        self.normalize = normalize
        self.Top_direction = _normalize_components(Top_direction[0],Top_direction[1],Top_direction[2])
        self.Bottom_direction = _normalize_components(Bottom_direction[0],Bottom_direction[1],Bottom_direction[2])
        
    def __call__(self, x, y, z):
        x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
        
        nx = (self.height-z-self.height/2)*self.Top_direction[0]+(z+self.height/2)*self.Bottom_direction[0]
        ny = (self.height-z-self.height/2)*self.Top_direction[1]+(z+self.height/2)*self.Bottom_direction[1]
        nz = (self.height-z-self.height/2)*self.Top_direction[2]+(z+self.height/2)*self.Bottom_direction[2]
        
        if self.normalize:
            nx, ny, nz = _normalize_components(nx, ny, nz)
        return nx, ny, nz
    
class LineDirector(BaseDirector):
    """
    Director field that aligns with a given line passing through (0,0,0).
    The director points in the projection of the line direction onto the xy-plane,
    and its magnitude scales as 1 / (distance_to_line_in_xy).

    The result is strictly 2D: nz = 0.

    Parameters
    ----------
    name : str
    direction : tuple(float, float, float)
        A vector (dx, dy, dz) defining the direction of the line through origin.
        Only its projection onto xy-plane is used.
    normalize : bool
        Normalize output to unit length.
    eps : float
        Small cutoff to avoid infinity at distance = 0.
    """
    def __init__(self, name, direction=(1,0,0), normalize=False, eps=1e-12):
        self.name = name
        self.normalize = normalize
        self.eps = eps

        # Convert to numpy and store XY projection
        d = np.asarray(direction, dtype=float)
        dx, dy = d[0], d[1]

        # If the direction is vertical (dx=dy=0), default to x-direction
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            dx, dy = 1.0, 0.0

        # Normalize projection
        mag = np.sqrt(dx*dx + dy*dy)
        dx /= mag; dy /= mag

        self.dx = dx
        self.dy = dy

    def __call__(self, x, y, z):
        x = np.asarray(x)
        y = np.asarray(y)

        # Vector from line to point is perpendicular distance in 2D:
        # distance = |(xx,yy) x (dx,dy)| = |xx*dy - yy*dx|
        dist = np.abs(x*self.dy - y*self.dx)
        dist = np.maximum(dist, self.eps)

        # Intensity falls off as 1/dist
        intensity = 1.0 / dist

        # Director aligns with projected line direction
        nx = self.dx * intensity
        ny = self.dy * intensity
        nz = np.zeros_like(nx)

        if self.normalize:
            nx, ny, nz = _normalize_components(nx, ny, nz)

        return nx, ny, nz
class CircularDirector(BaseDirector):
    """Circular (bend) director around a chosen axis.

    For axis='z', director is tangent to circles in xy plane:
      n = (-sin(theta), cos(theta), 0) where theta = atan2(y - yc, x - xc)

    Parameters:
    - center: (xc, yc, zc) center of circles (only xc,yc used if axis in xy plane)
    - axis: 'x'|'y'|'z' axis around which the bend occurs (default 'z')
    - handedness: +1 or -1 to reverse direction
    - tilt: optional tilt angle (radians) giving some out-of-plane component
    - normalize: normalize result
    """
    Type = "Circular"
    def __init__(self, name, center=(0.0,0.0,0.0), axis='z', tilt=0.0, normalize=False):
        self.name = name
        self.center = tuple(center)
        if axis not in ('x','y','z'):
            raise ValueError("axis must be 'x','y' or 'z'")
        self.axis = axis
        self.tilt = float(tilt)
        self.normalize = bool(normalize)

    def __call__(self, x, y, z):
        x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
        xc, yc, zc = self.center
        if self.axis == 'z':
            X = x - xc; Y = y - yc
            theta = np.arctan2(Y, X)
            intensity = 1/np.sqrt(X**2+Y**2)
            nx = -np.sin(theta)*intensity
            ny =   np.cos(theta)*intensity
            nz = np.full_like(theta, 0.0)*intensity
        elif self.axis == 'y':
            X = x - xc; Z = z - zc
            theta = np.arctan2(Z, X)
            intensity = 1/np.sqrt(X**2+Z**2)
            nx = -np.sin(theta)*intensity
            ny = np.full_like(theta, 0.0)*intensity
            nz =   np.cos(theta)*intensity
        else:  # axis == 'x'
            Y = y - yc; Z = z - zc
            theta = np.arctan2(Z, Y)
            intensity = 1/np.sqrt(Y**2+Z**2)
            nx = np.full_like(theta, 0.0)*intensity
            ny = -np.sin(theta)*intensity
            nz =   np.cos(theta)*intensity

        # apply tilt by mixing in a constant out-of-plane component
        if self.tilt != 0.0:
            # tilt is angle between tangent and plane: small tilt adds constant component along axis
            if self.axis == 'z':
                nz = np.tan(self.tilt) * np.ones_like(nz)
            elif self.axis == 'y':
                ny = np.tan(self.tilt) * np.ones_like(ny)
            else:
                nx = np.tan(self.tilt) * np.ones_like(nx)

        if self.normalize:
            nx, ny, nz = _normalize_components(nx, ny, nz)
        return nx, ny, nz

class RadialDirector(BaseDirector):
    """Radial (splay) director field pointing away from (or towards) a center.

    Parameters:
    - center: (xc,yc,zc)
    - sign: +1 outward, -1 inward
    - normalize: normalize result
    """
    Type = "Radial"
    def __init__(self, name, center=(0.0,0.0,0.0), normalize=False, axis_mask=(1,1,0)):
        self.name = name
        self.center = tuple(center)
        self.normalize = bool(normalize)
        self.axis_mask = axis_mask

    def __call__(self, x, y, z):
        x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
        xc, yc, zc = self.center
        X = (x - xc);Y = (y - yc);Z = (z - zc)
        intensity = 1/(self.axis_mask[0]*X**2+self.axis_mask[1]*Y**2+self.axis_mask[2]*Z**2)
        nx = X*self.axis_mask[0]*intensity
        ny = Y*self.axis_mask[1]*intensity
        nz = Z*self.axis_mask[2]*intensity
        if self.normalize:
            nx, ny, nz = _normalize_components(nx, ny, nz)
        return nx, ny, nz
    
class ParabolarDirector(BaseDirector):
    """
    Director field whose in-plane directors follow tangents of parabolas y = a(x,y) * x^2 + c(x,y).

    Parameters
    - name: string name
    - a: scalar or callable a(x, y) giving parabola coefficient. Can also be an array broadcastable to x,y.
         If a is None, defaults to a(x,y) = 1/(1 + x) (a simple decreasing function along +x).
    - c: scalar or callable c(x, y) giving parabola vertical offset. Can be scalar/array/callable.
         If c is None, defaults to 0.
    - normalize: whether to normalize the director vectors (default True)
    - use_local_x: if True, interpret 'x' in tangent slope formula as local coordinate relative to a center;
                 if False (default), use the absolute x coordinate.
    - center: (x0,y0) used when use_local_x=True (defaults to (0,0))
    """
    Type = "Parabolar"

    def __init__(self,name: str,center=(0.0, 0.0, 0.0),normalize=True):
        self.name = name
        # store a and c: can be scalar/array/callable
        self.normalize = bool(normalize)
        self.center = tuple(center)



    def __call__(self, x, y, z=None):
        """
        Return nx, ny, nz arrays with same shape as x,y.
        Directors tangent to parabola y = a(x,y) * (x_local)^2 + c(x,y).
        The slope dy/dx = 2 * a * x_local, so theta = atan(2 * a * x_local).
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        xc = self.center[0]
        yc = self.center[1]

        X = x-xc
        Y = y-yc
        t = (Y+np.sqrt(Y**2+4*X**2))/2
        a_map = -1/t

        # compute slope and angle: dy/dx = 2 * a * x_local
        slope = 2.0 * a_map * X
        theta = np.arctan(slope)

        # director components (planar)
        nx = np.cos(theta)
        ny = np.sin(theta)
        nz = np.zeros_like(nx)

        if self.normalize:
            nx, ny, nz = _normalize_components(nx, ny, nz)

        return nx, ny, nz

class RandomDirector(BaseDirector):
    """
    Smooth pseudo-random director field using a truncated 2D Fourier series.
    No z-dependence. Produces visually complex but smooth patterns.

    Parameters
    ----------
    name : str
    seed : int
    normalize : bool
    n_modes : int
        Number of Fourier components. (3–12 recommended)
    k_min, k_max : float
        Frequency range (controls roughness)
    amplitude : float
        Overall magnitude scale
    """
    def __init__(self, name, seed=None, normalize=False,
                 n_modes=8, k_min=1.0, k_max=5.0, amplitude=1.0):
        self.name = name
        self.normalize = normalize
        self.amplitude = amplitude
        self.n_modes = n_modes

        rng = np.random.default_rng(seed)

        # Random wavevectors (kx, ky), amplitudes, and phases
        self.kx = rng.uniform(k_min, k_max, size=n_modes)
        self.ky = rng.uniform(k_min, k_max, size=n_modes)
        self.Ax = rng.normal(size=n_modes)
        self.Ay = rng.normal(size=n_modes)
        self.phix = rng.uniform(0, 2*np.pi, size=n_modes)
        self.phiy = rng.uniform(0, 2*np.pi, size=n_modes)

    def __call__(self, x, y, z):
        x = np.asarray(x)
        y = np.asarray(y)

        # Accumulate Fourier modes
        nx = np.zeros_like(x, dtype=float)
        ny = np.zeros_like(x, dtype=float)

        for i in range(self.n_modes):
            phase = 2 * np.pi * (self.kx[i] * x + self.ky[i] * y)
            nx += self.Ax[i] * np.sin(phase + self.phix[i])
            ny += self.Ay[i] * np.sin(phase + self.phiy[i])

        nx *= self.amplitude
        ny *= self.amplitude
        nz = np.zeros_like(nx)

        if self.normalize:
            nx, ny, nz = _normalize_components(nx, ny, nz)

        return nx, ny, nz

class NoiseDirector(BaseDirector):
    """Random Noise director field. 

    Parameters:
    - seed: RNG seed for reproducibility
    - normalize: normalize
    - distribution: distribution of noise, e.g. normal,uniform ; currently only normal is implemented
    Note: For smooth random fields, generate in Fourier space or create noise on coarse grid and interpolate.
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
        x = np.asarray(x); shape = x.shape
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

# Example usage in docstring form:
if __name__ == "__main__":

    # Example: planar twist with q = 2*pi/20 (same as your snippet)
    q = 2*np.pi/20
    arr1 = np.zeros((1,1))
    DirectorFieldInitializer = NoiseDirector("random1")
    nx, ny, nz = DirectorFieldInitializer.as_init_funcs()
    print(DirectorFieldInitializer.seed)
    print(arr1)
    print(nx(0,0,0),nx(0,0,1))
    #utilities_functions.save_np_img(nx(arr1,arr1,arr1)[:,:,np.newaxis],"Test","random.bmp")
