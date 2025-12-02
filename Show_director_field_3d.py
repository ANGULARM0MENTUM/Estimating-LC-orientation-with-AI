import pyvista as pv

def visualize_3d_director_field(vti_filename):
    """
    Visualizes the 3D director field from a VTI file using PyVista.
    Uses glyphs (arrows) to represent the vector field.
    """
    # Load the VTI file
    data = pv.read(vti_filename)
    
    # Print available point data for verification (optional)
    print("Available point data arrays:", data.point_data.keys())
    
    # The vector field name in Nemaktis is 'n'
    vector_field_name = 'n'
    if vector_field_name not in data.point_data:
        raise ValueError(f"Vector field '{vector_field_name}' not found in the VTI file.")
    
    # Create a plotter
    plotter = pv.Plotter()
    
    # Add the grid outline for context
    plotter.add_mesh(data.outline(), color='black', opacity=0.5)
    
    # Create glyphs: arrows representing the director vectors
    # Subsample for clarity if the grid is dense (stride=2 means every other point)
    subsampled = data #.threshold(0.1, scalars=vector_field_name)  # Optional: threshold if needed
    glyphs = subsampled.glyph(
        orient=vector_field_name,      # Vector field to orient glyphs
        scale=False,                   # Do not scale by magnitude (directors are unit vectors)
        factor=0.5,                    # Size factor for glyphs
        geom=pv.Arrow()                # Use arrows; could use pv.Line() for lines
    )
    
    # Add glyphs to the plotter
    plotter.add_mesh(glyphs, color='blue', show_scalar_bar=False)
    
    # Set camera/view options
    plotter.view_isometric()
    plotter.add_axes()  # Add XYZ axes for orientation
    
    # Show the interactive plot
    plotter.show()

# Example usage
visualize_3d_director_field('combined1/OUTPUTFIELD_combined1.vti')