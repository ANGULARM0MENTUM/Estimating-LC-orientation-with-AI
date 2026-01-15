import json
import random
import numpy as np
import argparse

# --- 1. Configuration of Available Classes ---
available_2d_classes = [
    "UniformDirector", "LineDirector", "CircularDirector", 
    "RadialDirector", "ParabolarDirector", "RandomDirector", 
    "SinusoidalDirector", "DefectDirector", "DomainDirector",
    "DomainLineDirector"
]

def get_random_params(class_name, mesh_scale=20.0):
    """Returns reasonable random parameters for specific classes."""
    params = {"normalize": True}
    
    if class_name == "UniformDirector":
        angle = random.uniform(0, 2*np.pi)
        params["direction"] = angle
        
    elif class_name == "LineDirector":
        params["direction"] = (random.uniform(-1,1), random.uniform(-1,1))
        
    elif class_name in ["CircularDirector", "RadialDirector", "ParabolarDirector", "DefectDirector"]:
        # Random center relative to mesh scale
        #half_scale = mesh_scale / 2.0
        params["center"] = (0,0)#(random.uniform(-half_scale, half_scale), random.uniform(-half_scale, half_scale))
        if class_name == "DefectDirector":
            params["k"] = random.choice([-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0])
            params["phi0"] = random.uniform(0, np.pi)
        elif class_name =="ParabolarDirector":
            params["angle"] = random.uniform(-180, 180)
            
    elif class_name == "RandomDirector":
        params["n_modes"] = random.randint(1, 10)
        params["amplitude"] = random.uniform(0, 2.0)
        
    elif class_name == "SinusoidalDirector":
        params["q"] = (random.uniform(0.1, 2.0), random.uniform(0.1, 2.0))
        params["amplitude"] = random.uniform(0, 2.0)
        
    elif class_name == "DomainDirector":
        params["cell_size"] = (random.uniform(5, 15), random.uniform(5, 15))
        params["seed"] = random.randint(0, 10000)

    elif class_name == "DomainLineDirector":
        params["height"] = mesh_scale
        params["number_of_lines"] = random.randint(1, 7)
        params["seed"] = random.randint(0, 10000)

    return params

def generate_component_list(max_complexity, mesh_scale=20.0):
    """Generates a list of components for linear combination."""
    components = []
    # Choose a random number of components between 1 and max_complexity
    num_comps = random.randint(1, max_complexity)
    
    for _ in range(num_comps):
        cls = random.choice(available_2d_classes)
        
        comp = {
            "class": cls,
            "params": get_random_params(cls,mesh_scale),
            "weight": round(random.uniform(0.2, 1.5), 2),
            # Shift in XY plane (approx +/- 5 units)
            "shift": [round(random.uniform(-mesh_scale/2, mesh_scale/2), 2), round(random.uniform(-mesh_scale/2, mesh_scale/2), 2)]
        }
        components.append(comp)
    return components

if __name__ == "__main__":
    # --- 2. Argument Parsing Logic ---
    parser = argparse.ArgumentParser(description="Generate random recipes for Director Field simulation.")
    
    # Argument: Number of samples
    parser.add_argument("--samples", type=int, default=50, 
                        help="Number of samples to generate (default: 50)")
    
    # Argument: Output filename
    parser.add_argument("--output", type=str, default="recipes.json", 
                        help="Output JSON filename (default: generation_recipes.json)")
    
    # Argument: Probability of 3D (Interpolation/Bend)
    parser.add_argument("--ratio", type=float, default=0.3, 
                        help="Probability (0.0 to 1.0) of generating a 3D Interpolation/Bend field. (default: 0.3)")
    
    # Argument: Complexity (Max number of layers mixed together)
    parser.add_argument("--complexity", type=int, default=3, 
                        help="Maximum number of linear components to mix per field. (default: 3)")

    args = parser.parse_args()

    # --- 3. Generation Loop ---
    recipes = []

    print(f"Generating {args.samples} samples...")
    print(f" - 3D/Bend Ratio: {args.ratio}")
    print(f" - Max Complexity: {args.complexity}")

    for i in range(args.samples):
        # Use args.ratio to determine if it is 3D
        is_interpolation = random.random() < args.ratio
        
        recipe = {}
        
        if is_interpolation:
            recipe["type"] = "Interpolation" 
            recipe["height_factor"] = 1.0
            # Generate top and bottom with specified complexity
            recipe["top"] = generate_component_list(args.complexity)
            recipe["bottom"] = generate_component_list(args.complexity)
        else:
            recipe["type"] = "2D"
            # Generate 2D mix with specified complexity
            recipe["components"] = generate_component_list(args.complexity)
            
        recipes.append(recipe)

    # --- 4. Save to File ---
    with open(args.output, 'w') as f:
        json.dump(recipes, f, indent=4)

    print(f"Done! Saved to {args.output}")