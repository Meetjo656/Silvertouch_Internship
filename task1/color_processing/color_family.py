import numpy as np
import colour
from matplotlib.colors import XKCD_COLORS

def hex_to_rgb(hex_str):
    """
    Helper to convert standard #RRGGBB HEX string to RGB numpy array.
    """
    hex_str = hex_str.lstrip('#')
    return np.array(
        [int(hex_str[i:i+2], 16) for i in (0, 2, 4)],
        dtype=np.float64
    )

# Dynamic Perceptual Semantic Color Database Initialization
SEMANTIC_NAMES = []
SEMANTIC_LABS = []

for key, hex_code in XKCD_COLORS.items():
    name = key.replace('xkcd:', '')
    
    # Convert HEX -> RGB [0, 255] -> normalized [0, 1]
    center_rgb = hex_to_rgb(hex_code) / 255.0
    
    # Convert to LAB via colour-science
    xyz_center = colour.sRGB_to_XYZ(center_rgb)
    lab_center = colour.XYZ_to_Lab(xyz_center)
    
    SEMANTIC_NAMES.append(name)
    SEMANTIC_LABS.append(lab_center)

# Store coordinates in a vectorized NumPy array for fast operations
SEMANTIC_LABS = np.array(SEMANTIC_LABS)

def get_color_family(rgb):
    """
    Map an RGB color to a stable semantic Color Family dynamically using colour-science
    and the matplotlib XKCD color database via dynamic nearest-neighbor DeltaE (CIE 2000) matching.
    """
    # Normalize input RGB to [0, 1] for colour-science
    rgb_norm = np.array(rgb, dtype=np.float64) / 255.0
    
    # Convert input to LAB
    xyz_target = colour.sRGB_to_XYZ(rgb_norm)
    lab_target = colour.XYZ_to_Lab(xyz_target)

    # Calculate vectorized CIEDE2000 DeltaE distance against all 949 semantic palette colors
    distances = colour.delta_E(lab_target, SEMANTIC_LABS, method='CIE 2000')
    
    # Find the nearest semantic neighbor
    min_idx = np.argmin(distances)
    best_name = SEMANTIC_NAMES[min_idx]

    # Return the semantic name in uppercase
    return best_name.upper()

def get_meta_family(exact_name):
    """
    Map an exact semantic XKCD color name to a broader Meta-Family.
    Uses substring keyword matching of the lowercase name for fully dynamic grouping.
    """
    name_lower = exact_name.lower()
    
    # 1. Light Green / Teal / Seafoam (perceptual groupings)
    if any(k in name_lower for k in [
        "blue green", "greeny blue", "seafoam", "aqua green", 
        "teal", "mint", "turquoise", "light green", "pale green", 
        "soft green", "sea green", "aquamarine", "emerald", "dark mint"
    ]):
        return "LIGHT GREEN"
        
    # 2. Dark Red (perceptual groupings)
    elif any(k in name_lower for k in [
        "maroon", "wine", "burgundy", "crimson", "dark red"
    ]):
        return "DARK RED"
        
    # 3. Light Blue (perceptual groupings)
    elif any(k in name_lower for k in [
        "sky blue", "baby blue", "powder blue", "pale cyan", 
        "light blue", "pale blue", "soft blue"
    ]):
        return "LIGHT BLUE"
        
    # 4. Standard Broader Color Categories Fallbacks
    elif "green" in name_lower or "olive" in name_lower or "lime" in name_lower:
        return "GREEN"
    elif "blue" in name_lower or "navy" in name_lower or "ultramarine" in name_lower:
        return "BLUE"
    elif "red" in name_lower or "rose" in name_lower:
        return "RED"
    elif "yellow" in name_lower or "gold" in name_lower:
        return "YELLOW"
    elif any(k in name_lower for k in [
        "purple", "violet", "lavender", "plum", "magenta", "lilac"
    ]):
        return "PURPLE"
    elif "pink" in name_lower:
        return "PINK"
    elif any(k in name_lower for k in ["orange", "peach", "apricot"]):
        return "ORANGE"
    elif any(k in name_lower for k in [
        "brown", "tan", "khaki", "beige", "taupe", "copper"
    ]):
        return "BROWN"
    elif "grey" in name_lower or "gray" in name_lower:
        return "GRAY"
    elif "black" in name_lower or "charcoal" in name_lower:
        return "BLACK"
    elif "white" in name_lower or "ivory" in name_lower:
        return "WHITE"
    elif "cyan" in name_lower or "aqua" in name_lower:
        return "CYAN"
        
    return "OTHER"
