import numpy as np

def binarize_hue(hue_val):
    """
    Binarize hue into Warm (1) vs Cool (0).
    Assuming hue is in [0, 1].
    Warm colors: Red, Orange, Yellow, Magenta-ish?
    Cool colors: Green, Cyan, Blue.
    
    Standard HSV hue:
    0.0 - 0.16: Red to Yellow (Warm)
    0.16 - 0.5: Green to Cyan (Cool)
    0.5 - 0.83: Cyan to Blue to Purple (Cool/Mixed?)
    0.83 - 1.0: Purple to Red (Warm)
    
    Let's define a simple split:
    Warm: [0, 0.25] U [0.75, 1.0]
    Cool: [0.25, 0.75]
    """
    if hue_val < 0.25 or hue_val >= 0.75:
        return 1 # Warm
    else:
        return 0 # Cool

def binarize_scale(scale_val):
    """
    Binarize scale into Small (0) vs Large (1).
    The raw values in 3dshapes.h5 seem to be in [0.75, 1.25] rather than [0, 1].
    [0.75, 0.82, 0.89, 0.96] -> Small
    [1.03, 1.10, 1.17, 1.25] -> Large
    Midpoint is 1.0.
    """
    return 1 if scale_val >= 1.0 else 0

def get_concept_labels(labels_raw):
    """
    Process raw labels (N, 6) into dictionary of concept labels.
    Raw Indices:
    0: floor_hue
    1: wall_hue
    2: object_hue
    3: scale
    4: shape (4 values: 0,1,2,3)
    5: orientation
    """
    N = labels_raw.shape[0]
    
    floor_hue = labels_raw[:, 0]
    wall_hue = labels_raw[:, 1]
    obj_hue = labels_raw[:, 2]
    scale = labels_raw[:, 3]
    shape = labels_raw[:, 4].astype(int)
    
    # Binarize attributes
    floor_warm = np.array([binarize_hue(h) for h in floor_hue])
    wall_warm = np.array([binarize_hue(h) for h in wall_hue])
    obj_warm = np.array([binarize_hue(h) for h in obj_hue])
    scale_large = np.array([binarize_scale(s) for s in scale])
    
    # --- Model Training Labels ---
    
    # M.1: 64 labels = Scale(2) * Floor(2) * Wall(2) * Obj(2) * Shape(4)
    # We can encode this as a base-10 integer for simplicity
    # Order: Shape(4) * 16 + Scale(2) * 8 + Floor(2) * 4 + Wall(2) * 2 + Obj(2) * 1
    # Range: 0 to 63
    y_m1 = (shape * 16 + scale_large * 8 + floor_warm * 4 + wall_warm * 2 + obj_warm * 1).astype(int)
    
    # M.2: 8 labels = Shape(4) * Scale(2)
    # Order: Shape(4) * 2 + Scale(2)
    y_m2 = (shape * 2 + scale_large).astype(int)
    
    # M.3: 8 labels = Floor(2) * Wall(2) * Obj(2)
    # Order: Floor(2) * 4 + Wall(2) * 2 + Obj(2)
    y_m3 = (floor_warm * 4 + wall_warm * 2 + obj_warm).astype(int)
    
    # --- Probing Task Labels ---
    
    # P.1: Binarized colors (3 separate tasks or combined?)
    # User request: "task P.1: binarized colors (floor, wall, object)"
    # We'll store them individually.
    y_p1_floor = floor_warm
    y_p1_wall = wall_warm
    y_p1_obj = obj_warm
    
    # P.2: Binarized scale and 4 shapes
    y_p2_scale = scale_large
    y_p2_shape = shape
    
    return {
        'M1': y_m1,
        'M2': y_m2,
        'M3': y_m3,
        'P1_floor': y_p1_floor,
        'P1_wall': y_p1_wall,
        'P1_obj': y_p1_obj,
        'P2_scale': y_p2_scale,
        'P2_shape': y_p2_shape
    }

