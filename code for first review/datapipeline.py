import numpy as np
import matplotlib.pyplot as plt

def simulate_drr(ct_volume, projection_plane='PA', voxel_size=(1, 1, 1)):
    """
    Simulates a digitally reconstructed radiograph (DRR) from a CT volume.
    
    Args:
        ct_volume (numpy.ndarray): 3D CT data array (z, y, x).
        projection_plane (str): 'PA' for posterior-anterior, 'LAT' for lateral.
        voxel_size (tuple): Voxel size in mm (z, y, x).
    
    Returns:
        numpy.ndarray: 2D DRR image.
    """
    if projection_plane == 'PA':
        drr = np.sum(ct_volume, axis=0)  # Collapse along the z-axis.
    elif projection_plane == 'LAT':
        drr = np.sum(ct_volume, axis=1)  # Collapse along the y-axis.
    else:
        raise ValueError("Invalid projection_plane. Use 'PA' or 'LAT'.")
    
    drr = drr / np.max(drr)  # Normalize for visualization.
    return drr