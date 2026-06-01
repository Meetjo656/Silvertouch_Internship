import numpy as np

def get_dominant_color(cluster_info):
    """
    Selects the dominant semantic color from the cluster info list,
    which is sorted by Importance Score in descending order.
    """
    if not cluster_info:
        return np.array([0, 0, 0], dtype=np.uint8)
    return np.array(cluster_info[0]["color"], dtype=np.uint8)
