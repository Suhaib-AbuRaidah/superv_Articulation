import numpy as np
from sklearn.neighbors import KDTree

def find_contact_points(cloud1, cloud2, predicted_pivot, k_neighbors=10):
    """
    Finds contact points between cloud1 and cloud2, then returns the contact point
    (from the concatenated set of both clouds' contact points) closest to predicted_pivot.

    Returns:
      contact_points_1: (N,3)
      contact_points_2: (N,3)
      contact_distances: (N,)
      total_contacts: (2N,3)
      closest_contact_point: (3,) or None if no contacts
      closest_contact_dist: float or None
      closest_contact_index: int or None (index in total_contacts)
    """
    # Build KD-tree for cloud2
    tree2 = KDTree(cloud2)

    # 1-NN from cloud1 -> cloud2
    distances, indices = tree2.query(cloud1, k=1)
    distances = distances.squeeze(-1)  # (N,)
    indices = indices.squeeze(-1)      # (N,)

    adaptive_threshold = np.percentile(distances, 1)

    contact_points_1 = []
    contact_points_2 = []
    contact_distances = []

    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if dist < adaptive_threshold:
            # local neighborhood around matched point in cloud2
            nn_idx = tree2.query(cloud2[idx].reshape(1, -1), k=k_neighbors)[1].squeeze(0)
            local_points = cloud2[nn_idx]

            # PCA normal
            cov = np.cov(local_points.T)
            _, eigenvectors = np.linalg.eigh(cov)
            normal = eigenvectors[:, 0]

            # point-to-plane distance
            vec = cloud1[i] - cloud2[idx]
            pt_plane_dist = abs(np.dot(vec, normal))

            if pt_plane_dist < adaptive_threshold:
                contact_points_1.append(cloud1[i])
                contact_points_2.append(cloud2[idx])
                contact_distances.append(pt_plane_dist)

    contact_points_1 = np.asarray(contact_points_1)
    contact_points_2 = np.asarray(contact_points_2)
    contact_distances = np.asarray(contact_distances)

    # If no contacts found, return safely
    if contact_points_1.size == 0 or contact_points_2.size == 0:
        return contact_points_1, contact_points_2, contact_distances, np.empty((0, 3)), None, None, None

    total_contacts = np.concatenate([contact_points_1, contact_points_2], axis=0)  # (2N,3)

    # Find closest contact point to predicted pivot
    pivot = np.asarray(predicted_pivot).reshape(1, 3)
    tree_contacts = KDTree(total_contacts)
    d_closest, idx_closest = tree_contacts.query(pivot, k=1)

    closest_contact_dist = float(d_closest[0, 0])
    closest_contact_index = int(idx_closest[0, 0])
    closest_contact_point = total_contacts[closest_contact_index]

    return closest_contact_point