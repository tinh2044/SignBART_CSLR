import numpy as np

def rotate_keypoints(keypoints, origin, angle_degrees):
    
    angle_radians = np.radians(angle_degrees)

    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)

    rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

    shifted_points = keypoints - np.array(origin)

    rotated_points = np.einsum('ij,klj->kli', rotation_matrix, shifted_points)

    rotated_keypoints = rotated_points + np.array(origin)

    return rotated_keypoints   
    
def flip_keypoints(keypoints):
    flipped_keypoints = keypoints.copy()
   
    flipped_keypoints[..., 0] = 1 - flipped_keypoints[..., 0]
    
    return flipped_keypoints

