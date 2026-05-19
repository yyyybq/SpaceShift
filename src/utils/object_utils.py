import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from global_config import IGNORE_OBJECTS_GLOBAL, IMAGE_WIDTH_HEIGHT

def check_dependency_on_top(obj1, obj2):
    """
    Checks if obj2 is dependent on obj1, if any of the following criteria are met:
    1. obj2 is "on top of" obj1, as determined by vertical and horizontal proximity / overlap. 
    2. obj2 is "contained by" obj1 if at least 80% containment of obj2's bounding box volume within obj1's bounding box.

    Args:
        obj1 (dict): Metadata dictionary for the potential supporting object (A).
        obj2 (dict): Metadata dictionary for the potentially dependent object (B).
    """
    if not obj1 or not obj2:
        return False
    if obj1 == obj2:
        return False

    obj1_aabb = obj1.get('axisAlignedBoundingBox')
    obj2_aabb = obj2.get('axisAlignedBoundingBox')

    if not obj1_aabb or not obj2_aabb:
        print(f"Warning: Missing bounding box data for {obj1.get('objectId', 'obj1_unknown')} or {obj2.get('objectId', 'obj2_unknown')}. Cannot check dependency.")
        return False

    obj1_coords = np.array(obj1_aabb['cornerPoints'])
    obj1_min_x, obj1_min_y, obj1_min_z = np.min(obj1_coords, axis=0)
    obj1_max_x, obj1_max_y, obj1_max_z = np.max(obj1_coords, axis=0)

    obj2_coords = np.array(obj2_aabb['cornerPoints'])
    obj2_min_x, obj2_min_y, obj2_min_z = np.min(obj2_coords, axis=0)
    obj2_max_x, obj2_max_y, obj2_max_z = np.max(obj2_coords, axis=0)

    # Vertical Proximity and Horizontal Overlap
    obj2_bottom_y = obj2_min_y 
    obj1_top_y = obj1_max_y

    # Check vertical proximity: Is obj2's bottom close to obj1's top?
    epsilon_vertical = 0.05 # Adjust this value if needed for vertical alignment
    vertical_alignment = abs(obj2_bottom_y - obj1_top_y) < epsilon_vertical

    # Check horizontal overlap: Is obj2 significantly overlapping obj1 horizontally?
    horizontal_overlap_x = max(0, min(obj2_max_x, obj1_max_x) - max(obj2_min_x, obj1_min_x))
    horizontal_overlap_z = max(0, min(obj2_max_z, obj1_max_z) - max(obj2_min_z, obj1_min_z))

    # Calculate dimensions for minimum of current bounding boxes (to normalize substantial overlap check)
    min_dim_x = min(obj2_max_x - obj2_min_x, obj1_max_x - obj1_min_x)
    min_dim_z = min(obj2_max_z - obj2_min_z, obj1_max_z - obj1_min_z)

    # Avoid division by zero if dimensions are effectively zero
    substantial_overlap_x = horizontal_overlap_x > (min_dim_x * 0.1) if min_dim_x > 1e-9 else (horizontal_overlap_x > 0 and min_dim_x == 0)
    substantial_overlap_z = horizontal_overlap_z > (min_dim_z * 0.1) if min_dim_z > 1e-9 else (horizontal_overlap_z > 0 and min_dim_z == 0)

    # Ensure that obj2 is above obj1, not below
    is_above = obj2_bottom_y > obj1_top_y - epsilon_vertical

    on_top_condition = vertical_alignment and substantial_overlap_x and substantial_overlap_z and is_above

    # Volume Containment

    # Calculate dimensions for obj1 (A)
    obj1_width = obj1_max_x - obj1_min_x
    obj1_height = obj1_max_y - obj1_min_y
    obj1_depth = obj1_max_z - obj1_min_z

    # Calculate dimensions for obj2 (B)
    obj2_width = obj2_max_x - obj2_min_x
    obj2_height = obj2_max_y - obj2_min_y
    obj2_depth = obj2_max_z - obj2_min_z
    
    # Calculate volume of obj2's bounding box
    volume_B = obj2_width * obj2_height * obj2_depth

    # If obj2 has effectively no volume, it cannot be "contained" meaningfully.
    # We use a small epsilon for floating point comparison.
    if volume_B <= 1e-9:
        return False

    # Calculate the intersection (overlap) bounding box coordinates
    overlap_min_x = max(obj1_min_x, obj2_min_x)
    overlap_max_x = min(obj1_max_x, obj2_max_x)
    overlap_min_y = max(obj1_min_y, obj2_min_y)
    overlap_max_y = min(obj1_max_y, obj2_max_y)
    overlap_min_z = max(obj1_min_z, obj2_min_z)
    overlap_max_z = min(obj1_max_z, obj2_max_z)

    # Calculate overlap dimensions (ensure non-negative in case there's no overlap)
    overlap_width = max(0.0, overlap_max_x - overlap_min_x)
    overlap_height = max(0.0, overlap_max_y - overlap_min_y)
    overlap_depth = max(0.0, overlap_max_z - overlap_min_z)

    # Calculate overlap volume
    overlap_volume = overlap_width * overlap_height * overlap_depth

    # Calculate percentage contained
    percentage_contained = overlap_volume / volume_B
    
    threshold_volume_containment = 0.90

    containment_condition = percentage_contained >= threshold_volume_containment

    epsilon_plane = 0.01 # Tolerance for considering objects on the same plane
    if containment_condition:
        # Check if the bottoms of obj1 and obj2 are at roughly the same y-coordinate
        # Ex: television bbox contains tissue box bbox but both television and tissue box rest on tv stand
        if abs(obj1_min_y - obj2_min_y) < epsilon_plane:
            containment_condition = False

    return on_top_condition or containment_condition

def check_occlusion(corner_points, bbox_actual, metadata, field_of_view_deg, 
                    occlusion_threshold=0.50, return_occlusion_ratio=False
):
    """
    Check that the ratio of the area of actual 2D bounding box (from instance segmentation) to ideal 2D bounding box (from projection of axis-aligned 3D bounding box) is above the occlusion threshold.
    If 'occlusion ratio' is 0 then the object is entirely occluded.
    """
    projected_points = project_3d_to_2d(corner_points, metadata['agent'], field_of_view_deg)
    projected_points = [p for p in projected_points if p is not None and len(p) == 2] # filter out None values to avoid error

    # determine calculated 2d bounding box
    min_x, min_y = np.min(projected_points, axis=0)
    max_x, max_y = np.max(projected_points, axis=0)

    # clip it to within image bounds
    min_x = int(max(0, min_x))
    min_y = int(max(0, min_y))
    max_x = int(min(IMAGE_WIDTH_HEIGHT, max_x))
    max_y = int(min(IMAGE_WIDTH_HEIGHT, max_y))

    # determine area
    predicted_area = (max_x - min_x) * (max_y - min_y)
    bbox_predicted = [min_x, min_y, max_x, max_y]

    # determine actual area from metadata
    actual_area = calculate_area_2dbbox(bbox_actual)

    if predicted_area <= 0:
        occlusion_ratio = 0
    else:
        occlusion_ratio = actual_area / predicted_area

    if not return_occlusion_ratio:
        return occlusion_ratio >= occlusion_threshold
    else:
        return occlusion_ratio >= occlusion_threshold, occlusion_ratio, bbox_actual, bbox_predicted

def get_object_size_from_obb(corner_points):
    """
    Calculates the length, width, and height of an object from its
    object-oriented bounding box corner points.

    Args:
        corner_points (list): A list of 8 lists, where each inner list
                              represents the [x, y, z] coordinates of a
                              corner of the bounding box.

    Returns:
        list: A list [length, width, height] representing the object's dimensions,
              where length is the longest base dimension, width is the shorter
              base dimension, and height is the dimension along the y-axis.
              Returns None if the input is not valid (e.g., not enough points).
    """
    if not corner_points or len(corner_points) != 8:
        print("Error: objectOrientedBoundingBox must contain exactly 8 corner points.")
        return None

    # We can determine the dimensions by finding the unique x, y, and z values
    # from the corner points. The difference between the max and min values for
    # each axis gives us the dimension along that axis.
    
    xs = [p[0] for p in corner_points]
    ys = [p[1] for p in corner_points]
    zs = [p[2] for p in corner_points]

    x_dimension = max(xs) - min(xs)
    y_dimension = max(ys) - min(ys)
    z_dimension = max(zs) - min(zs)

    height = y_dimension
    
    base_dimensions = [x_dimension, z_dimension]
    base_dimensions.sort(reverse=True)
    length, width = base_dimensions

    return [length, width, height]

def generate_unique_samples(population, sample_size, num_unique_samples):
    """
    Generates a specified number of unique random samples from a population.

    Args:
        population (list): The list to sample from.
        sample_size (int): The number of elements in each sample (X).
        num_unique_samples (int): The number of unique samples to generate (Y).

    Returns:
        list[list]: A list of unique samples.
        
    Raises:
        ValueError: If the requested number of unique samples is not possible.
    """
    # Step 1: Check if the request is possible.
    num_possible_combinations = math.comb(len(population), sample_size)
    if num_possible_combinations == 0:
        raise ValueError(
            f"Cannot generate {num_unique_samples} unique samples. "
            f"Only {num_possible_combinations} unique combinations are possible."
        )
    elif num_possible_combinations < num_unique_samples:
        num_unique_samples = num_possible_combinations

    # Step 2: Use a set to store unique samples as tuples.
    unique_samples_set = set()
    while len(unique_samples_set) < num_unique_samples:
        # Generate a sample
        random.seed(42)
        sample = random.sample(population, sample_size)
        # Sort it and convert to a tuple to make it hashable and order-independent
        unique_samples_set.add(tuple(sorted(sample)))

    # Step 3: Convert the set of tuples back to a list of lists.
    return [list(s) for s in unique_samples_set]


# Define a wall check based on distance to nearest reachable point
def is_reachable(point, reachable_xz, threshold=0.5):
    # Find distance to closest reachable point
    dists = np.sqrt(np.sum((reachable_xz - point)**2, axis=1))
    min_dist = np.min(dists)

    # If the closest walkable point is too far away, we are likely outside a wall
    return min_dist > threshold

# Get a valid agent position y
def get_snapped_y(reachable_positions, target_x, target_z):
    # Find the point with the minimum Euclidean distance in XZ plane
    best_p = min(reachable_positions, 
                 key=lambda p: (p['x'] - target_x)**2 + (p['z'] - target_z)**2)
    
    return best_p['y']

def is_colliding(point, ignore_id, all_objects, tolerance=0.2):
    px, pz = point
    for other in all_objects:
        # if other['id'] == ignore_id: continue

        # Check against DILATED AABB of the obstacle
        # [min_x - r, min_z - r, max_x + r, max_z + r]
        ox1, oz1, ox2, oz2 = other['aabb']
        if (px > ox1 - tolerance and px < ox2 + tolerance and
            pz > oz1 - tolerance and pz < oz2 + tolerance):
            return True
    return False


# Extract 2D bounding box information from object metadata
def get_objects_2d(objects_metadata):
  objects_2d = []
  for obj in objects_metadata:
      if not obj['objectOrientedBoundingBox']: continue

      IGNORE_OBJ = ['floor']
      if obj['name'].split('_')[0].lower() in IGNORE_OBJ: continue
      if obj.get('objectType', '').lower() in IGNORE_OBJECTS_GLOBAL:
          continue

      # Extract 3D corners and flatten to 2D (X, Z)
      wc_corners_3d = obj['objectOrientedBoundingBox']['cornerPoints']

      # PROJECT TO 2D: Drop the Y-coordinate (height) to get a top-down view
      x_coords = [corner[0] for corner in wc_corners_3d]
      z_coords = [corner[2] for corner in wc_corners_3d]
      y_coords = [corner[1] for corner in wc_corners_3d]

      # Calculate 2D Bounding Box from these flattened points
      min_x = min(x_coords)
      max_x = max(x_coords)
      min_z = min(z_coords)
      max_z = max(z_coords)
      min_y = min(y_coords)
      max_y = max(y_coords)

      # --- 4. Plot on Canvas ---
      width = max_x - min_x
      length = max_z - min_z

      # Calculate 2D AABB [min_x, min_z, max_x, max_z]
      aabb = [min_x, min_z, max_x, max_z]

      # Calculate Centroid and Encompassing Radius (distance to furthest corner)
      centroid = np.array([(aabb[0] + aabb[2]) / 2.0, (aabb[1] + aabb[3]) / 2.0])
      # Radius of the object itself (from centroid to a corner)
      obj_radius = np.max(np.sqrt((x_coords - centroid[0])**2 + (z_coords - centroid[1])**2))

      objects_2d.append({
          'id': obj['objectId'],
          'name': obj['name'],
          'type': obj['objectType'],
          'aabb': aabb,
          'centroid': centroid,
          'centroid_height': (min_y + max_y) / 2,
          'obj_radius': obj_radius,
          '2d_width': width,
          '2d_length': length,
      })
  return objects_2d


def project_3d_to_2d(points_3d, agent, field_of_view_deg):
    """
    Transforms 3D points from world space to 2D pixel coordinates.
    """
    image_width, image_height = IMAGE_WIDTH_HEIGHT, IMAGE_WIDTH_HEIGHT
    camera_position = np.array(list(agent['position'].values()))

    # Use 'cameraHorizon' for pitch and standard 'y' rotation for yaw.
    # We use 'yx' order (Yaw then Pitch).
    agent_yaw = agent['rotation']['y']
    camera_pitch = agent['cameraHorizon']
    
    # Create rotation: Yaw x Pitch
    # Create separate rotations
    r_yaw = R.from_euler('y', agent_yaw, degrees=True)
    r_pitch = R.from_euler('x', camera_pitch, degrees=True)
    r_cam_to_world = r_yaw * r_pitch

    # World to camera matrix
    view_matrix = np.eye(4)
    view_matrix[:3, :3] = r_cam_to_world.inv().as_matrix()
    view_matrix[:3, 3] = -view_matrix[:3, :3] @ camera_position

    field_of_view_rad = np.radians(field_of_view_deg)

    focal_length = (image_width / 2.0) / np.tan(field_of_view_rad / 2.0)
    cx = image_width / 2.0
    cy = image_height / 2.0

    # Camera Intrinsics Matrix K
    K = np.array([
        [focal_length, 0, cx],
        [0, -focal_length, cy],
        [0, 0, 1]
    ])

    points_2d = []
    for point in points_3d:
        point_homogeneous = np.append(point, 1.0)
        point_camera_space = (view_matrix @ point_homogeneous)[:3]

        # Check if the point is in front of the camera
        if point_camera_space[2] > 0.1: # Near clipping plane
            # Project from 3D camera space to 2D image plane
            point_image_space = K @ point_camera_space

            # Perspective divide to get pixel coordinates
            w = point_camera_space[2]
            x_pixel = point_image_space[0] / w
            y_pixel = point_image_space[1] / w

            points_2d.append((x_pixel, y_pixel))
        else:
            points_2d.append(None) # Point is behind the camera

    return points_2d

def calculate_volume(size_dict):
    return size_dict['x'] * size_dict['y'] * size_dict['z']

def calculate_area_2dbbox(bbox):
    ul_x, ul_y, lr_x, lr_y = bbox
    width = lr_x - ul_x
    height = lr_y - ul_y
    return width * height

