import cv2
import numpy as np

def rotate(image: np.ndarray, pivot: tuple, angle_degrees: float) -> np.ndarray:
    (img_height, img_width) = image.shape[:2]
    rotation_radians = np.deg2rad(angle_degrees)
    pivot_x, pivot_y = pivot

    adjusted_width = int(abs(img_width * np.cos(rotation_radians)) + abs(img_height * np.sin(rotation_radians)))
    adjusted_height = int(abs(img_height * np.cos(rotation_radians)) + abs(img_width * np.sin(rotation_radians)))

    rotation_matrix = cv2.getRotationMatrix2D(pivot, angle_degrees, 1)
    rotation_matrix[0, 2] += (adjusted_width - img_width) / 2
    rotation_matrix[1, 2] += (adjusted_height - img_height) / 2 + pivot_y / 2

    output_image = cv2.warpAffine(image, rotation_matrix, (adjusted_width, adjusted_height))

    return output_image

def apply_warpAffine(input_image, source_points, destination_points) -> np.ndarray:
    affine_matrix = cv2.getAffineTransform(np.float32(source_points), np.float32(destination_points))
    (image_height, image_width) = input_image.shape[:2]

    corner_points = np.array([
        [0, 0],
        [image_width, 0],
        [0, image_height],
        [image_width, image_height]
    ], dtype=np.float32)
    transformed_corners = cv2.transform(np.array([corner_points]), affine_matrix)[0]

    min_corner_x = int(np.min(transformed_corners[:, 0]))
    min_corner_y = int(np.min(transformed_corners[:, 1]))
    max_corner_x = int(np.max(transformed_corners[:, 0]))
    max_corner_y = int(np.max(transformed_corners[:, 1]))

    output_width = max_corner_x - min_corner_x
    output_height = max_corner_y - min_corner_y

    affine_matrix[0, 2] -= min_corner_x
    affine_matrix[1, 2] -= min_corner_y

    result_image = cv2.warpAffine(input_image, affine_matrix, (output_width, output_height))

    return result_image
