import cv2
import numpy as np

def convert_image_to_hsv(input_img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(input_img, cv2.COLOR_RGB2HSV)

def generate_mask(hsv_img: np.ndarray, color_lower: np.ndarray, color_upper: np.ndarray) -> np.ndarray:
    return cv2.inRange(hsv_img, color_lower, color_upper)

def extract_contours(mask_img: np.ndarray) -> list:
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def find_road_number(image: np.ndarray) -> int:
    if image is None or image.size == 0:
        raise ValueError("Input image cannot be empty or None")

    hsv_img = convert_image_to_hsv(image)

    blue_lower_bound = np.array([100, 150, 50])
    blue_upper_bound = np.array([140, 255, 255])

    red_lower_bound_1 = np.array([0, 120, 70])
    red_upper_bound_1 = np.array([10, 255, 255])
    red_lower_bound_2 = np.array([170, 120, 70])
    red_upper_bound_2 = np.array([180, 255, 255])

    blue_mask = generate_mask(hsv_img, blue_lower_bound, blue_upper_bound)

    red_mask_1 = generate_mask(hsv_img, red_lower_bound_1, red_upper_bound_1)
    red_mask_2 = generate_mask(hsv_img, red_lower_bound_2, red_upper_bound_2)
    combined_red_mask = red_mask_1 | red_mask_2

    red_contours = extract_contours(combined_red_mask)

    img_height, img_width, _ = image.shape
    total_lanes = 5
    lane_width = img_width // total_lanes

    lanes_status = [False] * total_lanes

    for contour in red_contours:
        x_coord, y_coord, contour_width, contour_height = cv2.boundingRect(contour)
        lane_id = x_coord // lane_width
        if lane_id < total_lanes:
            lanes_status[lane_id] = True

    for lane_id, has_obstacle in enumerate(lanes_status):
        if not has_obstacle:
            return lane_id

    return -1