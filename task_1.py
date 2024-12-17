import cv2
import numpy as np
from collections import deque

def process_image(input_image: np.ndarray) -> np.ndarray:
    grayscale = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(grayscale, 128, 255, cv2.THRESH_BINARY)
    return thresholded

def locate_start_and_exit(thresholded_img: np.ndarray) -> tuple:
    img_height, img_width = thresholded_img.shape
    entry_point = (0, np.where(thresholded_img[0] == 255)[0][0])
    exit_point = (img_height - 1, np.where(thresholded_img[-1] == 255)[0][0])
    return entry_point, exit_point

def compute_path(thresholded_img: np.ndarray, entry_point: tuple, exit_point: tuple) -> list:
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    search_queue = deque([entry_point])
    explored = set([entry_point])
    previous_nodes = {entry_point: None}

    while search_queue:
        current_node = search_queue.popleft()

        if current_node == exit_point:
            break

        for direction in directions:
            neighbor_node = (current_node[0] + direction[0], current_node[1] + direction[1])

            if (0 <= neighbor_node[0] < thresholded_img.shape[0]) and \
               (0 <= neighbor_node[1] < thresholded_img.shape[1]) and \
               thresholded_img[neighbor_node] == 255 and neighbor_node not in explored:
                search_queue.append(neighbor_node)
                explored.add(neighbor_node)
                previous_nodes[neighbor_node] = current_node

    path_coordinates = []
    if exit_point in previous_nodes:
        step = exit_point
        while step:
            path_coordinates.append(step)
            step = previous_nodes[step]
        path_coordinates.reverse()

    return path_coordinates

def find_way_from_maze(image: np.ndarray) -> tuple:
    thresholded_img = process_image(image)
    entry_point, exit_point = locate_start_and_exit(thresholded_img)
    path_coordinates = compute_path(thresholded_img, entry_point, exit_point)

    if path_coordinates:
        x_positions, y_positions = zip(*path_coordinates)
        return np.array(x_positions), np.array(y_positions)
    else:
        return None
