import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
# from skimage import measure, color, io
# Load your image
# img = cv2.imread('437-1-03.tif')
# print(img[::15, ::15,:])
# # check if r == g == b for all pixels in the image
# print("the truth is:", np.all(img[:,:,0] == img[:,:,1]) and np.all(img[:,:,2] == img[:,:,1]))
# print(img.shape)
# Convert to grayscale


def dirs_to_point(point: np.ndarray, points: np.ndarray):
    return points - point



def dirs_to_line_orth(startpoint: np.ndarray, direction_vector: np.ndarray, points: np.ndarray):
    line_direction = direction_vector / np.linalg.norm(direction_vector)
    orth_direction = np.array([line_direction[1], -line_direction[0]])
    orth_direction /= np.linalg.norm(orth_direction)
    return np.dot(points - startpoint, orth_direction).reshape(-1, 1) * orth_direction

def scalars_to_line_orth(startpoint: np.ndarray, direction_vector: np.ndarray, points: np.ndarray):
    line_direction = direction_vector / np.linalg.norm(direction_vector)
    orth_direction = np.array([line_direction[1], -line_direction[0]])
    orth_direction /= np.linalg.norm(orth_direction)
    return np.dot(points - startpoint, orth_direction).reshape(-1, 1)


def dirs_to_point_with_line(startpoint: np.ndarray, direction_vector: np.ndarray, points: np.ndarray):
    line_direction = direction_vector / np.linalg.norm(direction_vector)
    return np.dot(points - startpoint, line_direction) * line_direction

def scalars_to_with_line(startpoint: np.ndarray, direction_vector: np.ndarray, points: np.ndarray):
    line_direction = direction_vector / np.linalg.norm(direction_vector)
    print("line_direction.shape", line_direction.shape, (points - startpoint).shape)
    print(startpoint, points[0])
    print(np.dot(points - startpoint, line_direction))
    return (np.dot(points - startpoint, line_direction)).reshape(-1, 1)



def arrow(canvas, startpoint, endpoint, color = (0, 0, 255), thickness = 2):
    startpoint = np.array(startpoint) 
    endpoint = np.array(endpoint)
    cv2.arrowedLine(canvas, tuple(startpoint[::-1].astype(np.int32)), tuple(endpoint[::-1].astype(np.int32)), color, thickness)

def line(canvas, startpoint, endpoint, color = (0, 0, 255), thickness = 2):
    startpoint = np.array(startpoint) 
    endpoint = np.array(endpoint)
    cv2.line(canvas, tuple(startpoint[::-1].astype(np.int32)), tuple(endpoint[::-1].astype(np.int32)), color, thickness)



# _, thresh_old = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
# # _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
# thresh = thresholds[0]
# kernel = np.ones((5,5),np.uint8)
# eroded = cv2.erode(thresh, kernel, iterations = 1)
# re_thresh = cv2.adaptiveThreshold(eroded,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,11,2)
# cv2.imshow("gray", gray)
# cv2.waitKey(0)
# cv2.imshow("blurred", blurred)
# cv2.waitKey(0)
# cv2.imshow("thresh_old", thresh_old)
# cv2.waitKey(0)
# cv2.imshow("thresh", thresh)
# cv2.waitKey(0)
# cv2.imshow("eroded", eroded)
# cv2.waitKey(0)
# cv2.imshow("re_thresh", re_thresh)
# cv2.waitKey(0)
def show(img, s="img"):
    if isinstance(img, list):
        for i in img:
            show(i, s + str(i))
    else:
        if img.shape[-1] == 1:
            img = img[:, :, 0]
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
            plt.plot()
        # elif img.shape[-1] == 3:
        #     img = img[:, :, 0]
        #     plt.imshow(img)
        #     plt.plot()
        # else:
        cv2.imshow(s, img)
        cv2.waitKey(0)
thresholds = []
# for threshold_levels in [120, 155, 180, 210, 240]:
#     thresholds.append(do_threshold(gray, threshold_levels))


def combine_thresh_masks(masks):
    scale = 255 // len(masks)
    gradient = np.zeros_like(masks[0])
    binary = np.zeros_like(masks[0])
    for m in masks:
        gradient += m // 255 * scale
        binary += m
        binary[binary == 0] = 254
        binary[binary == 255] = 0
        binary[binary == 254] = 255
    return gradient, binary


# print(thresh.dtype)

# Find contours
# thresh_8uc1 = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

def get_regions_and_add_function():
    regions = {}
    def add_region(region):
        s = np.sum(region)
        print(s)
        s = int(s)
        if s in regions:
            regions[s].append(region)
        else:
            regions[s] = [region]
    return regions, add_region


def get_contour_masks(thresh, level = None):
    regions, add_region = get_regions_and_add_function()
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    parsed, top = parse_hierarchy(contours, hierarchy, level = level)
    # Loop over the contours
    for i, contour in enumerate(contours):
        # Create a black image with the same size as your original image
        mask = np.zeros_like(thresh)
        # Fill the contour in the mask image
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
        parsed[i].region_mask = mask
        # Bitwise-and with the original image to get  the region corresponding to the contour
        region = cv2.bitwise_and(thresh, mask)
        # Append region to list
        add_region(mask)
    return regions, parsed, top

def remove_bottom_bar(img):
    return img[:-50, :]

def partition_contours(region_dict, minsum = 1200, maxsum = None):
    high = []
    low = []
    mid = []
    for i in region_dict.keys():
        if maxsum is not None and i > maxsum:
            high += region_dict[i]    
        elif minsum is not None and i < minsum:
            low += region_dict[i]
        else:
            mid += region_dict[i]
    return mid, low, high

class Contour():
    def __init__(self, contour, parent = None, level = None):
        self.contour = contour
        self.children = []
        self.parent = parent
        self.region_mask = None
        self.level = level

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

import sys
sys.setrecursionlimit(2000)
    
def parse_hierarchy(contours, hierarchy, minimum_sum = 1000, level = None):
    """
    c: List[np.ndarray]; list of contours  
    h: np.ndarray; hierarchy of contours; h.shape == (1, len(c), 4)
    """
    print("hierarchy.shape", hierarchy.shape)
    checked = np.zeros(len(hierarchy[0, :, 0]) + 1)
    checked[-1] = 1
    C = [None] * len(hierarchy[0, :, 0]) + [None] #+none for top level nodes
    i = 0
    top = set()
    while not np.all(checked):
        # def check_next(i):
        def check(i):
            print("checking", i, checked.shape)
            if checked[i]:
                return
            h = hierarchy[0, i, :]
            c = contours[i]
            sib_next, sib_prev, child_first, parent = h[0], h[1], h[2], h[3]
            if not checked[parent]:
                check(parent)
            checked[i] = 1
            C[i] = Contour(c, C[parent])
            if parent != -1:
                C[parent].add_child(C[i])
            else:
                top.add(C[i])
            checked[i] = 1
            if not checked[child_first]:
                check(child_first)
            if not checked[sib_next]:
                check(sib_next)
            if not checked[sib_prev]:
                check(sib_prev)
        if checked[i]:
            i = (i + 1) % len(checked)
        else:
            check(i)
    return C, top


def label_mask(mask):
    structure = [[1,1,1],[1,1,1],[1,1,1]]
    labeled_mask, num_labels = ndimage.label(mask, structure=structure)
    colored = color.label2rgb(labeled_mask, bg_label=0)
    return labeled_mask, num_labels, colored


def make_labeled_contour_aggregation(gray, region_masks):
    z = np.zeros_like(gray)
    z = z.astype(np.int16)
    z[:,:] = -1
    for i, mask in enumerate(region_masks):
        pass


import os
def all_imagestrs():
    for f in os.listdir("."):
        if f.endswith(".tif"):
            yield f