# importing relevant libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pytesseract
import easyocr
from filters import grayscale

# global vars
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Kelly\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# returns bounding boxes and text predictions from EasyOCR
def easyocr_text_pred(img):
    reader = easyocr.Reader(['en'])
    coords = reader.readtext(img)
    return coords

def check_bounds(coord, bound):
    return max(0, min(int(coord), bound))

# reshapes easyocr predictions into set format
# ensure coordinates do not exceed the height and width of image
def recons_eocr_coords(coordinates, width, height):
    reconstruct = []
    for point in coordinates:
        coords, text, _ = point
        reconstruct.append((text, np.array([[check_bounds(coord[0], width), 
                     check_bounds(coord[1], height)] 
                    for coord in coords], dtype=int)))
    return reconstruct

def rescale(img, fx, fy):
    return cv2.resize(img, None, fx=fx, fy=fy, interpolation = cv2.INTER_CUBIC)

def order_points(rectangle_coords):
    if isinstance(rectangle_coords, np.ndarray):
        rectangle_coords = rectangle_coords.tolist()
    # Find the three coordinates with the smallest y-axis values
    sorted_coords = sorted(rectangle_coords, key=lambda coord: coord[1])[:3]

    # Calculate distances between the three points using Pythagoras' theorem
    def distance(x1, y1, x2, y2):
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    
    distances = [
        distance(sorted_coords[i][0], sorted_coords[i][1], sorted_coords[j][0], sorted_coords[j][1])
        for i in range(3)
        for j in range(i + 1, 3)
    ]
    # Find the second longest distance and return the indices of the points
    second_longest_distance = sorted(distances)[-2]
    lst = [(0,1),(1,2),(0,2)]
    indexes = (0,0)
    for x in lst:
        if distance(sorted_coords[x[0]][0], sorted_coords[x[0]][1], sorted_coords[x[1]][0], sorted_coords[x[1]][1]) == second_longest_distance:
            indexes = x
            break
        else:
            continue
        
    coordinates = sorted((sorted_coords[indexes[0]],sorted_coords[indexes[1]]), key=lambda coord: coord[0])[0]
    index = rectangle_coords.index(coordinates)
    if index == 0:
        return rectangle_coords
    else:
        # Reorder the points based on the second longest distance
        rectangle_coords = rectangle_coords[index:] + rectangle_coords[:index]
    return rectangle_coords

def check_angle(coordinates, slant, scale):
    word_list = []
    for text, box in coordinates:
        if abs(box[0][0] - box[3][0]) < slant:
            word_list.append([text.strip(), box/scale, "not slanted"])

        else:
            points = order_points(box)
            word_list.append([text.strip(), np.divide(points, scale), "slanted"])
    return word_list

def get_coordinates(filt, hths, wths, slant):
    min = 5
    scale = 2

    im = cv2.imread(img_path)
    im = rescale(im, scale, scale)
    filt_img = grayscale(im)
    height, width, _ = im.shape
    print(height, width)

    eocr_pred_set = easyocr_text_pred(filt_img)
    coords_eocr = recons_eocr_coords(eocr_pred_set, width, height)

    final_coords = check_angle(coords_eocr, slant, scale)

    return final_coords