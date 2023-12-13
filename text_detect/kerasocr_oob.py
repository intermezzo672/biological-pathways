# importing relevant libraries
import numpy as np
import keras_ocr
import cv2
import pytesseract
from util import grayscale, rescale

# could potentially take this out depending
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Kelly\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def draw_kocr_boxes(coordinates, img):
    for text, coord in coordinates:
        (topleft, topright, bottomright, bottomleft) = coord
        tl_x,tl_y = (int(topleft[0]), int(topleft[1]))
        tr_x, tr_y = (int (topright[0]), int(topright[1]))
        bl_x,bl_y = (int(bottomleft[0]), int(bottomleft[1]))
        br_x,br_y = (int(bottomright[0]), int(bottomright[1]))
        cv2.line(img, (tl_x,tl_y), (tr_x,tr_y), (0, 0, 255), 2)
        cv2.line(img, (tr_x,tr_y), (br_x,br_y), (0, 0, 255), 2)
        cv2.line(img, (tl_x,tl_y), (bl_x,bl_y), (0, 0, 255), 2)
        cv2.line(img, (bl_x,bl_y), (br_x,br_y), (0, 0, 255), 2)
        
    cv2.imwrite("C:\\Users\\Kelly\\OneDrive - Yale University\\Fall2023\\CPSC490\\Foundational Code\\output\\kerasocr-bbonly.jpg", img)

def kerasocr_text_pred(img_path):
    pipeline = keras_ocr.pipeline.Pipeline()
    img = keras_ocr.tools.read(img_path)
    height, width, channels = img.shape 
    prediction_groups = pipeline.recognize([img])
    return prediction_groups[0], height, width, channels, img

def check_bounds(coord, bound):
    return max(0, min(int(coord), bound))

# reshapes kerasocr predictions into set format
# ensure coordinates do not exceed the height and width of image
def recons_kocr_coords(coordinates, width, height):
    reconstruct = []
    for word, coords in coordinates: 
        reconstruct.append((word, np.array([[check_bounds(coord[0], width), 
                     check_bounds(coord[1], height)] 
                    for coord in coords], dtype=int)))
    return reconstruct

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

def check_angle(coordinates, slant):
    word_list = []
    for text, box in coordinates:
        if abs(box[0][0] - box[3][0]) < slant:
            word_list.append([text.strip(), box, "not slanted"])

        else:
            points = order_points(box)
            word_list.append([text.strip(), points, "slanted"])
    return word_list

def get_coordinates(img_path, filt, hths, wths, slant):
    im = cv2.imread(img_path)
    kocr_pred_set, height, width, _, _ = kerasocr_text_pred(img_path)
    coords_kocr = recons_kocr_coords(kocr_pred_set, width, height)
    draw_kocr_boxes(coords_kocr, im)
    final_coords = check_angle(coords_kocr, slant)

    return final_coords