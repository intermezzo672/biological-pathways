# importing relevant libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pytesseract
from pytesseract import Output
import easyocr
from util import draw_ocr_boxes, add_white_bg, grayscale, binthres, otsu

# global vars
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Kelly\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def easyocr_text_pred(img, hths, wths):
    reader = easyocr.Reader(['en'])
    coords = reader.readtext(img, height_ths=hths, width_ths=wths)
    return coords

def check_bounds(coord, bound):
    return max(0, min(int(coord), bound))

# reshapes easyocr predictions into set format
# ensure coordinates do not exceed the height and width of image
def recons_eocr_coords(coordinates, width, height):
    reconstruct = []
    for point in coordinates:
        coords, _, _ = point
        reconstruct.append(np.array([[check_bounds(coord[0], width), 
                     check_bounds(coord[1], height)] 
                    for coord in coords], dtype=int))
    return reconstruct


# Convert to a numpy array (the correct format)
def convert_output(text_coords):
    converted_text_coords = []
    
    for word, coords in text_coords:
        converted_coords = np.array(coords, dtype=np.float32)
        converted_text_coords.append((word, converted_coords))
    
    return converted_text_coords

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

def check_angle(coordinates, img, slant, height, width):
    word_list = []
    reader = easyocr.Reader(['en'])
    for box in coordinates:
        if abs(box[0][0] - box[3][0]) < slant:
            roi = img[max(0, int(box[0][1])):min(height, int(box[2][1])), int(box[0][0]):int(box[1][0])]
            custom_config = r'--oem 1 --psm 7 -c preserve_interword_spaces=1'
            border_im = add_white_bg(roi, 100)
            d = pytesseract.image_to_data(border_im, output_type=Output.DICT)
            (x, y, w, h) = (d['left'][0], d['top'][0], d['width'][0], d['height'][0])
            new_roi = border_im[y:y+h, x:x+w] 
            coords = reader.readtext(new_roi, height_ths=1, width_ths=1)
            
            if coords:
                word_list.append([coords[0][1].strip(), box/2, "not slanted"])

        else:
            points = order_points(box)
            corners = np.array(points, dtype=np.float32)

            roi_width = int(((corners[1][0] - corners[0][0]) ** 2 + (corners[1][1] - corners[0][1]) ** 2) ** 0.5)
            roi_height = int(((corners[3][0] - corners[0][0]) ** 2 + (corners[3][1] - corners[0][1]) ** 2) ** 0.5)

            rectified_corners = np.array([[0, 0], [roi_width, 0], [roi_width, roi_height], [0, roi_height]], dtype=np.float32)

            transformation_matrix = cv2.getPerspectiveTransform(corners, rectified_corners)
            
            warp_roi = cv2.warpPerspective(img, transformation_matrix, (roi_width, roi_height))
            border_im = add_white_bg(warp_roi, 100)

            # Apply OCR on the adjusted ROI
            custom_config = r'--oem 1 --psm 7 -c preserve_interword_spaces=1'
            results = pytesseract.image_to_data(border_im, config=custom_config, output_type=pytesseract.Output.DICT)
            # results4 = pytesseract.image_to_data(warp_inv_thresh, config=custom_config, output_type=pytesseract.Output.DICT)       
            coords = reader.readtext(border_im)
            # final_test = pytesseract.image_to_data(warp_roi, config=custom_config, output_type=pytesseract.Output.DICT)

            # if coords:
            #     draw_eocr_boxes([((coords[0][1]), coords[0][0])], border_im)

            if coords:
                word_list.append([coords[0][1].strip(), np.divide(points, 2), "slanted"])
    return word_list

def run_filter(img, filt):
    if filt == "grayscale":
        filt_img = grayscale(img)
    elif filt == "binthres":
        filt_img = binthres(img)
    elif filt == "otsu":
        filt_img = otsu(img)
    else:
        filt_img = img
    return filt_img

def get_coordinates(img_path, filt, hths, wths, slant):    
    im = cv2.imread(img_path)
    im = rescale(im, 2, 2)

    height, width, channels = im.shape
    print(height, width)
    eocr_pred_set = easyocr_text_pred(im, hths, wths)
    coords_eocr = recons_eocr_coords(eocr_pred_set, width, height)
    # draw_ocr_boxes(coords_eocr, im)
    filt_img = run_filter(im, filt)
    final_coords = check_angle(coords_eocr, filt_img, slant, height, width)

    return final_coords