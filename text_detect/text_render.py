from PIL import Image, ImageDraw, ImageFont
import numpy as np
import easyocr_oob, method1, method2
import math
import cv2
import time
import sys
# Setting font 
font_path = 'C:\Windows\Fonts\himalaya.ttf'
# Color: black
text_color = (0, 0, 0)

# Finding the size of the image 
def get_image_size(image_path):
    img = cv2.imread(image_path)
    height, width, channels = img.shape
    return width, height

# Finding the best size of font, so that the box is filled up
# Not quite sure if this is working properly
def get_best_font_size(word, box_size, font_path, max_font_size=100):
    font_size = max_font_size
    font = ImageFont.truetype(font_path, font_size)
    text_bbox = font.getbbox(word)

    while text_bbox[2] - text_bbox[0] > box_size[0] or text_bbox[3] - text_bbox[1] > box_size[1]:
        font_size -= 1
        if font_size <= 0:
            break
        font = ImageFont.truetype(font_path, font_size)
        text_bbox = font.getbbox(word)

    return font

# Finding the middle of the rectangle
def get_middle(coords):
    center_x = (coords[0][0] + coords[1][0] + coords[2][0] +  coords[3][0]) / 4
    center_y = (coords[0][1] + coords[1][1] + coords[2][1] + coords[3][1]) / 4
    return (center_x, center_y)

# Finding the angle of tilt of box using tanx = opp/adj
def calculate_tilt_angle(coords):
    if coords[0][1] > coords[1][1]:
        # counterclockwise
        opposite = abs(coords[0][1] - coords[3][1])
        adj = abs(coords[0][0] - coords[3][0])
        angle_rad = math.atan(opposite / adj)
        angle_deg = 90 - math.degrees(angle_rad)
    else:
        opposite = abs(coords[0][0] - coords[3][0])
        adj = abs(coords[0][1] - coords[3][1])
        angle_rad = math.atan(opposite / adj)
        
    # Convert radians to degrees
        angle_deg = - math.degrees(angle_rad)

    return angle_deg


def draw_canvas(width, height, word_list, img_path, save_path):
# Creating a blank canvas for us to draw the words on 
    width, height = get_image_size(img_path)
    canvas = Image.new('RGB', (int(width), int(height)), color='white')
    draw = ImageDraw.Draw(canvas)

    # This loop goes through the word_list and outputs the word onto the blank canvas
    for word, coords, slanted in word_list:
        canvas_array = np.array(canvas)
        if slanted == "slanted":
            word = str(word)
            width = math.sqrt((coords[1][0] - coords[0][0]) ** 2 + (coords[1][1] - coords[0][1]) ** 2)
            height = math.sqrt((coords[3][0] - coords[0][0]) ** 2 + (coords[3][1] - coords[0][1]) ** 2)

            top_left = tuple(map(int, coords[0]))
            top_right = [coords[0][0] + width, coords[0][1]]
            bottom_right = [coords[0][0] + width, coords[0][1] + height]
            bottom_left = [coords[0][0], coords[0][1] + height]
            box_size = (width, height)
            max_font_size = 100
            font = get_best_font_size(word, box_size, font_path, max_font_size)
            angle = calculate_tilt_angle(coords)
            # draw.text(top_left, word, font=font, fill=text_color)

            if coords[0][1] > coords[1][1]:
                # counterclockwise
                # problem here fix
                # Create a temporary canvas for the slanted text box
                temp_canvas = Image.new('RGBA', (int(width), int(height)), color=(0, 0, 0, 0))
                temp_draw = ImageDraw.Draw(temp_canvas)

                temp_draw.text((0, -5), word, font=font, fill=(0, 0, 0, 255))
                # temp_canvas.show()

                temp_canvas = temp_canvas.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))
                
                # Separate the alpha channel from the temporary canvas
                _, _, _, alpha = temp_canvas.split()
                # Paste the transformed text box onto the main canvas using the alpha channel as the mask
                canvas.paste(temp_canvas, tuple(map(int, (coords[0][0], coords[0][1] - (coords[3][1]-coords[1][1])))), mask=alpha)
            else:
                # Create a temporary canvas for the slanted text box
                temp_canvas = Image.new('RGBA', (int(width), int(height)), color=(0, 0, 0, 0))
                temp_draw = ImageDraw.Draw(temp_canvas)

                temp_draw.text((0, -5), word, font=font, fill=(0, 0, 0, 255))
                

                temp_canvas = temp_canvas.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))

                # Separate the alpha channel from the temporary canvas
                _, _, _, alpha = temp_canvas.split()

                # Paste the transformed text box onto the main canvas using the alpha channel as the mask
                canvas.paste(temp_canvas, top_left, mask=alpha)

        else:
            # Extract the word and coordinates
            word = str(word)
            coords = tuple(coords.astype(np.int32))
            x1, y1 = coords[0]
            x2, y2 = coords[2]
            position = (x1, y1)
            box_size = (x2 - x1, y2 - y1)
            max_font_size = 100
            font = get_best_font_size(word, box_size, font_path, max_font_size)
            # Draw the text on the original canvas
            draw.text(position, word, font=font, fill=text_color)

    canvas.save(save_path)
    canvas.show()

def main(method, img_path, save_path):
    if method == "easyocr":
        py_file = easyocr_oob
    # elif method == "tesseract":
    elif method == "method2":
        py_file = method2
    else: # default to method1
        py_file = method1

    start_time = time.time()
    width, height = get_image_size(img_path)

    filters = ["grayscale"]
    boundbox = [1]
    slant_angles = [10]

    # word_list, save_path = image_process.get_coordinates(filt, hbound, wbound, slant)
    word_list = py_file.get_coordinates(img_path, "grayscale", 1, 1, 5)
    draw_canvas(width, height, word_list, img_path, save_path)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
