import cv2

# grayscale filter
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("./output/grayscale_test.jpg", img)
    return img

# binary threshold filter
def binthres(img):
    img = grayscale(img)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return img[1]

def otsu(img):
    img = grayscale(img)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return img[1]

def find_line(p1, p2):
    rise = p2[1] - p1[1] 
    run = p2[0] - p1[0]
    slope = rise/run 
    intercept = p1[1] - (slope * p1[0])
    return slope, intercept

def perpendicular(point, slope):
    slope = -slope
    intercept = point[1] - (slope * point[0])
    return slope, intercept
    

def intersection(l1s, l1i, l2s, l2i):
    x = (l2i - l1i) / (l1s - l2s)
    y = l1s * x + l1i
    return [x, y]


def draw_eocr_boxes(coordinates, img):
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
        
    cv2.imwrite("./output/boundbox_test_ocr.jpg", img)
    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def add_white_bg(img, border):
    return cv2.copyMakeBorder(img, border, border, 
                              border, border, 
                              cv2.BORDER_CONSTANT, 
                              value=[255, 255, 255])
    # return cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_REPLICATE)
