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