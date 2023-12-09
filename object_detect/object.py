import cv2
import numpy as np
from skimage.io import imshow, imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import diplib as dip


def grayscale(img_path):
    sample = imread(img_path)
    sample_g = rgb2gray(sample)
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].imshow(sample)
    ax[1].imshow(sample_g,cmap='gray')
    ax[0].set_title('Colored Image',fontsize=15)
    ax[1].set_title('Grayscale Image',fontsize=15)
    plt.show()
    return sample_g

def grayscale_cv(img_path):
    image = cv2.imread(img_path) 
    print(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    return gray

def binarize(sample_g):
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    sample_b = sample_g > 0.6
    ax[0].set_title('Grayscale Image',fontsize=20)
    ax[0].imshow(sample_g,cmap='gray')
    ax[1].plot(sample_g[600])
    ax[1].set_ylabel('Pixel Value')
    ax[1].set_xlabel('Width of Picture')
    ax[1].set_title('Plot of 1 Line',fontsize=15)
    ax[2].set_title('Binarized Image',fontsize=15)
    ax[2].imshow(sample_b,cmap='gray')
    plt.show()

def sobel_edges(img_path):
    image = cv2.imread(img_path) 
    src = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    window_name = ('Sobel Demo - Simple Edge Detector')
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S    
    
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv2.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    cv2.imwrite('./output/sobel-edges.jpg', grad)

def canny_edges(img_path):
    image = cv2.imread(img_path) 
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(img, 30, 120)
    cv2.imwrite('./output/canny-edges.jpg', canny)

def tophat_edges(img_path):
    image = cv2.imread(img_path) 
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    a = dip.Image(img)
    # a.Show()
    lines = dip.Tophat(a, 30, polarity='black')

    dip.SetBorder(lines, [0], [2])

    lines = dip.PathOpening(lines, length=100, polarity='opening', mode={'robust'})
    lines = dip.Threshold(lines, method='otsu')[0]
    dip.viewer.ShowModal(lines)

def main():
    img_path = './Images/test.jpg'
    gs_img = grayscale_cv(img_path)
    # binarize(gs_img)
    # sobel_edges(img_path)
    # canny_edges(img_path)
    # tophat_edges(img_path)
    return

if __name__ == "__main__":
    main()