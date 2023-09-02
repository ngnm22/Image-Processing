import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage.exposure import is_low_contrast
from utils2 import *
import os
import math

def Butterworth_High_Pass(width, height, d, n):
    hp_filter = np.zeros((height, width,3), np.float32)
    centre = (width / 2, height / 2)

    for i in range(0, hp_filter.shape[1]):  # image width
        for j in range(0, hp_filter.shape[0]):  # image height
            radius = max(1, math.sqrt(math.pow((i - centre[0]), 2.0) + math.pow((j - centre[1]), 2.0)))
            hp_filter[j, i] = 1 / (1 + math.pow((d / radius), (2 * n)))
    return hp_filter
# create a butterworth low pass filter

def Butterworth_Low_Pass(width, height, d, n):
    lp_filter = np.zeros((height, width,3), np.float32)
    centre = (width / 2, height / 2)

    for i in range(0, lp_filter.shape[1]):  # image width
        for j in range(0, lp_filter.shape[0]):  # image height
            radius = max(1, math.sqrt(math.pow((i - centre[0]), 2.0) + math.pow((j - centre[1]), 2.0)))
            lp_filter[j, i] = 1 / (1 + math.pow((radius / d), (2 * n)))
    return lp_filter
def remove_noise(image):                      
        #median = cv2.GaussianBlur(image, (1,1), 0)
        #median = cv2.bilateralFilter(median,7,10,10)
        median = cv2.medianBlur(image, 3)# Remove salt and pepper noise
        #kernel = np.array([[-1,-1,-1],[-1,12,-1],[-1,-1,-1]])
        #median = cv2.filter2D(median,-1,kernel)
        return median

def dewarp(image):# dewrap the image
        pts1 = np.float32([[20,20], [945,7],[965,375], [16,385]])                 
        width =1024                                 
        height=394									
        pts2 = np.float32([[0, 0], [width, 0],[width,height], [0, height]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)                    
        result = cv2.warpPerspective(image, matrix, (width,height))         
        return result
def contrast_level(image,alpha,beta):                                             
	clahe = cv2.createCLAHE(clipLimit=alpha, tileGridSize=beta)      
	equalized = clahe.apply(image)
	return equalized
def change_brightness(image,beta):                                     
	image=cv2.convertScaleAbs(image, beta=beta)
	return image
def gamma_correction(src, gamma):
	src = (((src.astype("float64")/255)**gamma) * 255).astype("uint8")
	return src

def contrast_correction(src, alpha, beta):
	src = np.clip(src.astype("float64")*alpha + beta, 0, 255)
	src = (src).astype("uint8")
	return src

def brightness_correction(src, brightness):
	src = np.clip(src.astype("float64") + brightness, 0, 255)
	src = (src).astype("uint8")
	return src
def passing(image):
        #github code to do high pass and low pass
        highpass = Butterworth_High_Pass(1024,394,387,3)
        lowpass = Butterworth_Low_Pass(1024,394,421,2)
        #fourier transformation
        transformation = np.fft.fft2(image)
        fourier = np.fft.fftshift(transformation)
        comb_h = highpass * fourier
        comb_l = lowpass * fourier
        back_ishift_high = np.fft.ifftshift(comb_h)
        back_ishift_low = np.fft.ifftshift(comb_l)
        back_ishift_high = np.fft.ifft2(back_ishift_high)
        back_ishift_low = np.fft.ifft2(back_ishift_low)
        realhigh = np.real(back_ishift_high)
        reallow = np.real(back_ishift_low)
        high = np.uint8(realhigh)
        low = np.uint8( reallow)
        image = high + low
        return image
def contrast(image):
        image = contrast_correction(image, 0.51,10)
        image = gamma_correction(image, 0.91)
        return image
#noiseremoval
def denoise(image):
        image = remove_noise(image)
        return image
#brightening
def brightness(image):
        image = change_brightness(image,6)
        image = brightness_correction(image, -1)
        return image
#sharpening
def sharp(image):
        image = passing(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  
        image=contrast_level(image,0.6,(1, 1)) 
        canny = cv2.Canny(image,500,50)
        image = image - canny
        return image
#all functions together
def process(image):
        image= cv2.imread(image)
        # Load Image
        
        result = dewarp(image)                    


        contrasting = contrast(result)
 
        noise = denoise(contrasting)
        
        bright = brightness(noise)

        
        image = sharp(bright)
        return image
corrupted = 'ngnm/l2-ip-assignment/l2-ip-images/test/corrupted'

results = 'ngnm22/l2-ip-assignment/l2-ip-images/test/results'
i = 0
for image in os.listdir(corrupted):
        path = os.path.join(corrupted, image)
        if i < 101:
                new = process(path)
                join = os.path.join(corrupted,image)
                cv2.imwrite(join, new)
        if i>=101:
                new = process(path)
                new_path = os.path.join(corrupted,image)
                cv2.imwrite(join, new)
        i += 1
