# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 15:43:10 2020

@author: ok
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2d
from skimage.util import random_noise 

img = cv2.imread('SunnyLake.bmp')
cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()

#%% Q2:

gray_img = np.mean(img,axis=2).astype(np.uint8)
cv2.imwrite('gray.jpg',gray_img)


#%% Q3:

def histogram(img,bins=100):
    temp = 255
    hist = []
    while temp>=(255//bins):
        temp =temp-(255//bins)
        ret = np.where(img>temp,1,0)
        img = img - (ret*img*2)
        hist.append([np.sum(ret),temp])
    hist.append([np.sum(np.where(img>=0,1,0)),temp])
    hist = np.flip(hist)
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(hist[:,0],hist[:,1],width = 5)
    plt.show()
        
    return hist
        
# hist2,b = np.histogram(gray_img,128)
hist = histogram(gray_img)

#%% Q4, Q5: 
thres_img = np.where(gray_img>60,255,0).astype(np.uint8)
cv2.imwrite('thres_img.jpg',thres_img)
cv2.imshow('img',thres_img)
cv2.waitKey()
cv2.destroyAllWindows()
# plt.hist(gray_img,bins=100)



#%% Q6: sigma * N(0,1)+ mu


def random_noise2(shape = [300,400],std=1):
    def noise(shape = shape,std=std):
        return std*np.random.randn(shape[0],shape[1])
    
    return np.rollaxis(np.asarray([noise(),noise(),noise()]),0,3)
#%% Q7: 
I_1 =img + random_noise2(std=1)
I_5 = img + random_noise2(std=5)
I_10 = img + random_noise2(std=10)
I_20 = img + random_noise2(std=20)
cv2.imwrite('I_1rgb.jpg',I_1)
cv2.imwrite('I_5rgb.jpg',I_5)
cv2.imwrite('I_10rgb.jpg',I_10)
cv2.imwrite('I_20rgb.jpg',I_20)

I_1 = np.mean(img + random_noise2(std=1),axis = 2)
I_5 = np.mean(img + random_noise2(std=5),axis = 2)
I_10 = np.mean(img + random_noise2(std=10),axis = 2)
I_20 = np.mean(img + random_noise2(std=20),axis = 2)
cv2.imwrite('I_1.jpg',I_1)
cv2.imwrite('I_5.jpg',I_5)
cv2.imwrite('I_10.jpg',I_10)
cv2.imwrite('I_20.jpg',I_20)


#%% Q8: Filtering with Low pass filter. Additive gaussian noise includes all the frequencies in spectrum.
# Most of the information in images  are in lower frequencies so we can remove the higher frequencies with low pass filter.
# We might lose some edge information though.
def f(img):
    Gaussian = np.asarray([[1, 2, 1],[2, 4, 2], [1, 2, 1]])/16
    lowpass3 = np.ones([3,3])/9
    lowpass5 = np.ones([5,5])/25
    return conv2d(img,Gaussian,'same'),conv2d(img,lowpass3,'same','wrap'),conv2d(img,lowpass5,'same','wrap')

I_1G,I_1L3,I_1L5 = f(I_1)
cv2.imwrite('I_1G.jpg',I_1G)
cv2.imwrite('I_1L3.jpg',I_1L3)
cv2.imwrite('I_1L5.jpg',I_1L5)


I_5G,I_5L3,I_5L5 = f(I_5)

cv2.imwrite('I_5G.jpg',I_5G)
cv2.imwrite('I_5L3.jpg',I_5L3)
cv2.imwrite('I_5L5.jpg',I_5L5)

I_10G,I_10L3,I_10L5 = f(I_10)
cv2.imwrite('I_10G.jpg',I_10G)
cv2.imwrite('I_10L3.jpg',I_10L3)
cv2.imwrite('I_10L5.jpg',I_10L5)

I_20G,I_20L3,I_20L5 = f(I_20)
cv2.imwrite('I_20G.jpg',I_20G)
cv2.imwrite('I_20L3.jpg',I_20L3)
cv2.imwrite('I_20L5.jpg',I_20L5)


#%% Filtering High pass. We can detect edges from images with high pass filtering.
## High boost filter: It is often desirable to emphasize high frequency components
# representing the image details (such as sharpening) without eliminating 
# low frequency components. High boost filter can be used in this situation.


def fh(img,A=2):
    h1 = np.asarray([[-1, -1, -1],[-1, 8, -1], [-1, -1, -1]])/9
    h2 = np.asarray([[0.17, 0.67, 0.17],[0.67, -3.33, 0.67], [0.17, 0.67, 0.17]])
    def highboost(A=A):
        f = np.ones([3,3])*-1
        f[1,1] =9*A-1
        return f
    return conv2d(img,highboost(),'same'),conv2d(img,h1,'same','wrap'),conv2d(img,h2,'same','wrap')


I_1hb,I_1h1,I_1h2 = fh(gray_img,1.1)
cv2.imwrite('I_1hb.jpg',I_1hb)
cv2.imwrite('I_1h1.jpg',I_1h1)
cv2.imwrite('I_1h2.jpg',I_1h2)

I_5hb,I_5h1,I_5h2 = fh(I_5)
cv2.imwrite('I_5hb.jpg',I_5hb)
cv2.imwrite('I_5h1.jpg',I_5h1)
cv2.imwrite('I_5h2.jpg',I_5h2)

I_10hb,I_10h1,I_10h2 = fh(I_10)
cv2.imwrite('I_10h2.jpg',I_10hb)
cv2.imwrite('I_10h1.jpg',I_10h1)
cv2.imwrite('I_10h2.jpg',I_10h2)

I_20hb,I_20h1,I_20h2 = fh(I_20)
cv2.imwrite('I_20hb.jpg',I_20hb)
cv2.imwrite('I_20h1.jpg',I_20h1)
cv2.imwrite('I_20h2.jpg',I_20h2)

cv2.imshow('img',I_1hb.astype(np.uint8))
cv2.waitKey()
cv2.destroyAllWindows()

#%% Q10: The noise is Salt and Pepper noise. We can eliminate it with median
#filtering which is a nonlinear filter.
img_sp = random_noise(img, mode='s&p',amount=0.01)*255
cv2.imwrite('img_sp.jpg',img_sp)
median = cv2.medianBlur(img_sp.astype(np.uint8),3)
cv2.imwrite('median.jpg',median)



