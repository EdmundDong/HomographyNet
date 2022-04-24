# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 22:06:44 2019

@author: saitej
"""

# '''
# !pip install numpy
# !pip install matplotlib
# !pip install opencv-python
# !pip install opencv-contrib-python
# #!pip install opencv-python-headless
# #!pip install opencv-contrib-python-headless
# '''


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from PIL import Image, ImageOps

def slightTransformByOrigin(srcImg, shift, transform_limit, image_name):
    width = height = 300
    srcImg = ImageOps.grayscale(srcImg)
    srcImg = ImageOps.pad(srcImg, (width, height))
    srcImg = ImageOps.expand(srcImg, 100)
    srcImg = srcImg.resize((width, height))
    srcImg = np.asarray(srcImg)
    #cv.imwrite(f"output/{image_name}_src.jpg", srcImg)

    proj2dto3d = np.array([[1,0,-width/2],
                        [0,1,-height/2],
                        [0,0,1],
                        [0,0,1]],np.float32)

    rx   = np.array([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0],
                    [0,0,0,1]],np.float32)  

    trans= np.array([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,199],   #350 to shrink image so entire thing fits
                    [0,0,0,1]],np.float32)

    proj3dto2d = np.array([ [200,0,width/2,0],
                            [0,200,height/2,0],
                            [0,0,1,0] ],np.float32)


    x = 0.0 
    four_points = np.float32([[(75, 75), (75, 375), (375, 375), (375, 75)]])
    with open("matrix_info.csv", 'w') as large_file:
        for i in range(0,transform_limit):
            
            ax = float(x * (math.pi / 180.0)) 

            rx[1,1] = math.cos(ax) 
            rx[1,2] = -math.sin(ax) 
            rx[2,1] = math.sin(ax) 
            rx[2,2] = math.cos(ax) 
            
            r = rx 
            transform_matrix = proj3dto2d.dot(trans.dot(r.dot(proj2dto3d)))
            dst = cv.warpPerspective(srcImg, transform_matrix,(width,height),None,cv.INTER_LINEAR
                                ,cv.BORDER_CONSTANT,(0,0,0))
            cv.imwrite(f"output/{image_name}-"+str(i)+"-f1.jpg", srcImg)
            cv.imwrite(f"output/{image_name}-"+str(i)+"-f2.jpg", dst)
            
            perturbed_four_points = cv.perspectiveTransform(four_points, np.linalg.inv(transform_matrix))
            # H = cv.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
            # warped_image = cv.warpPerspective(srcImg, np.linalg.inv(H), (width,height))
            # cv.imwrite("transformImage"+str(i)+"Test.jpg", warped_image)
            
            large_file.write(f'{image} {i}\n{four_points}\n{perturbed_four_points}\n')
            with open(f"output/{image_name}-"+str(i)+"-m.csv", 'w') as indiv_file:
                printstring = f'{four_points}\n{perturbed_four_points}\n'.replace('[', '').replace('.]', '').replace(']', '').replace('. ', ' ').split(' ')
                while("" in printstring) :
                    printstring.remove("")
                #print(printstring)
                indiv_file.write(' '.join(printstring).replace('\n ', '\n'))
                indiv_file.write(f'{transform_matrix}\n{np.array2string(transform_matrix, suppress_small=True)}\n')
            
            x -= shift  

if __name__=="__main__":
    #os.system("pip install numpy")
    #os.system("pip install matplotlib")
    #os.system("pip install opencv-contrib-python")
    files = [f for f in os.listdir('input') if f.lower().endswith(('.jpg', '.jpeg'))]
    for image in files:
        slightTransformByOrigin(Image.open(f'input/{image}'), 2, 30, image.split('.')[0])

