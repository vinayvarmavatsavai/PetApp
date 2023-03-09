import streamlit as st
import cv2
from PIL import Image,ImageEnhance
import numpy as np 
import os
from skimage.metrics import structural_similarity as compare_ssim
import argparse
import imutils
import math

def show_growth_page():
    
    image_file_A = st.file_uploader("Upload Image",type=['jpg','png','jpeg'],key="1")
    image_file_B = st.file_uploader("Upload Image",type=['jpg','png','jpeg'],key="2")
    if image_file_A is not None:
    	st.text("NOW")
    	# st.write(type(our_image))
    
    our_image_A = Image.open(image_file_A)
    st.image(our_image_A)
    our_image_A=np.array(our_image_A.convert('RGB'))
    
    
    
    if image_file_B is not None:
    	st.text("THEN")
    	# st.write(type(our_image))
    our_image_B = Image.open(image_file_B)
    st.image(our_image_B)
    our_image_B=np.array(our_image_B.convert('RGB')) 
    
    
    grayA = cv2.cvtColor(our_image_A, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(our_image_B, cv2.COLOR_BGR2GRAY)
    
    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    st.write("SSIM: {}".format(score))
    
    (T,thresh) = cv2.threshold(diff, -9, 255,cv2.THRESH_BINARY_INV)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    
    img2=thresh
    font = cv2.FONT_HERSHEY_COMPLEX
    for cnt in cnts :
        area = cv2.contourArea(cnt)
    
        #print(area)
    
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
    
        # draws boundary of contours.
        cv2.drawContours(img2, [approx], 0, (0, 0, 255), 5) 
    
        # Used to flatted the array containing
        # the co-ordinates of the vertices.
        n = approx.ravel() 
        i = 0
    
        for j in n :
            if(i % 2 == 0):
                x = n[i]
                y = n[i + 1]
                hp=x
                hj=y
                hkl=[hp,hj]
                #print(hkl)
    
                # String containing the co-ordinates.
                string = str(x) + " " + str(y) 
                #print(string)
    
                #if(i == 0):
                    # text on topmost co-ordinate.
                #    cv2.putText(img2, "Arrow tip", (x, y),
                #                    font, 0.5, (255, 0, 0)) 
                #else:
                #    # text on remaining co-ordinates.
                #   cv2.putText(img2, string, (x, y), 
                #              font, 0.5, (0, 255, 0)) 
            i = i + 1
            
    # loop over the contours
    for c in cnts:
        area_ = cv2.contourArea(c)
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        #cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if (area_>=10.0 and area_<=1900):
            cv2.rectangle(image_file_B, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #print(area_)
    
            
    #cv2.imshow("kn",imageB)
    st.image(image_file_B)
    #cv2.imshow("jb",diff)
    st.image(diff)
    #cv2_imwrite("result",diff)
    #cv2.imshow("hvh",thresh)
    st.image(thresh)
