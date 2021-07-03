import cv2
import numpy as np
from PIL import Image
import dlib
import imutils
import math
from matplotlib import pyplot as plt
import argparse

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
def func1(rect):
  if len(rects) > 0:
    for rect in rects:
        x = rect.left()
        y = rect.top()
        w = rect.right() -x
        h = rect.bottom() -y
  return (x,y,w,h)
def shape_to_normal(shape):
    shape_normal = np.zeros((68,2))
    for i in range(0, 68):
      shape_normal[i] = (shape.part(i).x, shape.part(i).y)
    return shape_normal
def func3(shape_normal):
  dis = (np.sum(shape_normal[42:47,1]) - np.sum(shape_normal[36:41,1]))/(np.sum(shape_normal[42:47,0]) - np.sum(shape_normal[36:41,0])) 
  angle=math.degrees(math.atan(dis))
  return angle
def rotate_image(image, shapex):
  angle=func3(shapex)
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


#facecropping
def forehead_dist(shape_normal):

  d = (np.sum(shape_normal[42:47,1]) - np.sum(shape_normal[36:41,1]))/ 6
  return d
def face_cropping_without_forehead(image):  

 
  rects = detector(image ,1)           
  if len(rects) > 0:
    images = []
    for (i, rect) in enumerate(rects):
	
      shape = predictor(image, rect)
      shape = shape_to_normal(shape)
   
      # [i.e., (x, y, w, h)], then draw the face bounding box
      #(x1, y1, w1, h1) = func1(rect)

      d =(np.sum(shape[42:47,1]) - np.sum(shape[36:41,1]))/ 6
      top_y = int(np.sum(shape[42 : 47, 1]) / 6 - 0.6 * d)
      left_x, left_y = shape[0]
      bottom_x, bottom_y = shape[8]
      right_x, right_y = shape[16]
      cropped_image = image[int(top_y) : int(bottom_y), int(left_x) : int(right_x)]
     # print(image.shape[1])
     # print(shape.shape)
      if cropped_image.shape[0] == 0: 
        cropped_image = image[0:-1,int(left_x) : int(right_x)] 
      if cropped_image.shape[1] == 0:
        cropped_image = image[int(top_y) : int(bottom_y),  0:-1]
      images.append(cropped_image)
    if len(rects) == 1 :
      return cropped_image
    else:
      return images

  
  else:
    #print("Error : number of detected face is zero, so we just return original image")
    return image
def face_cropping_without_background(image):         # implementing face cropping without background
  #gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  # convert color image to grayscale image
  rects = detector(image ,1)             # detect faces in the grayscale image
  if len(rects) > 0:
    images = []
    for (i, rect) in enumerate(rects):
	
      shape = predictor(image, rect)
      shape = shape_to_normal(shape)
    
      # convert dlib's rectangle to a OpenCV-style bounding box
      # [i.e., (x, y, w, h)], then draw the face bounding box
      (x1, y1, w1, h1) = func1(rect)

      top_x, top_y = shape[19]
      left_x, left_y = shape[0]
      bottom_x, bottom_y = shape[8]
      right_x, right_y = shape[16]
      cropped_image = image[ min(int(top_y), abs(y1)) : max(int(bottom_y), abs(y1) + w1), min(int(left_x), abs(x1)) : max(int(right_x), abs(x1) + w1)]
      if cropped_image.shape[0] == 0: 
        cropped_image = image[:,min(left_x, abs(x1)) : max(right_x, abs(x1) + w1)] 
      if cropped_image.shape[1] == 0:
        cropped_image = image[min(top_y, abs(y1)) : max(bottom_y, abs(y1) + w1), :]
      images.append(cropped_image)
    if len(rects) == 1 :
      return cropped_image
    else:
      return images
  else:
    print("Error : number of detected face is zero, so we just return original image")
    return image
  
 #histogram equalisation
def histogram_equilization(image):
  hist,bins = np.histogram(image.flatten(),256,[0,256])
 # Calculate the cumulative distribution map
  cdf = hist.cumsum()
  cdf_normalized = cdf * hist.max()/ cdf.max()
  cdf_m = np.ma.masked_equal(cdf,0)
  cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
 # Assign a value to the masked element, where the assignment is 0
  cdf = np.ma.filled(cdf_m,0).astype('uint8')
  image= cdf[image]
  return image


