#!/usr/bin/env python3
# This is template for your program, for you to expand with all the correct
"""
CE866: Computer Vision Assignment

TASK: 

The task of this assignment is to find the location and bearing angle of a 
pointer placed on a map.
    
Given Information: 
    
    1) It is given that the map will be placed on a blue background.
    
    2) The camera capturing the image is placed directly on top.
    
    3) The lighting illuminates the table fairly uniformly.
    
    4) The map has a green arrow showing the direction of north.
    
    5) The pointer is red isosceles triangle.
    
    6) The tip of the red pointer determines the viewpoint.
    
    7) The direction in which it is pointing determines the view direction.
    
    
    
Approach: 
    
    The program follows the given steps:
        
        1) Filtering out the blue background and extracting the map
        
        2) Filtering out the green arrow region and determining its location.
        
        3) If the green arrow is not on the top-right corner the image will
           be rotated.
        
        4) Filtering out the red triangle.
        
        5) The tip will be the point opposite to the smallest side.
        
        6) Finding the pixel location of the tip.
        
        7) Finding the pixel location of the centroid of the triangle.
        
        8) Finding the bearing angle using the centroid and tip point of
           the pointer.
        
        9) Scaling the tip point co-ordinates to the range 0-1 in x axis
           and y axis.
           
           
Usage:
    
  1)  python3 mapreader.py <image location>
    
    Here the command line argument is the location of the image
    
             eg: python3 mapreader.py develop/develop-001.jpg
             
             This will generate the output as:
             
                       The filename to work on is develop/develop-001.jpg.
                       POSITION 0.459 0.591
                       BEARING 262.9
                       
                       
   
 
        

Author:
    
    Registration number : 2100641
    Department: School of Computer Science and Electronics
    Course: MSc Artificial Intelligence
        
    I hereby certify that this program is entirely my own work .


    
"""
#-------------------------------------------------------------------------------
# Boilerplate.
#-------------------------------------------------------------------------------
import cv2, sys
import numpy as np
import math

#-------------------------------------------------------------------------------
# Routines.
#-------------------------------------------------------------------------------

def img_preprocessing(im,color):

    """
    Checks for the contour points enclosing the colour regions
    The following function takes in 2 arguments im,color


    Args:
       im (ndarray)    : image that needs to be processed
       color (string)  : color region that we need to segment
       Here the colors can only be "red","blue","green"
       
    Returns:
       (list) of contour points 
       
    """
    
    
#   the image needs to be converted to hsv format
    img = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    
    
#   The red color is distributed at both the ends of the hsv spectrum 
#   So 2 seperate masks should be created for the red 
#   These masks are then added up to create our final mask.
    
    
    if color == "red":
        
#       The following 7 lines of code was taken from:
#       ("https://cvexplained.wordpress.com/2020/04/28/color-detection-hsv/")
        
        lower1 = np.array([0, 100, 20], dtype="uint8")
        upper1 = np.array([10, 255, 255], dtype="uint8")
        
        lower2 = np.array([160,100,20], dtype="uint8")
        upper2 = np.array([179,255,255], dtype="uint8")
        
        lower_mask = cv2.inRange(img, lower1, upper1) 
        upper_mask = cv2.inRange(img, lower2, upper2)
 
        mask = lower_mask + upper_mask 
        
#       contours are calculated using the mask, chain_approx_simple is used as 
#       we are only interested in the corner points

        contours,_= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        return contours
    
    
#   the folowing lines of code for blue and green was adapted from
#   ("https://youtu.be/Q0IPYlIK-4A")
    if color == "blue":
        lower = np.array([94, 80, 2], dtype="uint8")
        upper = np.array([126,255,255], dtype="uint8")
        
    if color == "green":
        lower = np.array([50 ,40, 40], dtype="uint8")
        upper = np.array([70 ,255, 255], dtype="uint8")
        
    mask = cv2.inRange(img, lower,upper)
    
    contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    return contours
    



def map_isolation(im):

    """
	This function returns a new image of the map by filtering out the background.
	
	Args:
	  im (ndarray) : im is the image from which the map should 
	                 be extracted.
	                 
	Returns:
	  (ndarray) : new image of the map

    """
    
#   The contour points of the blue masked areas are obtained using 
#   the img_preprocessing function     
    contours = img_preprocessing(im,"blue")

    
    for i,c in enumerate(contours):
        
        
#       The first group of contour points are ignored as this will
#       be the entire window.
       
        if i>0:
            
            
            approx = cv2.approxPolyDP(c, 15, True)
            
#       we are looking for a rectangular shape with 4 contour points.

#       Since the blue color is uniformily distribuited the first 
#       polygon with 4 points will be our map.

            if len(approx) == 4:
                location = approx
                break

    pt_A, pt_B, pt_C, pt_D = (location[3][0], location[2][0], 
                               location[1][0], location[0][0])

#   the following lines of codes have been taken from:
#   ("https://theailearner.com/tag/cv2-getperspectivetransform/")

   
#   the length of line connecting each points are calculated using:
#   equation  √[( x 2 − x 1 )**2 + (y 2 − y 1 )**2].


    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) 
                       + ((pt_A[1] - pt_D[1]) ** 2))  

    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) 
                       + ((pt_B[1] - pt_C[1]) ** 2))

    

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) 
                        + ((pt_A[1] - pt_B[1]) ** 2))
    
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2)
                        + ((pt_C[1] - pt_D[1]) ** 2))
    
    
    
    maxWidth, maxHeight = (max(round(width_AD), round(width_BC)), 
                             max(round(height_AB), round(height_CD)))
    
#   maxHeight will be the height of our output image and maxWidth will
#   be the width of our image.    



    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],
                            [0, maxHeight - 1],
                            [maxWidth - 1, maxHeight - 1],
                            [maxWidth - 1, 0]])

    M = cv2.getPerspectiveTransform(input_pts,output_pts)
    out = cv2.warpPerspective(im,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)
    
#   The new image of the map is obtained by using cv2.wrapPerspectiveTransform

    return out
    


def green_arrow_detection(im):

    """
    This function detects the location of the green arrow in the image
    
    Args:
        im (ndarray) : the extracted image of the map
        
    Returns:
        threshY : y pixel of the midpoint of the image(im)
        threshX : x pixel of the midpoint of the image(im)
        imageY  : y pixel of the centroid of the green arrow region.
        imageX  : x pixel of the centroid of the green arrow region.
    """

    
    
    
    
#   The contours of the green regions in the image are obtained 
#   using the function img_preprocessing.
    
    contours = img_preprocessing(im,"green")

    
#   since the arrow head is the only green area in the image, 
#   the contour with the maximum area will be our green arrow.

    area = 0
    for c in contours:
        a = cv2.contourArea(c) 
        
#       the function gives the area of the region enclosed by 
#       the contour points.
        
        if a > area:
            area = a
            cntr = c
            
   
    
#   the cv2.moments function in cv2 gives a dictionary of image moments.

#   the following 2 lines of code was taken from
#   (https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html)

    M = cv2.moments(cntr)
    imageX, imageY = M['m10']/M['m00'], M['m01']/M['m00']
#   imageX, imageY are the co-ordinates of the centroid of the contour.      

    maxY,maxX,_= im.shape
    threshY,threshX = maxY/2,maxX/2
#   threshY,threshX are the co-ordinates of the center point of the image.       


    return threshY,threshX,imageY,imageX
    



def image_orientation(im):

    """
    It is known that the correct postion of the green arrow 
    should be at the north east corner of the map.
    This function compares the loctaion of the green arrow 
    on the map if it is not on the north-east part, it image 
    will be rotated by 180 degrees.
    
    Args:
      im (ndarray) : the extracted image of the map
      
    Returns:
      (ndarray) of the image in correct orientation
    
    """
    
    flag = True
    while (flag):
    
        threshY,threshX,imageY,imageX = green_arrow_detection(im)
        
        """
         The y position of the green arrow should be less than 
         the threshold y co-ordinate.This is because the y 
         co-ordinates numbered from top to bottom in openCV, so
         the top-most value will be the lowest.

         Also the X co-ordinate of the green arrow should be 
         greater than the threshould X co-ordinate.
        """
         
        if((imageY < threshY) and (imageX > threshX)):
             flag = False
             break
             
#           When the condition is satisfied, we break out of the loop. 
            
        im = cv2.rotate(im, cv2.ROTATE_180)
        
        """
        If the green arrow is not at the top-right corner, 
        the image is rotated by 180 degrees.
        """    
    
    return im






    
"""
From the input images provided we can observe that the red pointer
is an isosceles triangle with the base being shorter than the other sides.

So our base will be the line segment with the shortest legth and the 
region of interest will be the point opposite to the base.

The centroid of the red-pointer region is also calculated in the 
function. This will be usefull when calculating the bearing angle.

The following function finds the tip and centroid of the pointer.

"""



def red_arrow_detection(im):

    """
    This function isolates the red pointer region in the map.
    and finds the tip of the pointer
    
    Args:
       im (ndarray) : image after orienting correctly
       
    returns:
       roi (list)     : x and y pixel values of the tip of the pointer
       centroid(list) : x and y pixel values of the centroid of the pointer
    """
    
#   contours for the red region are found using img_preprocessing function.
    contours = img_preprocessing(im ,"red")

 
    for c in contours:
        
        approx = cv2.approxPolyDP(c, 0.07 * cv2.arcLength(c, True), True)
                
#       We know that the triangular area will have three points, so the 
#       polygon with 3 points are used to find the pointer region. 
        if len(approx) == 3:
            location = approx
            cntr = c
            break


    pt_A, pt_B, pt_C = location[0][0], location[1][0], location[2][0]
    
#   the following 3 lines of codes have been taken from:   
#   ("https://theailearner.com/tag/cv2-getperspectivetransform/")


    length_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    length_AC = np.sqrt(((pt_A[0] - pt_C[0]) ** 2) + ((pt_A[1] - pt_C[1]) ** 2))
    length_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    
#   After getting the length of all the sides we have to find the 
#   line with the least length this line will be the base.

    min_len = min([length_AB,length_AC,length_BC])

#   The point opposite to the base will be the tip of the pointer (roi) 

    if min_len == length_AB:
        
        roi = pt_C

    elif min_len == length_AC:
        
        roi = pt_B

    else: 
        roi = pt_A

#   The centroid is calculated using moments
#   The following two lines of code was taken from:
#   (https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html)
    
    M = cv2.moments(cntr)
    cx,cy = M['m10']/M['m00'], M['m01']/M['m00']
    
    centroid = [cx,cy]

    return roi,centroid

"""
The roi and centroid are in pixel values where the y axis is numbered
from top to bottom. And x axis from right to left

The following fuction scales down the x,y co-ordinates of the pointer 
to the range 0 to 1 in both x axis and y axis.

The fuction also calculates the bearing angle of the pointer.

"""


def bearing(center,point,maxY,maxX): 
    
    """
    This function calculates the bearing angle on the pointer
    
    Args:
      Center : The x,y co-ordinates of the centroid of the pointer.
      point  : The x,y co-ordinates of the tip point of the pointer.
      maxY   : The total height of the image.
      maxX   : The total width of the image.
      
    Returns:
      scaled_x : The x co-ordinate of the pointer scaled to the range 0-1.
      scaled_y : The y co-ordinate of the pointer scaled to the range 0-1.
      angle    : The bearing angle of the pointer.
      
      
    """
    
#    We know that in opencv the pixes values of y axis are marked from
#    top to bottom.  
    
#    Before calculating the bearing angle we have to change 
#    y-cordinates from bottom to top.
  
  
  
    center[1] = maxY - center[1]
    point[1] = maxY - point[1]

    
    """    
    Bearing angle is the angle calculated from north in clock-wise direction.
 
    To calculate the bearing angle, we take the angle of the line from the
    centroid of the triangle to the tip of the triangle.


    The equation is θ = atan2(b1−a1,b2−a2)∈(−π,π]
    where, (b1,b2) is the x,y co-ordinates of the tip of the pointer.
           (a1,a2) is the x,y co-ordinates of the tip of the pointer.

    The math for the following 4 lines of code was taken from
    (https://math.stackexchange.com/questions/1596513/find-the-bearing-angle-between-two-points-in-a-2d-space)
    """
    
    
    angle = math.atan2(point[0]-center[0],point[1]-center[1])
    
    if angle < 0:
#       we have to make sure that the angle will be in the range 0-360        
        angle = angle + (2*math.pi) 
     
     
        
#   we have to convert the angle obtained in radians to degrees.    
    angle = (angle * 180)/ math.pi
    
       
#   Finally the pointer location should be scaled to the range 0 to 1
#   This can be done by dividing the x co-ordinate with the total width and
#   dividing the y co-ordinate with the total height
    
    scaled_x = point[0]/maxX
    scaled_y = point[1]/maxY
    
    
    return scaled_x,scaled_y,angle
    
    
#-------------------------------------------------------------------------------
# Main program.
#-------------------------------------------------------------------------------



# Ensure we were invoked with a single argument.

if len (sys.argv) != 2:
    print ("Usage: %s <image-file>" % sys.argv[0], file=sys.stderr)
    exit (1)
    
    
    
# Process each file on the command line in turn.
for fn in sys.argv[1:]:
     
     im = cv2.imread (fn)

     # The map can be isolated from the image using the map isolation function
     map_img = map_isolation(im)

     maxY,maxX,_ = map_img.shape

     # The image can be oriented using image_orientation function
     map_img = image_orientation(map_img)


     # The red pointer tip and the centroid of the red arrow can be obtaied 
     # using   red_arrow_detection
     
     roi,centroid = red_arrow_detection(map_img)
     """
     the obtained roi (location of the tip of the pointer) should be scaled down 
     to the range 0-1
     
     the centroid of the triangle is used to calculate the bearing angle
     both these can be done using bearing function.
     """
     xpos, ypos, hdg = bearing(centroid,roi,maxY,maxX)
     
     print ("The filename to work on is %s." % sys.argv[1])
     # Output the position and bearing in the form required by the test harness.
     print ("POSITION %.3f %.3f" % (xpos, ypos))
     print ("BEARING %.1f" % hdg)
     


#-------------------------------------------------------------------------------

