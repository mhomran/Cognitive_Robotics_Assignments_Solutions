"""
@file   : 4.1.py
@brief  : a solution for the problem 4.1 in Assignment 1
@author : Mohamed Hassanin Mohamed
@data   : 25/03/2022
"""

import cv2 as cv
import numpy as np

BGR_RED_COLOR = (0, 0, 255)
BGR_BLUE_COLOR = (255, 0, 0)
BGR_GREEN_COLOR = (0, 255, 0)

GRAYSCALE_BLACK_COLOR = 0

class RayCaster:
  def __init__(self, map, angle_range=250, angle_accuracy=2, 
  length_range=12, pixel_size=4):
    """
    Description: constructor for the ray caster.

    Input:
      - map: the filename of the map where the ray will be casted.
      - angle_range: the angle range to be scanned in degrees 
      from (1 to 360). Half of this range is for the right
      scan and the second half is for the left part.
      - angle_accuracy: An integer representing the step to
      increment the angle with in degrees.
      - length_range: the maximum length for the ray to be scanned
      in meters.
      - pixel_size: the pixel size in real world in centimeters.
    """
    self.angle_range = angle_range
    self.angle_accuracy = angle_accuracy
    self.length_range = length_range
    self.pixel_size = pixel_size

    self.map_bgr = cv.imread(map)
    self.map_gray = cv.cvtColor(self.map_bgr, cv.COLOR_BGR2GRAY)
    _, self.map_gray = cv.threshold(self.map_gray, 128, 255, cv.THRESH_OTSU)

  def _is_collided(self, p):
    """
    Description: check if a certain point is collided with
    an obstacle in the map.

    Input:
      - p: the point to be checked (x, y).

    Output: 
      - is_collided: True if collided, False otherwise
    """
    is_collided = False

    if self.map_gray[p[1]][p[0]] == GRAYSCALE_BLACK_COLOR:
      is_collided = True
    
    return is_collided

  def cast(self, x, y, theta, show_rays=False, show_collided=False):
    """
    Description: Cast a ray.
      
    Input:
      - x: the x coordinate on the map.
      - y: the y coordinate on the map.
      - theta: the heading direction on the map in degrees (1-360).
      0 degree makes the robot look horizontally right. The angle of the
      robot is incremented in the clock wise direction (i.e. 90 degree will
      make the robot look vertically down).
      - show_rays: if True, show the image with the rays shown.
      - show_collided: if True, show the image with the collided points
      shown. 
    Output:
      - measurements
    """
    if show_rays or show_collided:
      img = self.map_bgr.copy()

    measurements = []
    
    start_angle = theta - (self.angle_range // 2)
    end_angle = theta + (self.angle_range // 2)
    start_len = 1
    # from metric unit to the number of pixels 
    end_len = (self.length_range * 100) // self.pixel_size 
    
    for angle in range(start_angle, end_angle + 1, self.angle_accuracy):
      for len in range(start_len, end_len + 1):
        tobe_checked_x = x + int(len * np.cos(np.radians(angle)))
        if not (0 <= tobe_checked_x < self.map_gray.shape[1]):
          break 
        
        tobe_checked_y = y + int(len * np.sin(np.radians(angle)))
        if not (0 <= tobe_checked_y < self.map_gray.shape[0]):
          break 
        
        tobe_checked = (tobe_checked_x, tobe_checked_y)
        
        if show_rays:
          cv.circle(img, tobe_checked, 2, BGR_RED_COLOR, -1)
        
        if self._is_collided(tobe_checked): 
          measurements.append(tobe_checked)
          
          if show_collided:
            cv.circle(img, tobe_checked, 2, BGR_BLUE_COLOR, -1)
          
          # no need to progress with that ray since it's collided
          break 
        
    if show_rays or show_collided:
      cv.circle(img, (x, y), 2, BGR_GREEN_COLOR, 5)
      cv.imshow("Ray Casting", img)
      cv.waitKey(0)

    return measurements    

if __name__ == '__main__':
  ray_caster = RayCaster("Assignment_04_Grid_Map.png")

  measurements = ray_caster.cast(x=400, y=180, theta=180,
  show_rays=True, show_collided=True)