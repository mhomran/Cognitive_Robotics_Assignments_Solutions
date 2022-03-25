"""
@file   : 4.2.py
@brief  : a solution for the problem 4.2 in Assignment 1
@author : Mohamed Hassanin Mohamed
@data   : 25/03/2022
"""

import cv2 as cv
import numpy as np

BGR_RED_COLOR = (0, 0, 255)
BGR_GREEN_COLOR = (0, 255, 0)
BGR_BLUE_COLOR = (255, 0, 0)

GRAYSCALE_WHITE_COLOR = 255
GRAYSCALE_BLACK_COLOR = 0

CM_IN_METER = 100

class EndPoint:
  def __init__(self, map, angle_accuracy=20, length_range=12, pixel_size=4,
    gaussian_sigma=10) -> None:
    """
    Description: constructor for End Point model.

    Input:
      - map: the filename of the map where the ray will be casted.
      - angle_accuracy: An integer representing the step to
      increment the angle with in degrees.
      - length_range: the maximum length for the ray to be scanned
      in meters.
      - pixel_size: the pixel size in real world in centimeters.
      - gaussian_sigma: the standard deviation of the gaussians of 
      the obstacles.
    """
    self.length_range = length_range
    self.pixel_size = pixel_size
    self.angle_accuracy = angle_accuracy

    self.gaussian_sigma = gaussian_sigma

    self.map_bgr = cv.imread(map)
    self.map_gray = cv.cvtColor(self.map_bgr, cv.COLOR_BGR2GRAY)
    _, self.map_gray = cv.threshold(self.map_gray, 128, 255, cv.THRESH_OTSU)

    self.likelihood_field = self._construct_likelihood_field(self.map_gray)
  
  def _gaussian_dist(self, x, mean , sigma):
    """
    Description: Calculate the gaussian distribution.

    Input:
      - x
      - mean
      - sigma: standard deviation

    Output:
      prob
    """
    prob = np.exp(-0.5*((x-mean)/sigma)**2) / (sigma * np.sqrt(2*np.pi))
    return prob
  
  def _construct_likelihood_field(self, map_gray, show_map=False):
    """
    Description: construct the likelihood field that shows how
    likely there can be an obstacle in a specific location in the map.

    Input:
      - map_gray: the map in grayscale.
    
    Output:
      - likelihood_field: 2d float matrix having the likelihoods.
    """
    likelihood_field = np.zeros_like(map_gray, dtype=np.float64)
    
    # Perform the distance transform to get the closest obstacle
    dist = cv.distanceTransform(map_gray, cv.DIST_L2, 3)

    # Calculate the gassuian distribution.
    sigma = self.gaussian_sigma
    likelihood_field = self._gaussian_dist(x=dist, mean=0, sigma=sigma)

    if show_map:
      cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
      cv.imshow('Distance Transform Image', dist)
      cv.normalize(likelihood_field, likelihood_field, 0, 1.0, cv.NORM_MINMAX)
      cv.imshow('likelihood_field', likelihood_field)
      cv.waitKey(0)

    return likelihood_field

  def get_likelihood(self, x, y, theta, show_ray=False):
    """
    Description: Get the likelihood of the ending point of the 
    ray which starts at (x, y) with a theta angle.
      
    Input:
      - x: np_array of the x coordinates on the map.
      - y: np_array of the y coordinates on the map.
      - theta: the heading direction on the map in degrees (1-360).
      0 degree makes the robot look horizontally right. The angle of the
      robot is incremented in the clock wise direction (i.e. 90 degree will
      make the robot look vertically down).
      - show_ray: if true it shows the ray start and end points.
    Output:
      -likelihoods 3D-grid
    """
    likelihoods = 0

    if show_ray:
      img = self.map_bgr.copy()

    # from metric unit to the number of pixels 
    end_len = (self.length_range * CM_IN_METER) // self.pixel_size 
    
    h, w = self.map_gray.shape
    
    endps_x = x + np.int(end_len * np.cos(np.radians(theta)))
    zero_lh_x = np.logical_or(endps_x < 0, endps_x >= w) 
    endps_x[zero_lh_x] = 0

    endps_y = y + np.int(end_len * np.sin(np.radians(theta)))
    zero_lh_y = np.logical_or(endps_y < 0, endps_y >= h) 
    endps_y[zero_lh_y] = 0
    
    likelihoods = self.likelihood_field[endps_y, endps_x]
    likelihoods[zero_lh_y] = 0
    likelihoods[zero_lh_x] = 0

    if show_ray:
      for i in range(x.shape[0]):
        pt1 = (x[i], y[i])
        pt2 = (endps_x[i], endps_y[i])
        cv.circle(img, pt1, 2, BGR_RED_COLOR, 5)
        cv.circle(img, pt2, 2, BGR_BLUE_COLOR, 5)
        cv.line(img, pt1, pt2, BGR_GREEN_COLOR, 3)      

      cv.imshow("img", img)
      print(likelihoods)
      cv.waitKey(0)
      cv.destroyAllWindows()

    return likelihoods

  def visualize_highest_likelihood_map(self):
    """
    Description:
    For each 2D-cell of the x,y-grid of the map above, visualize
    the highest likelihood of all orientations θ as gray value
    
    Output:
      - highest_likelihood_map
    """
    highest_likelihood_map = np.zeros_like(self.likelihood_field)

    indices = np.where(highest_likelihood_map == 0)
    y = indices[0]
    x = indices[1]

    for angle in range(0, 360, self.angle_accuracy):
      likelihoods = self.get_likelihood(x, y, angle) 
      likelihoods = likelihoods.reshape(highest_likelihood_map.shape)
      highest_likelihood_map = np.maximum(highest_likelihood_map, likelihoods)

    cv.normalize(highest_likelihood_map, 
    highest_likelihood_map, 0, 1.0, cv.NORM_MINMAX)
    cv.imshow('highest_likelihood_map', highest_likelihood_map)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return highest_likelihood_map

if __name__ == '__main__':
  end_point_model = EndPoint("Assignment_04_Grid_Map.png")
  
  x = np.array([400, 200, 100])
  y = np.array([180, 100, 200])
  likelihoods = end_point_model.get_likelihood(x, y, 30, True)
  likelihoods = end_point_model.get_likelihood(x, y, 42, True)
  likelihoods = end_point_model.get_likelihood(x, y, 45, True)

  end_point_model.visualize_highest_likelihood_map()