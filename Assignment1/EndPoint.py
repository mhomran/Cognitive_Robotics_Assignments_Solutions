"""
@file   : EndPoint.py
@brief  : a solution for the problem 4.2 in Assignment 1
@author : Mohamed Hassanin Mohamed
@data   : 25/03/2022
"""

import cv2 as cv
import numpy as np
from RayCaster import RayCaster


BGR_RED_COLOR = (0, 0, 255)
BGR_GREEN_COLOR = (0, 255, 0)
BGR_BLUE_COLOR = (255, 0, 0)

GRAYSCALE_WHITE_COLOR = 255
GRAYSCALE_BLACK_COLOR = 0

CM_IN_METER = 100

# threshold for normalization
TH = 8

class EndPoint:
  def __init__(self, map, gaussian_sigma=10, show_map=False,
   save_results=False) -> None:
    """
    Description: constructor for End Point model.

    Input:
      - map: the filename of the map where the ray will be casted.
      - pixel_size: the pixel size in real world in centimeters.
      - gaussian_sigma: the standard deviation of the gaussians of 
      the obstacles.
      - show_map: if true, show the likelhood_field and distance maps
      - save_results: if true, save the likelihood_field map.
    """
    self.gaussian_sigma = gaussian_sigma

    self.map_bgr = cv.imread(map)
    self.map_gray = cv.cvtColor(self.map_bgr, cv.COLOR_BGR2GRAY)
    _, self.map_gray = cv.threshold(self.map_gray, 128, 255, cv.THRESH_OTSU)

    self.likelihood_field = self._construct_likelihood_field(self.map_gray, show_map, save_results)

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
  
  def _construct_likelihood_field(self, map_gray, show_map=False, 
  save_results=False):
    """
    Description: construct the likelihood field that shows how
    likely there can be an obstacle in a specific location in the map.

    Input:
      - map_gray: the map in grayscale.
      - show_map: if true, show the likelihood_field and the distance maps
    
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
      dist_temp = dist.copy()
      cv.normalize(dist, dist_temp, 0, 1.0, cv.NORM_MINMAX)
      cv.imshow('Distance Transform Image', dist_temp)

      lf_temp = likelihood_field.copy()
      cv.normalize(likelihood_field, lf_temp, 0, 1.0, cv.NORM_MINMAX)
      cv.imshow('likelihood_field', lf_temp)
      
      cv.waitKey(0)
      cv.destroyAllWindows()

    if save_results:
      lf_temp = likelihood_field.copy()
      cv.normalize(likelihood_field, lf_temp, 0, 255, cv.NORM_MINMAX)
      cv.imwrite('likelihood_field.png', lf_temp)

    return likelihood_field

  def get_likelihood(self, measurements, angle_range, angle_accuracy, 
  show_ray=False, save_results=False):
    """
    Description: Get the likelihood of laser scan of 
    which starts at (x, y) with a theta angle.
      
    Input:
      - measurements
      - angle_range: the angle range to be scanned in degrees 
      from (1 to 360). Half of this range is for the right
      scan and the second half is for the left part.
      - angle_accuracy: An integer representing the step to
      increment the angle with in degrees.
      - show_ray: if true it shows the ray start and end points.
      - save_results: save the ray image
    Output:
      - likelihoods 3D-grid
    """
    h, w = self.map_gray.shape
    m = measurements.shape[0]

    likelihoods_grid = np.zeros((h, w, 360 // angle_accuracy))

    # generate x and y coordinates
    temp = np.zeros((h, w))
    indices = np.where(temp == 0)
    y = indices[0]
    x = indices[1]

    # endpoints for every x and y

    for i, theta in enumerate(range(0, 360, angle_accuracy)):
      start_angle = theta - (angle_range // 2)
      end_angle = theta + (angle_range // 2)
      scan_thetas = np.arange(start_angle, end_angle+1, angle_accuracy)
      scan_thetas = np.radians(scan_thetas)
      
      eps_x = np.zeros((h*w, m), dtype=np.int)
      eps_y = np.zeros((h*w, m), dtype=np.int)

      eps_x = (eps_x.T + x).T # add the x offset
      x_d = (np.cos(scan_thetas) * measurements).astype(np.int)
      eps_x = eps_x +  x_d # add the endpoint x
      
      eps_y = (eps_y.T + y).T # add the y offset
      eps_y = eps_y + (np.sin(scan_thetas) * measurements).astype(np.int) # add the endpoint x

      # out of the map borders
      x_outbound = np.logical_or(eps_x < 0, eps_x >= w)
      y_outbound = np.logical_or(eps_y < 0, eps_y >= h)
      eps_y[y_outbound] = 0
      eps_x[x_outbound] = 0
      outbound = np.logical_or(x_outbound, y_outbound)

      likelihoods = self.likelihood_field[eps_y, eps_x]
      log_lh = likelihoods.copy()
      log_lh[log_lh!=0] = np.log(log_lh[log_lh!=0])

      # out of bound measurement
      log_lh[outbound] = -np.inf

      log_lh[:, measurements < 0] = 0

      log_lh = np.sum(log_lh, axis=-1)
      log_lh = log_lh.reshape((h, w))

      max_coord = np.unravel_index(log_lh.argmax(), log_lh.shape)
      print(f"theta:{theta}, {max_coord}, log lh:{log_lh[max_coord]}")

      likelihoods_grid[:, :, i] = log_lh

    return likelihoods_grid

  def visualize_highest_likelihood_map(self, likelihoods_grid):
    """
    Description:
    For each 2D-cell of the x,y-grid of the map above, visualize
    the highest likelihood of all orientations θ as gray value

    Input:
      - likelihoods_grid: 3d grid of x, y, theta 
    
    Output:
      - highest_likelihood_map
    """
    highest_likelihood_map = np.zeros_like(self.likelihood_field)
    highest_likelihood_map = np.max(likelihoods_grid, axis=-1)
    
    hl_map_flt = highest_likelihood_map.copy()
    max_coord = np.unravel_index(hl_map_flt.argmax(), hl_map_flt.shape)

    hl_map_flt[hl_map_flt < hl_map_flt[max_coord] - TH] = hl_map_flt[max_coord] - TH
    cv.normalize(hl_map_flt, hl_map_flt, 0, 1, cv.NORM_MINMAX)
    
    max_coord = np.unravel_index(hl_map_flt.argmax(), hl_map_flt.shape)
    print(max_coord, hl_map_flt[max_coord])
    
    cv.imshow('highest_likelihood_map', hl_map_flt)
    cv.waitKey(0)
    cv.destroyAllWindows()

    hl_map_u8 = highest_likelihood_map.copy()
    cv.normalize(hl_map_flt, hl_map_u8, 0, 255, cv.NORM_MINMAX)

    return hl_map_u8

if __name__ == '__main__':
  angle_accuracy = 10
  x = 380
  y = 140
  theta = 90

  ray_caster = RayCaster("Assignment_04_Grid_Map.png", 
  angle_accuracy=angle_accuracy)

  measurements = ray_caster.cast(x=x, y=y, theta=theta,
  show_rays=True, show_collided=True, save_results=True)

  ep_model = EndPoint("Assignment_04_Grid_Map.png", show_map=True, save_results=True)

  grid = ep_model.get_likelihood(measurements, angle_range=250, angle_accuracy=angle_accuracy)
  
  hl_map = ep_model.visualize_highest_likelihood_map(grid)

  cv.imwrite(f'Likelihood_theta_{theta}_x_{x}_y_{y}_angle_accuracy_{angle_accuracy}.png', hl_map)

