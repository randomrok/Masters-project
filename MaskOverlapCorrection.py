# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:59:44 2024

@author: Steven Goodman and Elijah Gardi
"""

from matplotlib import pyplot as plt

import numpy as np

from scipy.ndimage import gaussian_filter

from skimage import measure

import shapely.plotting
from shapely.geometry import Polygon

def contour_mask(mask, ax, color):
    '''

    Input:  

    mask: Binary mask 2D numpy array

    ax:   pyplot axes

    '''

    contours = measure.find_contours(mask, 0.5)

    for n, contour in enumerate(contours):

        ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color = color, linestyle='dashed')



def create_circular_mask(h, w, center=None, radius=None):



    if center is None: # use the middle of the image

        center = (int(w/2), int(h/2))

    if radius is None: # use the smallest distance between the center and image walls

        radius = min(center[0], center[1], w-center[0], h-center[1])



    Y, X = np.ogrid[:h, :w]

    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)




    mask = dist_from_center <= radius


    return mask



def spect_mask(organ_mask, expanded_mask, secondary_mask):

    ''' Mask that recovers the actual activity '''
    
    
    return expanded_mask > 0.5



''' Gamma Camera blur '''

SIGMA = 15



''' Change distance and size '''

small_dist = 360

small_radius = 35





'''L1 and L2 are lesion 1 and lesion 2.  The G is to indicate filtering '''

L1 = create_circular_mask(500, 500, center=(200,200), radius=110)


L1_G = gaussian_filter(L1.astype('float'), sigma=SIGMA)


L2 = create_circular_mask(500, 500, center=(small_dist,200), radius=small_radius)


L2_G = gaussian_filter(L2.astype('float'), sigma=SIGMA)


mask_L1  = create_circular_mask(500, 500, center=(200,200), radius=110*1.25)

mask_L2  = create_circular_mask(500, 500, center=(small_dist,200), radius=small_radius*1.35)



fig, (ax1, ax2) = plt.subplots(1, 2)

fig.set_figheight(8)

fig.set_figwidth(16)






'''Elijahs code'''

'''
Create a line between the two overlapping masks representing 50% on either side from each intersection point.

This might be achievable by creating a matrix of points that divides the intersection in two.
Could use a vector whose origin is the intersection origin plus the average radius 
and whose radius is the average radius.
'''

center1 = 200
radius1 = 110*1.25
center2 = small_dist
radius2 = small_radius*1.35

center3 = (center2 + center1 + radius1 - radius2)/2 + (radius2 + radius1)/2

radius3 = (radius2 + radius1)/2

L3 = create_circular_mask(500, 500, center = (center3,200), radius = radius3)


contour_mask(L3, ax2, 'blue')
contour_mask(L3, ax1, 'blue')


rows, cols = np.where(L3)
L3coordinates = np.column_stack((cols, rows))


rows, cols = np.where(mask_L2)
L2coordinates = np.column_stack((cols, rows))

rows, cols = np.where(mask_L1)
L1coordinates = np.column_stack((cols, rows))


"-----------------------------------------"
"Calculating intersection regions"

"Right intersection coordinates"
nrows, ncols = L1coordinates.shape
dtype = {'names': ['f{}'.format(i) for i in range(ncols)], 'formats': ncols * [L1coordinates.dtype]}
R_IntersectionCoords = np.intersect1d(L1coordinates.view(dtype), L3coordinates.view(dtype))
R_IntersectionCoords = R_IntersectionCoords.view(L1coordinates.dtype).reshape(-1, ncols)

"Calculates whole intersection coordinates"
nrows, ncols = L2coordinates.shape
dtype = {'names': ['f{}'.format(i) for i in range(ncols)], 'formats': ncols * [L2coordinates.dtype]}
IntersectionCoords = np.intersect1d(L2coordinates.view(dtype), L1coordinates.view(dtype))
IntersectionCoords = IntersectionCoords.view(L2coordinates.dtype).reshape(-1, ncols)

"Calculates set difference to find left intersection coordinates"
nrows, ncols = IntersectionCoords.shape
dtype = {'names': ['f{}'.format(i) for i in range(ncols)], 'formats': ncols * [IntersectionCoords.dtype]}
L_IntersectionCoords = np.setdiff1d(IntersectionCoords.view(dtype), R_IntersectionCoords.view(dtype))
L_IntersectionCoords = IntersectionCoords.view(L2coordinates.dtype).reshape(-1, ncols)



"Can display the left and right regions from coordinates using shapely"
'R_IntersectionShape = Polygon(R_IntersectionCoords)'
'shapely.plotting.plot_polygon(R_IntersectionShape)'


"-----------------------Something wrong here"
"Converting coordinate matrix back to image matrix to extract data"
L_Intersection = [L1_G[L_IntersectionCoords[r,1],L_IntersectionCoords[r,0]] + 
                  L2_G[L_IntersectionCoords[r,1],L_IntersectionCoords[r,0]]
                  for r,r in L_IntersectionCoords]
L_Intersection = np.array(L_Intersection)

'''contour_mask(L_Intersection, ax2, 'red')'''

R_Intersection = [L1_G[R_IntersectionCoords[r,1],R_IntersectionCoords[r,0]] + 
                  L2_G[R_IntersectionCoords[r,1],R_IntersectionCoords[r,0]] 
                  for r,r in R_IntersectionCoords]
R_Intersection = np.array(R_Intersection)

Intersection = [L1_G[IntersectionCoords[r,1],IntersectionCoords[r,0]] + 
                  L2_G[IntersectionCoords[r,1],IntersectionCoords[r,0]] 
                  for r,r in IntersectionCoords]
Intersection = np.array(Intersection)


"mask is the ratio of counts from the L & R intersection regions"
Lsum = np.sum(L_Intersection)
Rsum = np.sum(R_Intersection)
IntersectionSum = np.sum(Intersection)



mask = Lsum / Rsum


Small_masked_sum = np.sum((L1_G+L2_G)*mask_L2) - IntersectionSum/mask
Large_masked_sum = np.sum((L1_G+L2_G)*mask_L1) - Lsum*mask

actual_sum = np.sum(L1)

print('Overlap correction Small ROI vs actual: ', round(Small_masked_sum/actual_sum, 2))

actual_sum = np.sum(L2)

print('Overlap correction Large ROI vs actual: ', round(Large_masked_sum/actual_sum, 2))
''''Elijahs code'''






ax1.imshow(L1+L2)
contour_mask(mask_L1, ax2, 'red')
contour_mask(mask_L1, ax1, 'red')
contour_mask(mask_L2, ax1, 'black')
contour_mask(mask_L2, ax2, 'black')

ax1.axis('off')

ax2.imshow(L1_G+L2_G, vmin=0,vmax=1)

ax2.axis('off')




masked_sum = np.sum((L1_G+L2_G))

actual_sum = np.sum(L1)


print('Large ROI masked vs actual counts: ', round(masked_sum/actual_sum, 2))

plt.title(round(masked_sum/actual_sum, 3))


actual_sum = np.sum(L2)
 

print('Small ROI masked vs actual counts: ', round(masked_sum/actual_sum, 2))

