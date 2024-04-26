# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 11:31:09 2024

@author: Elijah Gardi
"""

from matplotlib import pyplot as plt

import numpy as np

from scipy.ndimage import gaussian_filter

from skimage import measure

import math

from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max

import shapely.plotting
from shapely.geometry import Polygon

import cv2

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



def spect_mask(organ_mask, expanded_mask, seconday_mask):

    ''' Mask that recovers the actual activity '''
    
    
    return expanded_mask > 0.5


''' Gamma Camera blur '''

SIGMA = 15

 

''' Change distance and size '''

small_dist = 360

small_radius = 50

 

'''L1 and L2 are lesion 1 and lesion 2. The G is to indicate filtering '''

L1 = create_circular_mask(500, 500, center=(200,200), radius=110)

L1_G = gaussian_filter(L1.astype('float'), sigma=SIGMA)



L2 = create_circular_mask(500, 500, center=(small_dist,200), radius=small_radius)

L2_G = gaussian_filter(L2.astype('float'), sigma=SIGMA)



mask_L1  = create_circular_mask(500, 500, center=(200,200), radius=110*1.25)

mask_L2  = create_circular_mask(500, 500, center=(small_dist,200), radius=small_radius*1.35)




"-------------------------------"
"Obtains coordinates of masks"
rows, cols = np.where(mask_L1)
maskL1coordinates = np.column_stack((cols, rows))

rows, cols = np.where(mask_L2)
maskL2coordinates = np.column_stack((cols, rows))

"Calculates intersection coordinates"
nrows, ncols = maskL1coordinates.shape
dtype = {'names': ['f{}'.format(i) for i in range(ncols)], 'formats': ncols * [maskL1coordinates.dtype]}
IntersectionCoords = np.intersect1d(maskL1coordinates.view(dtype), maskL2coordinates.view(dtype))
IntersectionCoords = IntersectionCoords.view(maskL1coordinates.dtype).reshape(-1, ncols)


"MIDDLE OF INTERSECTION"
'Uses the fact that the gradient of the pet image is a local minimum point at the middle of the intersection (see histogram)'
'would need to add a way to check which direction the intersection region is in respect to the origin'
PET_Image = gaussian_filter(L1.astype('float'), sigma=SIGMA) + gaussian_filter(L2.astype('float'), sigma=SIGMA)
size = PET_Image.shape
RateOfChange = np.ones(size, dtype=float)
LocalMinima = np.zeros(size, dtype=bool)

IntersectionCoordsXMax = max(IntersectionCoords[:,0])
IntersectionCoordsXMin = min(IntersectionCoords[:,0])
IntersectionCoordsYMax = max(IntersectionCoords[:,1])
IntersectionCoordsYMin = min(IntersectionCoords[:,1])

for a in range(size[0]-1):
    for b in range(size[1]-1):
        dy = PET_Image[a,b] - PET_Image[a+1,b]
        dx = PET_Image[a,b] - PET_Image[a,b+1]
        RateOfChange[a,b] = dx
        
        if a in range(IntersectionCoordsYMin, IntersectionCoordsYMax) and b in range(IntersectionCoordsXMin, IntersectionCoordsXMax):
            if abs(RateOfChange[a,b]) < 0.001:
                LocalMinima[a,b] = 'true'
            
"vestigial code used to show the histogram of the PET data and the local minima"
'''
Section = PET_Image[250,:]
plt.plot(Section)
plt.show(plt)
fig, (ax1) = plt.subplots(1)
ax1.imshow(LocalMinima, vmin=0,vmax=1)
'''

"Origin is the average of all points from mask"
x = [p[0] for p in maskL1coordinates]
y = [p[1] for p in maskL1coordinates]
centroid = (sum(x) / len(maskL1coordinates), sum(y) / len(maskL1coordinates))

"Create an average of all those true values generated into a single line of values..? Performance increase..?"

"LEFT INTERSECTION DIRECTION"
LeftIntersection = np.zeros(size, dtype=float)
MinimaCoordsXMin = min(LocalMinima[:,1])
MinimaCoordsXMax = max(LocalMinima[:,1])
MinimaCoordsYMin = min(LocalMinima[:,0])
MinimaCoordsYMax = max(LocalMinima[:,0])
"use the coordinates of the middle regoin to seperate the coordinates of the left and right regions"
"This code seperates the left region"
"Find a way to code populating the matrix from the left to right"
for c in range(IntersectionCoordsXMin, IntersectionCoordsXMax):
    for d in range(IntersectionCoordsYMin, IntersectionCoordsYMax): 
    # Loop through entire image
        if LocalMinima[d,c]:
        # Check for the LocalMinima points
            for a in range(IntersectionCoords.shape[0]):
            # Loop through every intersection coordinate
                if IntersectionCoords[a,1] == d and IntersectionCoords[a,0] < c: 
                # Check that the intersection coordinate is not greater than the LocalMinima coordinate
                    LeftIntersection[IntersectionCoords[a,0], IntersectionCoords[a,1]] = PET_Image[IntersectionCoords[a,0], IntersectionCoords[a,1]]
                    # Set all points (Limited to intersection points and less than local minima) to the PET image. 
                    # IntersectionCoords[a,0] == c leftIntersection matrix is updated only when the x coord is the same as the Localminia x coord

plt.imshow(LeftIntersection)
"-------------------------------"





mask = spect_mask(L1, mask_L1, mask_L2)

mask_spect2 = spect_mask(L2, mask_L2, mask_L1)

 

fig, (ax1, ax2) = plt.subplots(1, 2)

fig.set_figheight(8)

fig.set_figwidth(16)

 

ax1.imshow(L1+L2)

ax1.axis('off')

ax2.imshow(L1_G+L2_G, vmin=0,vmax=1)

ax2.axis('off')

contour_mask(mask, ax2, 'red')

contour_mask(mask, ax1, 'red')

contour_mask(mask_spect2, ax2, 'black')

 

masked_sum = np.sum(mask*(L1_G+L2_G))

actual_sum = np.sum(L1)

 

print('Large ROI masked vs actual counts: ', round(masked_sum/actual_sum, 2))

plt.title(round(masked_sum/actual_sum, 3))

 

masked_sum = np.sum(mask_spect2*(L1_G+L2_G))

actual_sum = np.sum(L2)

 

print('Small ROI masked vs actual counts: ', round(masked_sum/actual_sum, 2))