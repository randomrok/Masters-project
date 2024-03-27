# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:59:44 2024

@author: Steven Goodman and Elijah Gardi
"""

from matplotlib import pyplot as plt

import numpy as np

from scipy.ndimage import gaussian_filter

from skimage import measure



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


    
    
    return expanded_mask > 0.5



''' Gamma Camera blur '''

SIGMA = 15



''' Change distance and size '''

small_dist = 360

small_radius = 50



'''L1 and L2 are lesion 1 and lesion 2.  The G is to indicate filtering '''

L1 = create_circular_mask(500, 500, center=(200,200), radius=110)


L1_G = gaussian_filter(L1.astype('float'), sigma=SIGMA)


L2 = create_circular_mask(500, 500, center=(small_dist,200), radius=small_radius)


L2_G = gaussian_filter(L2.astype('float'), sigma=SIGMA)


mask_L1  = create_circular_mask(500, 500, center=(200,200), radius=110*1.25)

mask_L2  = create_circular_mask(500, 500, center=(small_dist,200), radius=small_radius*1.35)


mask = spect_mask(L1, mask_L1, mask_L2)

mask_spect2 = spect_mask(L2, mask_L2, mask_L1)



fig, (ax1, ax2) = plt.subplots(1, 2)

fig.set_figheight(8)

fig.set_figwidth(16)


''' Mask that recovers the actual activity 

Create a line between the two overlapping masks representing 50% on either side from each intersection point.

This might be achievable by creating a matrix of points that divides the intersection in two.
Could use a vector whose origin is the intersection origin plus the radius of the smaller circle
and whose radius is the smaller one plus 1/3 the larger radius.
'''

'intersection = expanded_mask.intersection(secondary_mask)'


'''Elijahs code'''
center1 = 200
radius1 = 110
center2 = small_dist*1.25
radius2 = small_radius*1.35

center3 = (center2 + center1 + radius1 - radius2)/2 + radius2

radius3 = (radius2 + radius1/3)
L3 = create_circular_mask(500, 500, center = (center3,200), radius = radius3)

ax1.imshow(L3)

contour_mask(L3, ax2, 'blue')

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

