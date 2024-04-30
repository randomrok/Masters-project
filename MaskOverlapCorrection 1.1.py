# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 11:31:09 2024

@author: Elijah Gardi
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



def spect_mask(organ_mask, expanded_mask, seconday_mask):

    ''' Mask that recovers the actual activity '''
    
    
    return expanded_mask > 0.5


''' Gamma Camera blur '''

SIGMA = 15

 

''' Change distance and size '''

small_dist = 360

small_height = 200

small_radius = 50

 

'''L1 and L2 are lesion 1 and lesion 2. The G is to indicate filtering '''

L1 = create_circular_mask(500, 500, center=(200,200), radius=110)

L1_G = gaussian_filter(L1.astype('float'), sigma=SIGMA)



L2 = create_circular_mask(500, 500, center=(small_dist,small_height), radius=small_radius)

L2_G = gaussian_filter(L2.astype('float'), sigma=SIGMA)



mask_L1  = create_circular_mask(500, 500, center=(200,200), radius=110*1.25)

mask_L2  = create_circular_mask(500, 500, center=(small_dist,small_height), radius=small_radius*1.35)




"-------------------------------"
"vestigial code used to show the histogram of the PET data and the local minima (origin of mask)"
'''
Section = PET_Image[250,:]
plt.plot(Section)
plt.show(plt)
fig, (ax1) = plt.subplots(1)
ax1.imshow(LocalMinima, vmin=0,vmax=1)
'''

"-------------------------------"
'Uses the fact that the gradient of the pet image is a local minimum point at the middle of the intersection (see histogram)'
"Uses the coordinates of the local minimum and the masks center to seperate the intersection regions into sections to be filled along +-x,+-y directions"

"VARIABLE DECLERATIONS"

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

"Origin is the average of all points from mask"
x = [p[0] for p in maskL2coordinates]
y = [p[1] for p in maskL2coordinates]
centroid = (sum(x) / len(maskL2coordinates), sum(y) / len(maskL2coordinates))
Direction_Vector = tuple

PET_Image = gaussian_filter(L1.astype('float'), sigma=SIGMA) + gaussian_filter(L2.astype('float'), sigma=SIGMA)
size = PET_Image.shape
RateOfChange = np.ones(size, dtype=float)
LocalMinima = np.zeros(size, dtype=bool)

IntersectionCoordsXMax = max(IntersectionCoords[:,0])
IntersectionCoordsXMin = min(IntersectionCoords[:,0])
IntersectionCoordsYMax = max(IntersectionCoords[:,1])
IntersectionCoordsYMin = min(IntersectionCoords[:,1])

L2Intersection = np.zeros(size, dtype=bool)

RateOfChange = np.ones(size, dtype=float) # Resets the rate of change matrix and LocalMinima matrix
LocalMinima = np.zeros(size, dtype=bool)
HorizontalLocalMinima = np.zeros(size, dtype=bool)
VerticalLocalMinima = np.zeros(size, dtype=bool)

"FUNCTIONS"

'Uses the fact that the gradient of the pet image is a local minimum point at the middle of the intersection (see histogram)'
for a in range(size[0]-1):
    for b in range(size[1]-1):
        dy = abs(PET_Image[a,b] - PET_Image[a+1,b])
        dx = abs(PET_Image[a,b] - PET_Image[a,b+1])
        RateOfChange[a,b] = dy + dx
        
        if a in range(IntersectionCoordsYMin, IntersectionCoordsYMax) and b in range(IntersectionCoordsXMin, IntersectionCoordsXMax):
            if abs(RateOfChange[a,b]) < 0.001:
                LocalMinima[a,b] = True
                
"calculates horizontal local minima matrix"
for a in range(size[0]-1):
    for b in range(size[1]-1):
        dy = abs(PET_Image[a,b] - PET_Image[a+1,b])
        dx = abs(PET_Image[a,b] - PET_Image[a,b+1])
        RateOfChange[a,b] = dx
        
        if a in range(IntersectionCoordsYMin, IntersectionCoordsYMax) and b in range(IntersectionCoordsXMin, IntersectionCoordsXMax):
            if abs(RateOfChange[a,b]) < 0.001:
                HorizontalLocalMinima[a,b] = True
                
"Calculates vertical local minima matrix"
for a in range(size[0]-1):
    for b in range(size[1]-1):
        dy = abs(PET_Image[a,b] - PET_Image[a+1,b])
        dx = abs(PET_Image[a,b] - PET_Image[a,b+1])
        RateOfChange[a,b] = dy
        
        if a in range(IntersectionCoordsYMin, IntersectionCoordsYMax) and b in range(IntersectionCoordsXMin, IntersectionCoordsXMax):
            if abs(RateOfChange[a,b]) < 0.001:
                VerticalLocalMinima[a,b] = True


"COMBINED FILLING FUNCTION"


for c in range(IntersectionCoordsXMin, IntersectionCoordsXMax):
    for d in range(IntersectionCoordsYMin, IntersectionCoordsYMax): 
    # Loop through intersection coordinates region
        if LocalMinima[d,c]:
        # Check for the LocalMinima points "Use local minima coordinate with the direction from the origin of the mask to determine left and right/top and down filling direction"
            Direction_Vector = np.subtract(np.array([c,d]), centroid)
            Magnitude = np.linalg.norm(Direction_Vector)
            Direction_Vector = Direction_Vector/Magnitude
            
            "LEFT FILLING DIRECTION"
            if Direction_Vector[0] < 0 and abs(Direction_Vector[1]) < 0.5: # Check unit vector is to left and between 0.5 vertically

                for c in range(IntersectionCoordsXMin, IntersectionCoordsXMax):
                    for d in range(IntersectionCoordsYMin, IntersectionCoordsYMax): 
                    # Loop through intersection coordinates region
                        if HorizontalLocalMinima[d,c]:
                        # Check for the LocalMinima points
                            for a in range(IntersectionCoords.shape[0]):
                            # Loop through every intersection coordinate
                                if IntersectionCoords[a,1] == d and IntersectionCoords[a,0] < c: 
                                # Check that the intersection coordinate is not greater than the LocalMinima coordinate
                                    L2Intersection[IntersectionCoords[a,1], IntersectionCoords[a,0]] = True
                                    # Set all points (Limited to intersection points and less than local minima) to the PET image. 
                                    # IntersectionCoords[a,1] == d leftIntersection matrix is updated only when the y coord is the same as the Localminia y coord
            
            "TOP FILLING DIRECTION"
            if Direction_Vector[1] < 0 and abs(Direction_Vector[0]) < 0.5: # Check unit vector is pointing up and between 0.5 horizontally

                "Uses vertical local minima matrix to seperate the intersection"
                for c in range(IntersectionCoordsXMin, IntersectionCoordsXMax):
                    for d in range(IntersectionCoordsYMin, IntersectionCoordsYMax): 
                    # Loop through intersection coordinates region
                        if VerticalLocalMinima[d,c]:
                        # Check for the LocalMinima points
                            for a in range(IntersectionCoords.shape[0]):
                            # Loop through every intersection coordinate
                                if IntersectionCoords[a,0] == c and IntersectionCoords[a,1] < d: #check it is the first local minima with that x coord
                                    L2Intersection[IntersectionCoords[a,1], IntersectionCoords[a,0]] = True
                                    # Set all points (Limited to intersection points and less than local minima) to the PET image. 
                                    # IntersectionCoords[a,0] == c leftIntersection matrix is updated only when the x coord is the same as the Localminia x coord


"CALCULATES NEW MASKS"
"RIGHT MASK"
mask_spect2 = mask_L2 ^ L2Intersection


"SET INTERSECTION"
Intersection = mask_L1 ^ mask_L2
Intersection = Intersection ^ (mask_L1 + mask_L2)

"SET DIFFERENCE TO OBTAIN LEFT MASK"
RightIntersection = Intersection ^ L2Intersection

"LEFT MASK"
mask = mask_L1 ^ RightIntersection



plt.imshow(L2Intersection)
"Rename to L1 intersection and L2 intersection to combine functions"
"Can calculate the total mask data from the original mask using the set difference between the calculated one and L2 for example"
"Calculate the fill direction based off individual coordinates to avoid problems at angles"
"Calculate the average"
"Performance issues likely due to looping where is unneccessary"
"-------------------------------"


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