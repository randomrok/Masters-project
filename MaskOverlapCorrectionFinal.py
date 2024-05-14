# -*- coding: utf-8 -*-
"""
Created on Mon May 13 19:20:35 2024

@author: Elijah Gardi
"""

from matplotlib import pyplot as plt

import numpy as np

from scipy import ndimage

from numpy import ndarray

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

small_dist = 200

small_height = 60

small_radius = 30



'''L1 and L2 are lesion 1 and lesion 2. The G is to indicate filtering '''

L1 = create_circular_mask(500, 500, center=(200,200), radius=100)

L1_G = gaussian_filter(L1.astype('float'), sigma=SIGMA)



L2 = create_circular_mask(500, 500, center=(small_dist,small_height), radius=small_radius)

L2_G = gaussian_filter(L2.astype('float'), sigma=SIGMA)

mask_L1  = create_circular_mask(500, 500, center=(200,200), radius=110*1.25)

mask_L2  = create_circular_mask(500, 500, center=(small_dist,small_height), radius=small_radius*1.35)

fig, (ax1, ax2) = plt.subplots(1, 2)

fig.set_figheight(8)

fig.set_figwidth(16)
"-------------------------------------------------------------------------------------"

"---------------------------------"
# Finding path between two cell in matrix
 
# Method for finding and printing
# whether the path exists or not
def isPath(matrix, n):
 
    # Defining visited array to keep
    # track of already visited indexes
    visited = [[False for x in range(n)]
               for y in range(n)]
    
    # Flag to indicate whether the
    # path exists or not
    flag = False
 
    for i in range(n):
        for j in range(n):
 
            # If matrix[i][j] is source
            # and it is not visited
            if (matrix[i][j] == 1 and not
                    visited[i][j]):
 
                # Starting from i, j and
                # then finding the path
                if (checkPath(matrix, i,
                              j, visited)):
 
                    # If path exists
                    flag = True
                    break
    if (flag):
        return True
    else:
        return False
 
# Method for checking boundaries
def isSafe(i, j, matrix):
 
    if (i >= 0 and i < len(matrix) and
            j >= 0 and j < len(matrix[0])):
        return True
    return False
 
# Returns true if there is a
# path from a source(a
# cell with value 1) to a
# destination(a cell with
# value 2)
 
 
def checkPath(matrix, i, j,
              visited):
 
    # Checking the boundaries, walls and
    # whether the cell is unvisited
    if (isSafe(i, j, matrix) and
        matrix[i][j] != 0 and not
            visited[i][j]):
 
        # Make the cell visited
        visited[i][j] = True
 
        # If the cell is the required
        # destination then return true
        if (matrix[i][j] == 2):
            return True
 
        # traverse up
        up = checkPath(matrix, i - 1,
                       j, visited)
 
        # If path is found in up
        # direction return true
        if (up):
            return True
 
        # Traverse left
        left = checkPath(matrix, i,
                         j - 1, visited)
 
        # If path is found in left
        # direction return true
        if (left):
            return True
 
        # Traverse down
        down = checkPath(matrix, i + 1,
                         j, visited)
 
        # If path is found in down
        # direction return true
        if (down):
            return True
 
        # Traverse right
        right = checkPath(matrix, i,
                          j + 1, visited)
 
        # If path is found in right
        # direction return true
        if (right):
            return True
 
    # No path has been found
    return False

# This code is contributed by Chitranayal
"---------------------------------"

"Variable declerations"
Intersection = mask_L1 ^ mask_L2
Intersection = Intersection ^ (mask_L1 + mask_L2)

"Obtains coordinates of masks"
rows, cols = np.where(mask_L1)
maskL1coordinates = np.column_stack((cols, rows))

rows, cols = np.where(mask_L2)
maskL2coordinates = np.column_stack((cols, rows))

rows, cols = np.where(Intersection)
IntersectionCoords = np.column_stack((cols, rows))

"Intersection boundries for speed"
IntersectionCoordsXMax = max(IntersectionCoords[:,0])
IntersectionCoordsXMin = min(IntersectionCoords[:,0])
IntersectionCoordsYMax = max(IntersectionCoords[:,1])
IntersectionCoordsYMin = min(IntersectionCoords[:,1])

"Define image"
PET_Image = L1_G + L2_G

"Empty path matrix"
pathMask = np.zeros(PET_Image.shape, bool)

L2Intersection = np.zeros(PET_Image.shape, bool)

"Functions"

def StepMinus(matrix, stepsize):
    for a in range (matrix.shape[0]):
        for b in range (matrix.shape[1]): # loop through matrix
            
            if matrix[a,b] > stepsize and not matrix[a,b] == 3 and not matrix[a,b] == 2 and not matrix[a,b] == 1:
                matrix[a,b] = matrix[a,b] - stepsize
            
            if matrix[a,b] <= stepsize: 
                matrix[a,b] = 0
    return matrix


def FindMinima(image, stepsize):
    imageMatrix = image * Intersection
    PathMatrix = imageMatrix
    
    "Start and end points"
    x_start, y_start = min(IntersectionCoords, key=lambda x: (x[0], -x[1]))
    x_end, y_end = max(IntersectionCoords, key=lambda x: (x[0], -x[1]))
    
    while not isPath(PathMatrix, 500):
        imageMatrix = StepMinus(imageMatrix, stepsize) # Subtract 'stepsize' from all pixels until there is a path
        PathMatrix = ndarray.copy(imageMatrix) # Matrix for checking path
        for a in range(PathMatrix.shape[0]):
            for b in range(PathMatrix.shape[1]): # Loop through all pixels
                if PathMatrix[a,b] == 0:
                    PathMatrix[a,b] = 3 # Path cell
                else: 
                    PathMatrix[a,b] = 0 # Blocked cell
        
        "Start and end pixels"
        PathMatrix[y_start, x_start] = 1 # 'Starting cell'
        PathMatrix[y_end, x_end] = 2 # 'Destination cell'
        
        PathMatrix = PathMatrix*Intersection # Reset out of intersection to blocked cells (0)
    
    if isPath(PathMatrix, 500):
        for a in range(PET_Image.shape[0]):
            for b in range(PET_Image.shape[1]):
                if imageMatrix[a,b] == 0:
                    pathMask[a,b] = True
                else: pathMask[a,b] = False
        return pathMask*Intersection
        
        return "Error"


def GetCentroid(mask):
    "Origin is the average of all points from mask"
    x = [p[0] for p in mask]
    y = [p[1] for p in mask]
    centroid = (sum(x) / len(mask), sum(y) / len(mask))
    return centroid


def Check_Direction(Centroid, c, d):
        # Check for the LocalMinima points "Use local minima coordinate with the direction from the origin of the mask to determine left and right/top and down filling direction"
            Direction_Vector = np.subtract(Centroid, np.array([c,d]))
            Magnitude = np.linalg.norm(Direction_Vector)
            Direction_Vector = Direction_Vector/Magnitude
            return Direction_Vector


pathMask = FindMinima(PET_Image, 0.05)
# copyMask = ndarray.copy(pathMask)
# pathMask = ndarray.copy(copyMask)

plt.imshow(pathMask)
plt.imshow(L2Intersection)
"Calculate centroid of secondary mask and intersection for direction"
centroid = GetCentroid(maskL2coordinates)
intersecCentroid = GetCentroid(IntersectionCoords)

"Create Direction (unit) Vector"
Direction_Vector = Check_Direction(intersecCentroid, centroid[0], centroid[1])


"LEFT FILLING DIRECTION"
if Direction_Vector[0] < 0 and abs(Direction_Vector[1]) < 0.5: # Check unit vector is to left and between 0.5 vertically
    
    "Average True values along x axis"
    for d in range(IntersectionCoordsYMin, IntersectionCoordsYMax):
        for k in range(IntersectionCoordsXMin, IntersectionCoordsXMax):
            Sum = 0
            if pathMask[d,k]:
                for a in range(IntersectionCoordsXMin, IntersectionCoordsXMax):
                    if pathMask[d,a]:
                        Sum +=1
                        pathMask[d,a] = False
                pathMask[d,k+int(Sum/2)] = True
             
    for c in range(IntersectionCoordsXMin, IntersectionCoordsXMax):
        for d in range(IntersectionCoordsYMin, IntersectionCoordsYMax):# Loop through intersection coordinates region
            if pathMask[d,c]: # Check for the LocalMinima points
                for a in range(IntersectionCoords.shape[0]): # Loop through every intersection coordinate
                    if IntersectionCoords[a,1] == d and IntersectionCoords[a,0] < c: # Check that the intersection coordinate is not greater than the LocalMinima coordinate
                        L2Intersection[IntersectionCoords[a,1], IntersectionCoords[a,0]] = True
                        # Set all points (Limited to intersection points and less than local minima) to the PET image. 
                        # IntersectionCoords[a,1] == d leftIntersection matrix is updated only when the y coord is the same as the Localminia y coord
 
"RIGHT FILLING DIRECTION"
if Direction_Vector[0] > 0 and abs(Direction_Vector[1]) < 0.5: # Check unit vector is to right and between 0.5 vertically
    
    "Average True values along x axis"
    for d in range(IntersectionCoordsYMin, IntersectionCoordsYMax):
        for k in range(IntersectionCoordsXMin, IntersectionCoordsXMax):
            Sum = 0
            if pathMask[d,k]:
                for a in range(IntersectionCoordsXMin, IntersectionCoordsXMax):
                    if pathMask[d,a]:
                        Sum +=1
                        pathMask[d,a] = False
                pathMask[d,k+int(Sum/2)] = True
                    
    for c in range(IntersectionCoordsXMin, IntersectionCoordsXMax):
        for d in range(IntersectionCoordsYMin, IntersectionCoordsYMax): # Loop through intersection coordinates region
            if pathMask[d,c]: # Check for the LocalMinima points
                for a in range(IntersectionCoords.shape[0]): # Loop through every intersection coordinate
                    if IntersectionCoords[a,1] == d and IntersectionCoords[a,0] > c: # Check that the intersection coordinate is not greater than the LocalMinima coordinate
                        L2Intersection[IntersectionCoords[a,1], IntersectionCoords[a,0]] = True
                        # Set all points (Limited to intersection points and less than local minima) to the PET image. 
                        # IntersectionCoords[a,1] == d leftIntersection matrix is updated only when the y coord is the same as the Localminia y coord
        
"TOP FILLING DIRECTION"
if (Direction_Vector[1] < 0 and abs(Direction_Vector[0]) < 0.5): # Check unit vector is pointing up and between 0.5 horizontally

    "Average True values along y axis"
    for k in range(IntersectionCoordsXMin, IntersectionCoordsXMax):
        for d in range(IntersectionCoordsYMin, IntersectionCoordsYMax):
            Sum = 0
            if pathMask[d,k]:
                for a in range(IntersectionCoordsYMin, IntersectionCoordsYMax):
                    if pathMask[a,k]:
                        Sum +=1
                        pathMask[a,k] = False
                pathMask[d+int(Sum/2),k] = True
                
    "Uses vertical local minima matrix to seperate the intersection"
    for c in range(IntersectionCoordsXMin, IntersectionCoordsXMax):
        for d in range(IntersectionCoordsYMin, IntersectionCoordsYMax): 
        # Loop through intersection coordinates region
            if pathMask[d,c]:
            # Check for the LocalMinima points
                for a in range(IntersectionCoords.shape[0]):
                # Loop through every intersection coordinate
                    if IntersectionCoords[a,0] == c and IntersectionCoords[a,1] < d: #check it is the first local minima with that x coord
                        L2Intersection[IntersectionCoords[a,1], IntersectionCoords[a,0]] = True
                        # Set all points (Limited to intersection points and less than local minima) to the PET image. 
                        # IntersectionCoords[a,0] == c leftIntersection matrix is updated only when the x coord is the same as the Localminia x coord
    
"BOTTOM FILLING DIRECTION"
if Direction_Vector[1] > 0 and abs(Direction_Vector[0]) < 0.5: # Check unit vector is pointing up and between 0.5 horizontally

    "Average True values along y axis"
    for k in range(IntersectionCoordsXMin, IntersectionCoordsXMax):
        for d in range(IntersectionCoordsYMin, IntersectionCoordsYMax):
            Sum = 0
            if pathMask[d,k]:
                for a in range(IntersectionCoordsYMin, IntersectionCoordsYMax):
                    if pathMask[a,k]:
                        Sum +=1
                        pathMask[a,k] = False
                pathMask[d+int(Sum/2),k] = True
                
    "Uses vertical local minima matrix to seperate the intersection"
    for c in range(IntersectionCoordsXMin, IntersectionCoordsXMax):
        for d in range(IntersectionCoordsYMin, IntersectionCoordsYMax): 
        # Loop through intersection coordinates region
            if pathMask[d,c]:
            # Check for the LocalMinima points
                for a in range(IntersectionCoords.shape[0]):
                # Loop through every intersection coordinate
                    if IntersectionCoords[a,0] == c and IntersectionCoords[a,1] > d: #check it is the first local minima with that x coord
                        L2Intersection[IntersectionCoords[a,1], IntersectionCoords[a,0]] = True
                        # Set all points (Limited to intersection points and less than local minima) to the PET image. 
                        # IntersectionCoords[a,0] == c leftIntersection matrix is updated only when the x coord is the same as the Localminia x coord

plt.imshow(mask_L1 + mask_L2)
plt.imshow(mask_L2)
mask_L2 = mask_L2 ^ L2Intersection
L1Intersection = Intersection ^ L2Intersection
mask_L1 = mask_L1 ^ L1Intersection

mask = mask_L1
mask_spect2 = mask_L2
"-------------------------------------------------------------------------------------"

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
