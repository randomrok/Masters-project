# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 20:59:26 2024

@author: Elijah Gardi
"""

import nibabel as nib

import numpy as np

from numpy import ndarray

from scipy import ndimage

import scipy.io

from os import path

from totalsegmentator.python_api import totalsegmentator

from scipy.ndimage import binary_dilation, binary_closing

import sys


def OpenSpectFiles(file_number):
    
    file_base = 'spect_petvp{}.mat'.format(file_number)
    
    folder_base = 'petvp{}'.format(file_number)
        
    mat = scipy.io.loadmat(path.join(file_r, folder_base, file_base))
    
    data = mat.get('x')
    
    # Convert to HU if file is CT
    
    if 'density' in file_base:
        data_HU = 1000*(data.astype(float)-1000)/1000
    else: 
        data_HU = data
        
    print('file:', file_base)
    print("data shape:", data.shape)
    print("data max:", data.max())
    
    # Assuming your 3D numpy array is named 'data'
    affine = np.array([ [-1., 0., 0., 0.],
                        [0., -1., 0., 0.],
                        [0., 0., -3., 0.],
                        [0., 0., 0., -1.]])
    
    # Create a Nifti image object
    nifti_img = nib.Nifti1Image(data_HU, affine = affine) # Using an identity affine sutrix
    
    # Specify the filename
    Filename = path.join(file_r, folder_base, 'SPECT p{}.nii.gz'.format(file_number))
    
    #Save the Nifti image to a file
    nib.save(nifti_img, Filename)
    
    return data_HU

def OpenCTFiles(file_number):
        
    file_base = 'density_petvp{}.mat'.format(file_number)
    
    folder_base = 'petvp{}'.format(file_number)
    
    if(path.exists(path.join(file_r, folder_base, file_base))):
    
        mat = scipy.io.loadmat(path.join(file_r, folder_base, file_base))
        
        data = mat.get('x')
        
        # Convert to HU if file is CT
        
        if 'density' in file_base:
            data_HU = 1000*(data.astype(float) - 1000)/1000
        else: 
            data_HU = data
            
        print('file:', file_base)
        print("data shape:", data.shape)
        print("data max:", data.max())
        
        # Assuming your 3D numpy array is named 'data'
        affine = np.array([ [-1., 0., 0., 0.],
                            [0., -1., 0., 0.],
                            [0., 0., -3., 0.],
                            [0., 0., 0., -1.]])
        
        # Create a Nifti image object
        nifti_img = nib.Nifti1Image(data_HU, affine = affine) # Using an identity affine sutrix
        
        # Specify the filename
        Filename = path.join(file_r, folder_base, 'CT p{}.nii.gz'.format(file_number))
        
        #Save the Nifti image to a file
        nib.save(nifti_img, Filename)
        
    return

def TotalSegmentator(file_number):
        
    file_base = 'CT p{}.nii.gz'.format(file_number)
    
    folder_base = 'petvp{}'.format(file_number)
        
    if(path.exists(path.join(file_r, folder_base, file_base))):
        
        input_path = path.join(file_r, folder_base, file_base)
        
        output_path = path.join(file_r, folder_base)
        
        totalsegmentator(input_path, output_path, roi_subset=['kidney_right', 'kidney_left', 'spleen', 'liver'])
        
    return

def spect_mask_inhouse (file_number):
    '''
    Current in-house method
    
    - first input is single left or right kidney mask [numpy ndarray type] 
    - second input is the adjacent organ mask (ie. spleen or liver) [numpy ndarray type]
    '''
    file_number = 1
    folder_base = 'petvp{}'.format(file_number)
    
    # load left kidney, then convert to numpy array
    
    lkidney_mask = nib.load(path.join(file_r, folder_base, 'kidney_left.nii.gz'))
    lkidney_mask = np.array(lkidney_mask.dataobj)
    
    # load right kidney, then convert to numpy array
    
    rkidney_mask = nib.load(path.join(file_r, folder_base, 'kidney_right.nii.gz'))
    rkidney_mask = np.array(rkidney_mask.dataobj)
    
    # load liver, then convert to numpy array
    
    liver_mask = nib.load(path.join(file_r, folder_base, 'liver.nii.gz'))
    liver_mask = np.array(liver_mask.dataobj)
    
    # load spleen, then convert to numpy array
    
    spleen_mask = nib.load(path.join(file_r, folder_base, 'spleen.nii.gz'))
    spleen_mask = np.array(spleen_mask.dataobj)

    #Expands the left kidney
    
    expanded_kidney_mask = binary_dilation(lkidney_mask, iterations=10)
    
    #Remove holes
    
    expanded_kidney_mask = binary_closing(expanded_kidney_mask, iterations=10)
    
    #Expands the spleen
    
    expanded_adjacent_mask = binary_dilation(spleen_mask, iterations=3)
    
    #Apply mask operation subtract kidney from expanded spleen/Liver
    
    lmask = np.bitwise_xor(expanded_kidney_mask, np.bitwise_and(expanded_kidney_mask, expanded_adjacent_mask))
    
    
    #Expands the right kidney
    
    expanded_kidney_mask = binary_dilation(rkidney_mask, iterations=10)
    
    #Remove holes
    
    expanded_kidney_mask = binary_closing(expanded_kidney_mask, iterations=10)
    
    #Expands the liver
    
    expanded_adjacent_mask = binary_dilation(liver_mask, iterations=3)
    
    #Apply mask operation subtract kidney from expanded spleen/Liver
    
    rmask = np.bitwise_xor(expanded_kidney_mask, np.bitwise_and(expanded_kidney_mask, expanded_adjacent_mask))
    
    mask = lmask + rmask
    
    return mask

def spect_mask_inhouse_tunable(file_number, 
                               kidney_exp_iter = 8, 
                               adjacent_exp_iter = 2):
    '''
    Tunable in-house method
    
    first input is single left or right kidney mask [numpy ndarray type]
    
    second input is the adjacent organ mask (ie. spleen or liver) [numpy ndarray type]
    '''
    
    folder_base = 'petvp{}'.format(file_number)
    
    # load left kidney, then convert to numpy array
    
    lkidney_mask = nib.load(path.join(file_r, folder_base, 'kidney_left.nii.gz'))
    lkidney_mask = np.array(lkidney_mask.dataobj)
    
    # load right kidney, then convert to numpy array
    
    rkidney_mask = nib.load(path.join(file_r, folder_base, 'kidney_right.nii.gz'))
    rkidney_mask = np.array(rkidney_mask.dataobj)
    
    # load liver, then convert to numpy array
    
    liver_mask = nib.load(path.join(file_r, folder_base, 'liver.nii.gz'))
    liver_mask = np.array(liver_mask.dataobj)
    
    # load spleen, then convert to numpy array
    
    spleen_mask = nib.load(path.join(file_r, folder_base, 'spleen.nii.gz'))
    spleen_mask = np.array(spleen_mask.dataobj)

    #Expands the left kidney
    
    expanded_kidney_mask = binary_dilation(lkidney_mask, iterations=kidney_exp_iter)
    
    #Remove holes
    
    expanded_kidney_mask = binary_closing(expanded_kidney_mask, iterations=10)
    
    #Expands the spleen
    
    expanded_adjacent_mask = binary_dilation(spleen_mask, iterations=adjacent_exp_iter)
    
    #Apply mask operation subtract kidney from expanded spleen/Liver
    
    lmask = np.bitwise_xor(expanded_kidney_mask, np.bitwise_and(expanded_kidney_mask, expanded_adjacent_mask))
    
    
    
    #Expands the right kidney
    
    expanded_kidney_mask = binary_dilation(rkidney_mask, iterations=kidney_exp_iter)
    
    #Remove holes
    
    expanded_kidney_mask = binary_closing(expanded_kidney_mask, iterations=10)
    
    #Expands the liver
    
    expanded_adjacent_mask = binary_dilation(liver_mask, iterations=adjacent_exp_iter)
    
    #Apply mask operation subtract kidney from expanded spleen/Liver
    
    rmask = np.bitwise_xor(expanded_kidney_mask, np.bitwise_and(expanded_kidney_mask, expanded_adjacent_mask))
    
    mask = lmask + rmask
    
    return mask

'-------------------------------------------'
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
'-------------------------------------------'


# Custom segmentation code utilising simple watershed function and path finding function above^

# Code essentially uses the watershed function to minus a specified stepsize from all pixels until
# there is a path between the smallest coord and the largest coord. The resulting image is converted 
# to a boolean and is called the pathmatrix.
# Then, it determines the direction from the center of the first mask to the centroid of the intersection 
# and fills the intersection with True values. The True values stop at half the distance between the furthest pathmatrix True value.

# This could be better implemented by using a function to average the values in the matrix in the direction 
# of the first mask to the centroid of the intersection, instead of vertically or horizontally.

"-------------------------------------------------------------------------------------"
def CustomSegmentation(kidney, OverlapOrgan, PET_Image):
    
    '''
    first input is single left or right kidney mask [numpy ndarray type]
    
    second input is the adjacent organ mask (ie. spleen or liver) [numpy ndarray type]
    '''
    
    # Minus stepsize from all pixels intensity unless zero
    def StepMinus(matrix, stepsize):
        for a in range (matrix.shape[0]):
            for b in range (matrix.shape[1]): # loop through matrix
                
                if matrix[a,b] > stepsize:
                    matrix[a,b] = matrix[a,b] - stepsize
                
                if matrix[a,b] <= stepsize: 
                    matrix[a,b] = 0
        return matrix
    
    # uses path finder and stepminus to subtract until there is a path, speed is inverse to stepsize and accuracy
    def Watershed(image):
        stepsize = np.amax(image*Intersection)/10
        imageMatrix = image * Intersection + ((np.ones(image.shape, dtype=bool) ^ Intersection)*(np.amax(image*Intersection)+2*stepsize)) # set equal to ones everywhere outside intersection
        PathMatrix = ndarray.copy(imageMatrix)
        pathMask = np.zeros(image.shape, dtype=bool)
        
        for i in range(10):
            imageMatrix = StepMinus(imageMatrix, stepsize) # Subtract 'stepsize' from all pixels
            PathMatrix = ndarray.copy(imageMatrix) # Matrix for checking path
            
            "Convert to isPath format"
            for a in range(PathMatrix.shape[0]):
                for b in range(PathMatrix.shape[1]): # Loop through all pixels
                    if PathMatrix[a,b] == 0:
                        PathMatrix[a,b] = 3 # Path cell
                    else: 
                        PathMatrix[a,b] = 0 # Blocked cell
                        
            
            "SET START AND FINISH POINTS"
            "top and bottom directions"
            if abs(Direction_Vector[0]) < 0.7071 and abs(Direction_Vector[1]) > 0.7071:
                "Start and end points"
                x_start, y_start = min(IntersectionCoords, key=lambda x: (x[0], -x[1]))
                x_end, y_end = max(IntersectionCoords, key=lambda x: (x[0], -x[1]))
                "Start and end pixels"
                PathMatrix[y_start, x_start] = 1 # 'Starting cell'
                PathMatrix[y_end, x_end] = 2 # 'Destination cell'
                
            "left and right directions"
            if abs(Direction_Vector[0]) > 0.7071 and abs(Direction_Vector[1]) < 0.7071:
                "Start and end points"
                y_start = min(IntersectionCoords[:,1])
                x_start = IntersectionCoords[np.where(IntersectionCoords[:,1] == min(IntersectionCoords[:,1]))[0], 0][0]
                y_end = max(IntersectionCoords[:,1])
                x_end = IntersectionCoords[np.where(IntersectionCoords[:,1] == max(IntersectionCoords[:,1]))[0], 0][0]
                "Start and end pixels"
                PathMatrix[y_start, x_start] = 1 # 'Starting cell'
                PathMatrix[y_end, x_end] = 2 # 'Destination cell'
            
        # If there is a path, then loop through the image and create a mask from the zero values
        if isPath(PathMatrix, PathMatrix.shape[0]):
            for a in range(PET_Image.shape[0]):
                for b in range(PET_Image.shape[1]):
                    if imageMatrix[a,b] == 0:
                        pathMask[a,b] = True
                    else: pathMask[a,b] = False
            return pathMask
        return Intersection
    
    # Obtains centroid of coordinates given
    def GetCentroid(mask):
        "Origin is the average of all points from mask"
        x = [p[0] for p in mask]
        y = [p[1] for p in mask]
        centroid = (sum(x) / len(mask), sum(y) / len(mask))
        return centroid
    
    # Returns the direction vector for two input positions
    def Check_Direction(Centroid, c, d):
        # Check for the LocalMinima points "Use local minima coordinate with the direction from the origin of the mask to determine left and right/top and down filling direction"
        Direction_Vector = np.subtract(Centroid, np.array([c,d]))
        Magnitude = np.linalg.norm(Direction_Vector)
        Direction_Vector = Direction_Vector/Magnitude
        return Direction_Vector
    
    # Fills intersection according to the direction vector
    def FillIntersection():
        L2Intersection = np.zeros(pathMask.shape, dtype = bool)
        
        "Intersection boundries for speed"
        IntersectionCoordsXMax = max(IntersectionCoords[:,0])
        IntersectionCoordsXMin = min(IntersectionCoords[:,0])
        IntersectionCoordsYMax = max(IntersectionCoords[:,1])
        IntersectionCoordsYMin = min(IntersectionCoords[:,1])
        
        "LEFT FILLING DIRECTION"
        if Direction_Vector[0] < 0 and abs(Direction_Vector[1]) < 0.7071: # Check unit vector is to left and between 45 degrees (0.7071) vertically
            
            "Average True values of path mask along the X axis"
            for d in range(IntersectionCoordsYMin, IntersectionCoordsYMax):
                for k in range(IntersectionCoordsXMin, IntersectionCoordsXMax):
                    Sum = 0
                    if pathMask[d,k]:
                        for a in range(IntersectionCoordsXMin, IntersectionCoordsXMax):
                            if pathMask[d,a]:
                                Sum +=1
                                pathMask[d,a] = False
                        pathMask[d,k+int(Sum/2)] = True
        
            "Uses path matrix to fill the intersection"
            for c in range(IntersectionCoordsXMin, IntersectionCoordsXMax):
                for d in range(IntersectionCoordsYMin, IntersectionCoordsYMax):# Loop through intersection coordinates region
                    if pathMask[d,c]: # Check for the LocalMinima points
                        for a in range(IntersectionCoords.shape[0]): # Loop through every intersection coordinate
                            if IntersectionCoords[a,1] == d and IntersectionCoords[a,0] < c: # Check that the intersection coordinate is not greater than the LocalMinima coordinate
                                L2Intersection[IntersectionCoords[a,1], IntersectionCoords[a,0]] = True
                                # Set all points (Limited to intersection points and less than local minima) to the PET image. 
                                # IntersectionCoords[a,1] == d leftIntersection matrix is updated only when the y coord is the same as the Localminia y coord
         
        "RIGHT FILLING DIRECTION"
        if Direction_Vector[0] > 0 and abs(Direction_Vector[1]) < 0.7071: # Check unit vector is to right and between 45 degrees (0.7071) vertically
                            
            "Average True values of path mask along the X axis"
            for d in range(IntersectionCoordsYMin, IntersectionCoordsYMax):
                for k in range(IntersectionCoordsXMin, IntersectionCoordsXMax):
                    Sum = 0
                    if pathMask[d,k]:
                        for a in range(IntersectionCoordsXMin, IntersectionCoordsXMax):
                            if pathMask[d,a]:
                                Sum +=1
                                pathMask[d,a] = False
                        pathMask[d,k-int(Sum/2)] = True
                        
            "Uses path matrix to fill the intersection"
            for c in range(IntersectionCoordsXMin, IntersectionCoordsXMax):
                for d in range(IntersectionCoordsYMin, IntersectionCoordsYMax): # Loop through intersection coordinates region
                    if pathMask[d,c]: # Check for the LocalMinima points
                        for a in range(IntersectionCoords.shape[0]): # Loop through every intersection coordinate
                            if IntersectionCoords[a,1] == d and IntersectionCoords[a,0] > c: # Check that the intersection coordinate is not greater than the LocalMinima coordinate
                                L2Intersection[IntersectionCoords[a,1], IntersectionCoords[a,0]] = True
                                # Set all points (Limited to intersection points and less than local minima) to the PET image. 
                                # IntersectionCoords[a,1] == d leftIntersection matrix is updated only when the y coord is the same as the Localminia y coord
                
        "TOP FILLING DIRECTION"
        if (Direction_Vector[1] < 0 and abs(Direction_Vector[0]) < 0.7071): # Check unit vector is pointing up and between 45 degrees (0.7071) horizontally
                        
            "Average True values of path mask along the Y axis"
            for d in range(IntersectionCoordsYMin, IntersectionCoordsYMax):
                for k in range(IntersectionCoordsXMin, IntersectionCoordsXMax):
                    Sum = 0
                    if pathMask[d,k]:
                        for a in range(IntersectionCoordsYMin, IntersectionCoordsYMax):
                            if pathMask[a,k]:
                                Sum +=1
                                pathMask[a,k] = False
                        pathMask[d-int(Sum/2),k] = True
        
            "Uses path matrix to fill the intersection"
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
        if Direction_Vector[1] > 0 and abs(Direction_Vector[0]) < 0.7071: # Check unit vector is pointing up and between 45 degrees (0.7071) horizontally
        
            "Average True values of path mask along the Y axis"
            for d in range(IntersectionCoordsYMin, IntersectionCoordsYMax):
                for k in range(IntersectionCoordsXMin, IntersectionCoordsXMax):
                    Sum = 0
                    if pathMask[d,k]:
                        for a in range(IntersectionCoordsYMin, IntersectionCoordsYMax):
                            if pathMask[a,k]:
                                Sum +=1
                                pathMask[a,k] = False
                        pathMask[d+int(Sum/2),k] = True                
        
            "Uses path matrix to fill the intersection"
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
        return L2Intersection

    # Checks for true value in the kidney matrix before beginning any processing. This would be false if theres no kidney segmentation in the section of the CT image.
    if np.any(kidney):
    
        # High recursion limit for this process from path finding code
        sys.setrecursionlimit(1000000)
        
        "Variable declerations"
        mask_L1 = np.array(kidney, dtype=bool)
        mask_L2 = np.array(OverlapOrgan, dtype=bool)
        
        Intersection = np.array(mask_L1 ^ mask_L2, dtype=bool)
        Intersection = np.array(Intersection ^ (mask_L1 + mask_L2), dtype=bool)
        
        "ONLY COMPUTE INTERSECTION SEGMENTATION WHEN THERE IS AN INTERSECTION (TRUE VALUE)"
        if np.any(Intersection):
            "Obtains coordinates of interefring mask and intersection"
            rows, cols = np.where(mask_L2) 
            maskL2coordinates = np.column_stack((cols, rows))
            
            rows, cols = np.where(mask_L1) 
            maskL1coordinates = np.column_stack((cols, rows))
            
            rows, cols = np.where(Intersection)
            IntersectionCoords = np.column_stack((cols, rows))
            
            "Calculate centroid of secondary mask and intersection for direction"
            centroid = GetCentroid(maskL2coordinates)
            Centroid = GetCentroid(maskL1coordinates)
            
            "Create Direction (unit) Vector"
            Direction_Vector = Check_Direction(Centroid, centroid[0], centroid[1])
            
            "Find the image/matrix with a path through it converted to boolean"
            pathMask = Watershed(PET_Image)
            
            "FILL INTERSECTION UP TO THE SHORTEST PATH THROUGH PET IMAGE MATRIX"
            L2Intersection = FillIntersection()
            
            "Compute which side of the seperated intersection belongs to which organ"
            mask_L2 = mask_L2 ^ L2Intersection
            L1Intersection = Intersection ^ L2Intersection
            mask_L1 = mask_L1 ^ L1Intersection
            
    else: mask_L1 = kidney
    
    return mask_L1
    
def ComputeCustom(file_number):
    
    "OPEN FILES AND CONVERT TO NUMPY ARRAYS"
    folder_base = 'petvp{}'.format(file_number)
    
    # load left kidney, then convert to numpy array
    
    lkidney_mask = nib.load(path.join(file_r, folder_base, 'kidney_left.nii.gz'))
    lkidney_mask = np.array(lkidney_mask.dataobj)
    
    # load right kidney, then convert to numpy array
    
    rkidney_mask = nib.load(path.join(file_r, folder_base, 'kidney_right.nii.gz'))
    rkidney_mask = np.array(rkidney_mask.dataobj)
    
    # load liver, then convert to numpy array
    
    liver_mask = nib.load(path.join(file_r, folder_base, 'liver.nii.gz'))
    liver_mask = np.array(liver_mask.dataobj)
    
    # load spleen, then convert to numpy array
    
    spleen_mask = nib.load(path.join(file_r, folder_base, 'spleen.nii.gz'))
    spleen_mask = np.array(spleen_mask.dataobj)
    
    "EXPAND ALL MASKS TO HAVE OVERLAPPING REGIONS"
    #Expands the left kidney
    lkidney_mask = np.uint8(binary_dilation(lkidney_mask, iterations=8))
    
    #Remove holes
    lkidney_mask = np.uint8(binary_closing(lkidney_mask, iterations=8))
    
    #Expands the spleen
    spleen_mask = np.uint8(binary_dilation(spleen_mask, iterations=3))
    
    #Expands the right kidney
    rkidney_mask = np.uint8(binary_dilation(rkidney_mask, iterations=8))
    
    #Remove holes
    rkidney_mask = np.uint8(binary_closing(rkidney_mask, iterations=8))
    
    #Expands the liver
    liver_mask = np.uint8(binary_dilation(liver_mask, iterations=3))
    
    # Creates empty lists to store the sections of the segmentations ie. [:,:,0] ... [:,:,130]
    mask1 = [0] * lkidney_mask.shape[2]
    mask2 = [0] * lkidney_mask.shape[2]
    
    PET = OpenSpectFiles(file_number)
    
    # Loops through each section of the image to apply the watershed and pathfinding segmentation (except where there is no intersection)
    for w in range(0, lkidney_mask.shape[2]):
        PET_Image = ndarray.copy(PET[:,:,w])
        
        lkidney = lkidney_mask[:,:,w]
        rkidney = rkidney_mask[:,:,w]
        
        spleen = spleen_mask[:,:,w]
        liver = liver_mask[:,:,w]
        
        mask1[w] = CustomSegmentation(lkidney, spleen, PET_Image)
        mask2[w] = CustomSegmentation(rkidney, liver, PET_Image)
        
    kidneys = np.array(mask1) + np.array(mask2)
    kidneys = np.transpose(kidneys, (1, 2, 0))
    return kidneys

if __name__ == '__main__':
    
    file_r = input("Enter your SPECT/PET path: ")
    NumberofFiles = int(input("Enter the number of files: "))
    
    # Empty lists corresponding to methods of project
    SpectFiles = [0] * NumberofFiles
    TotalSegmentatorData = [0] * NumberofFiles
    Inhouse_kidney_counts = [0] * NumberofFiles
    Inhouse_kidney_tunable_counts = [0] * NumberofFiles
    Custom_kidney_counts = [0] * NumberofFiles
    
    for i in range(1, NumberofFiles+1):
        # Create file opening prefixes
        file_base = 'spect_petvp{}.mat'.format(i)
        folder_base = 'petvp{}'.format(i)
        
        if(path.exists(path.join(file_r, folder_base, file_base))):
            
            # Updates list with spect files from multiple patients
            SpectFiles[i-1] = OpenSpectFiles(i)
            
            "OPENS CT FILES AND CREATES AI SEGMENTATIONS"
            OpenCTFiles(i)
            TotalSegmentator(i)
            kidneys = nib.load(path.join(file_r, folder_base, 'kidney_left.nii.gz')) + nib.load(path.join(file_r, folder_base, 'kidney_right.nii.gz'))
            TotalSegmentatorData[i] = np.sum(kidneys * SpectFiles[i-1])
            
            "METHOD 1 INHOUSE"
            kidneys_mask1 = spect_mask_inhouse(i)
            kidneys_mask2 = spect_mask_inhouse_tunable(i,8,2)
            
            # Updates list with data from each patients inhouse segmentation
            Inhouse_kidney_counts[i-1] = np.sum(kidneys_mask1 * SpectFiles[i-1])
            Inhouse_kidney_tunable_counts[i-1] = np.sum(kidneys_mask2 * SpectFiles[i-1])
            
            "CUSTOM SEGMENTATION"
            # Computes custom segmentation edit based on watershed and path finding
            kidneys = ComputeCustom(i)
            
            # Updates list with each patients data from custom segmentation
            Custom_kidney_counts[i-1] = np.sum(kidneys * SpectFiles[i-1])
            
        '''
        plt.imshow(PET_Image)
        plt.imshow(kidney)
        plt.imshow(OverlapOrgan)
        plt.imshow(kidney + OverlapOrgan)
        plt.imshow(Intersection)
        plt.imshow(pathMask)
        plt.imshow(imageMatrix)
        plt.imshow(mask_L1)
        '''