# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:52 2024
@author: Elijah Gardi
"""

import nibabel as nib
from totalsegmentator.python_api import totalsegmentator

if __name__ == "__main__":
    
    input_path = r"E:\Users\elija\OneDrive - Queensland University of Technology\PROJECT\Practice data\SPECT_Cts\scan1\ct"
    output_path = r"E:\Users\elija\OneDrive - Queensland University of Technology\PROJECT\Practice data\Output data"
    
    # option 1: provide input and output as file paths
    totalsegmentator(input_path, output_path, roi_subset=['kidney_right','kidney_left','spleen','liver'])
    