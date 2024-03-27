# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:41:28 2024
@author: Steven Goodman and Elijah Gardi

"""

import SimpleITK as sitk
import os
from matplotlib import pyplot as plt

segments_path = r"E:\Users\elija\OneDrive - Queensland University of Technology\PROJECT\Practice data\Output data"
spect_path = r"E:\Users\elija\OneDrive - Queensland University of Technology\PROJECT\Practice data\SPECT_Cts\scan1\spect"

left_kidney_itk = sitk.ReadImage(os.path.join(segments_path, 'kidney_left.nii.gz'))
right_kidney_itk = sitk.ReadImage(os.path.join(segments_path, 'kidney_right.nii.gz'))
spleen_itk = sitk.ReadImage(os.path.join(segments_path, 'spleen.nii.gz'))
liver_itk_ = sitk.ReadImage(os.path.join(segments_path, 'liver.nii.gz'))

left_kidney = sitk.GetArrayFromImage(left_kidney_itk)
right_kidney = sitk.GetArrayFromImage(right_kidney_itk)
spleen = sitk.GetArrayFromImage(spleen_itk)


combo_roi = left_kidney + right_kidney
combo_roi_itk = sitk.GetImageFromArray(combo_roi)

combo_roi_itk.SetDirection(left_kidney_itk.GetDirection())
combo_roi_itk.SetOrigin(left_kidney_itk.GetOrigin())
combo_roi_itk.SetSpacing(left_kidney_itk.GetSpacing())

'''Read Quant SPECT image'''

reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(spect_path)
reader.SetFileNames(dicom_names)
spect_itk = reader.Execute()

spect_array = sitk.GetArrayFromImage(spect_itk)

resample = sitk.ResampleImageFilter()
resample.SetReferenceImage(spect_itk)
resample.SetInterpolator(sitk.sitkNearestNeighbor)
resampled_combo_itk = resample.Execute(combo_roi_itk)

resampled_combo_array = sitk.GetArrayFromImage(resampled_combo_itk)


combo_masked = spect_array*resampled_combo_array


plt.figure()
plt.imshow(combo_masked[:,140,:])
plt.imshow(spect_array[140])

plt.imshow(resampled_combo_array[:,140,:], alpha=0.5)
