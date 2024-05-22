### This is LIAMIRPY from the land of loneliness. 

## In this code we want to split the data from the original folder of data set to Test and Train folder 


'''
        Procedure:
    
    1- Split the data for 3D mode:

        - 20% for test data
        -80% for train data
    
    2-convert the train data from 3D to 2D for axial, sagittal and coronal


    3- Based on each fold put the 2D images of each plan to their folds



'''




import os



import csv

import pandas as pd 


import re 

import shutil

import numpy as np


import nibabel as nib


import os 





def convert_3D_to_2D(mri_folder,mask_folder,axial_folder,coronal_folder,sagittal_folder):



    t1_file_name_list = os.listdir(mri_folder)

    for t1 in t1_file_name_list:

        # print(t1)

        #### convert t1 to png image
        T1_nii_nii=nib.load( mri_folder +'/'+ t1 )
        T1_nii = T1_nii_nii.get_fdata()


        new_header= T1_nii_nii.header.copy()



        mask_name=t1.split('_T1w')[0] + '_label-L_desc-T1lesion_mask.nii.gz'

        mask_nii=nib.load( f'{mask_folder}/{mask_name}' )
        mask_nii = mask_nii.get_fdata()


        # ### make a directory for this T1
        t1_name=((t1.split('_ses-1'))[0])


        print(t1_name)


        t1_path = f'{axial_folder}/{t1_name}'

        os.makedirs(t1_path)





        for ax in range(T1_nii.shape[2]):


            single_t1_slice=T1_nii[:,:,ax]


            
            ni_img = nib.nifti1.Nifti1Image(single_t1_slice, None, header=new_header)

            nib.save(ni_img, f'{t1_path}/{t1_name}_slice_{ax}_T1.nii.gz')



            # cv2.imwrite(f'{axial_folder}/{t1_name}_slice_{ax}_T1.png',normal_data)




            single_mask_slice=mask_nii[:,:,ax]

            single_mask_slice[single_mask_slice > 0] = 1


     
            # cv2.imwrite(f'{t1_path}/{t1_name}_slice_{ax}_mask.png',single_mask_slice)



            ni_img = nib.nifti1.Nifti1Image(single_mask_slice, None, header=new_header)

            nib.save(ni_img, f'{t1_path}/{t1_name}_slice_{ax}_mask.nii.gz')





        t1_path = f'{sagittal_folder}/{t1_name}'

        os.makedirs(t1_path)





        for sa in range(T1_nii.shape[0]):


            single_t1_slice=T1_nii[sa,:,:]


            
            ni_img = nib.nifti1.Nifti1Image(single_t1_slice, None, header=new_header)

            nib.save(ni_img, f'{t1_path}/{t1_name}_slice_{sa}_T1.nii.gz')







            single_mask_slice=mask_nii[sa,:,:]

            single_mask_slice[single_mask_slice > 0] = 1


     



            ni_img = nib.nifti1.Nifti1Image(single_mask_slice, None, header=new_header)

            nib.save(ni_img, f'{t1_path}/{t1_name}_slice_{sa}_mask.nii.gz')











        t1_path = f'{coronal_folder}/{t1_name}'

        os.makedirs(t1_path)





        for co in range(T1_nii.shape[1]):


            single_t1_slice=T1_nii[:,co,:]


            
            ni_img = nib.nifti1.Nifti1Image(single_t1_slice, None, header=new_header)

            nib.save(ni_img, f'{t1_path}/{t1_name}_slice_{co}_T1.nii.gz')







            single_mask_slice=mask_nii[:,co,:]

            single_mask_slice[single_mask_slice > 0] = 1


     



            ni_img = nib.nifti1.Nifti1Image(single_mask_slice, None, header=new_header)

            nib.save(ni_img, f'{t1_path}/{t1_name}_slice_{co}_mask.nii.gz')






















Train_folder_mri='Train_3D_Data/mri'




Train_folder_mask='Train_3D_Data/mask'




axial_folder='./axial_2D'
coronal_folder='./coronal_2D'
sagittal_folder='./sagittal_2D'


convert_3D_to_2D(Train_folder_mri,Train_folder_mask,axial_folder,coronal_folder,sagittal_folder)


    


