import numpy as np


def load_data(folder_directory,train_mri,train_mask,validation_mri,validation_mask):
    
    train_data_mri = np.load(f'{folder_directory}/{train_mri}').astype(np.float32)
  
    train_data_mask=np.load(f'{folder_directory}/{train_mask}').astype(np.float32)



    validation_data_mri=np.load(f'{folder_directory}/{validation_mri}').astype(np.float32)


    validation_data_mask=np.load(f'{folder_directory}/{validation_mask}').astype(np.float32)


    return train_data_mri,train_data_mask,validation_data_mri,validation_data_mask




