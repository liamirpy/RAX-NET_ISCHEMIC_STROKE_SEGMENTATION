import pyvista as pv
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
# Load your 3D volume data (e.g., lesion mask)
# Example: lesion_mask_path = 'path/to/your/lesion_mask.nii.gz'






# nii_file = 'axial_before_fusion.nii.gz'
# nii_file = 'sagittal_before_fusion.nii.gz'
# nii_file = 'coronal_before_fusion.nii.gz'
# nii_file = 'mask.nii.gz'
# nii_file = 'segmentation_mask.nii.gz'






# nii_data = nib.load(nii_file)
# nii_img = nii_data.get_fdata()

# # Create a figure
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the NIfTI data
# x, y, z = nii_img.nonzero()
# ax.scatter(x, y, z, c=nii_img[x, y, z], alpha=0.5)

# # Customize the axes
# ax.set_xlim(0, nii_img.shape[0])
# ax.set_ylim(0, nii_img.shape[1])
# ax.set_zlim(0, nii_img.shape[2])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Scatter Plot of NIfTI Data')

# # Show the plot
# plt.show()


import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# List of NIfTI files
nii_files = ['axial_before_fusion.nii.gz', 'sagittal_before_fusion.nii.gz', 'coronal_before_fusion.nii.gz','mask.nii.gz', 'segmentation_mask.nii.gz']

# Create a figure with 5 subplots
plot_labels = [
    'Axial Plane Before Fusion',
    'Sagittal Plane Before Fusion',
    'Coronal Plane Before Fusion',
    'Ground Turth',
    'Axial, Sagittal,Coronal Plane After Fusion'
]

# Create a figure with a 2x3 grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(20, 10), subplot_kw={'projection': '3d'})

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Plot each NIfTI file in a different subplot
for i, nii_file in enumerate(nii_files):
    # Load the NIfTI file
    nii_data = nib.load(nii_file)
    nii_img = nii_data.get_fdata()
    
    # Select non-zero data points
    x, y, z = nii_img.nonzero()
    
    # Plot the NIfTI data
    axes[i].scatter(x, y, z, c=nii_img[x, y, z], alpha=0.5)
    
    # Customize the axes
    axes[i].set_xlim(0, nii_img.shape[0])
    axes[i].set_ylim(0, nii_img.shape[1])
    axes[i].set_zlim(0, nii_img.shape[2])
    axes[i].set_xlabel('X')
    axes[i].set_ylabel('Y')
    axes[i].set_zlabel('Z')
    axes[i].set_title(plot_labels[i])

# Remove the last empty subplot
fig.delaxes(axes[-1])

# Adjust layout
plt.tight_layout()
plt.savefig('Fusion_in_3D_view.png',dpi=400)
# plt.show()