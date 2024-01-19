# ATTENTION: The code in this file has been adapted from
# here: https://stackoverflow.com/questions/71567315/how-to-get-the-ssim-comparison-score-between-two-images

from skimage.metrics import structural_similarity
import cv2
import numpy as np
import os
from skimage import io, color, measure
import statistics

# Define the paths to the two folders containing images
folder1 = '/home/photolap/Documents/Masterarbeit_Froech/SSM_Experiments/CMP_Original_Resampled_000_rem' # The folder containing the original Dataset
folder2 = '/home/photolap/Documents/Masterarbeit_Froech/Archiv/Experiments_Lama/CMP/output_lama_inference' # the folder containing the dataset with the inpainted images

# Get the list of image files in each folder
images1 = [f for f in os.listdir(folder1) if f.endswith('.png')]
images2 = [f for f in os.listdir(folder2) if f.endswith('.png')]

# Initialize variables for sum and count
total_ssim = 0
pair_count = 0
ssimList = []
# Iterate through images with the same names and calculate SSIM
for image_name in images1:
    print("Image name: ", image_name)
    if image_name in images2:
        print("Identical name available")
        image_path1 = os.path.join(folder1, image_name)
        image_path2 = os.path.join(folder2, image_name)

        # Load and convert images to grayscale (optional)
        image1 = io.imread(image_path1)
        image2 = io.imread(image_path2)
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Calculate SSIM
        (ssimScore, diff) = structural_similarity(image1_gray, image2_gray, full=True)
        print("Image similarity", ssimScore)

        # Appens ssim score to list
        ssimList.append(ssimScore)

        # Update sum and count
        total_ssim += ssimScore
        pair_count += 1
    else:
        print("No image with the same name has been found!")
# Calculate the mean SSIM score
mean_ssim = total_ssim / pair_count if pair_count > 0 else 0

# Find the minimum and maximum values
maximum_ssim = max(ssimList)
minimum_ssim = min(ssimList)

# Calculateion of the standard deviation
std_deviation = statistics.stdev(ssimList)

# Calculation of the variance
variance = statistics.variance(ssimList)

# Calculation of the median value
median_value = statistics.median(ssimList)

print("Mean SSIM Score: ", mean_ssim)
print("Maximal SSIM Score: ", maximum_ssim)
print("Minimal SSIM Score: ", minimum_ssim)
print("Standard Deviation: ", std_deviation)
print("Variance:", variance)
print("Median:", median_value)