# ATTENTION: The code in this file has been adapted from
# here: https://stackoverflow.com/questions/71567315/how-to-get-the-ssim-comparison-score-between-two-images

from skimage.metrics import structural_similarity
import cv2
import os
from skimage import io, color, measure
import statistics
import numpy as np
import lpips

# Define the paths to the two folders containing images
folder1 = '' # The folder containing the original Dataset
folder2 = '' # the folder containing the dataset with the inpainted images

# Get the list of image files in each folder
images1 = [f for f in os.listdir(folder1) if f.endswith('.png')]
images2 = [f for f in os.listdir(folder2) if f.endswith('.png')]

# Initialize variables for sum and count
total_ssim = 0
pair_count = 0
ssimList = []
jaccard_list = []
lpips_list = []
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
        (ssimScore, diff) = structural_similarity(image1_gray, image2_gray, win_size=71 ,full=True)
        print("Image similarity", ssimScore)

        # Calculate the IoU score
        # Calculate the intersection and union of the binary images
        intersection = np.logical_and(image1, image2)
        #print("Intersection: ",
        #      np.sum(intersection))
        union = np.logical_or(image1, image2)

        # Calculate the Jaccard Index (IoU)
        jaccard_index = np.sum(intersection) / np.sum(union)
        print("Jaccard Score: ", jaccard_index)
        # Append Jaccard Index to a list
        jaccard_list.append(jaccard_index)

        # Appens ssim score to list
        ssimList.append(ssimScore)

        # Update sum and count
        total_ssim += ssimScore
        pair_count += 1

        # convert the images to the LPIPS format
        image_1_lpips = lpips.im2tensor(lpips.load_image(image_path1))
        image_2_lpips = lpips.im2tensor(lpips.load_image(image_path2))

        # calculate LPIPS
        lpips_fn = lpips.LPIPS(net='alex')
        lpips_value = lpips_fn.forward(image_1_lpips, image_2_lpips)
        lpips_list.append(lpips_value[0, 0, 0, 0].item())
        print("LPIPS: ", lpips_value[0, 0, 0, 0].item())

    else:
        print("No image with the same name has been found!")
# Calculate the mean SSIM score and the mean jaccard-score
mean_ssim = total_ssim / pair_count if pair_count > 0 else 0
mean_jaccard = statistics.mean(jaccard_list)
mean_lpips = statistics.mean(lpips_list)


# Find the minimum and maximum values
maximum_ssim = max(ssimList)
minimum_ssim = min(ssimList)
maximum_jaccard = max(jaccard_list)
minimum_jaccard = min(jaccard_list)
maximum_lpips = max(lpips_list)
minimum_lpips = min(lpips_list)

# Calculateion of the standard deviation
std_deviation_ssim = statistics.stdev(ssimList)
std_deviation_jaccard = statistics.stdev(jaccard_list)
std_deviation_lpips = statistics.stdev(lpips_list)

# Calculation of the variance
variance_ssim = statistics.variance(ssimList)
variance_jaccard = statistics.variance(jaccard_list)
variance_lpips = statistics.variance(lpips_list)

# Calculation of the median value
median_value_ssim = statistics.median(ssimList)
median_value_jaccard = statistics.median(jaccard_list)
median_value_lpips = statistics.median(lpips_list)
print("////////////////////////////////////")
print("Mean SSIM Score: ", mean_ssim)
print("Maximal SSIM Score: ", maximum_ssim)
print("Minimal SSIM Score: ", minimum_ssim)
print("Standard Deviation: ", std_deviation_ssim)
print("Variance:", variance_ssim)
print("Median:", median_value_ssim)
print("////////////////////////////////////")
print("Mean Jaccard Score: ", mean_jaccard)
print("Maximal Jaccard Score: ", maximum_jaccard)
print("Minimal Jaccard Score Jaccard: ", minimum_jaccard)
print("Standard Deviation Jaccard: ", std_deviation_jaccard)
print("Variance Jaccard:", variance_jaccard)
print("Median Jaccard:", median_value_jaccard)
print("////////////////////////////////////")
print("Mean LPIPS: ", mean_lpips)
print("Maximal LPIPS: ", maximum_lpips)
print("Minimal LPIPS ", minimum_lpips)
print("Standard Deviation LPIPS: ", std_deviation_lpips)
print("Variance LPIPS:", variance_lpips)
print("Median LPIPS:", median_value_lpips)