import cv2
import numpy as np
import statistics
import json
import sys
from scipy.signal import find_peaks
from sys import path
from Symmetry_Evaluation import compare_axis_symmetry
import matplotlib.pyplot as plt

plt_params = {"font.size": 12, "figure.autolayout": True, "xtick.top": True, "ytick.right": True, "ytick.left": True,
              "xtick.bottom": True, "xtick.direction": "in", "ytick.direction": "in"}
plt.rcParams.update(plt_params)

# A simple function to load parameters from a configuration object, given as a .json file
def load_config(filename):
    try:
        with open(filename, 'r') as config_file:
            config = json.load(config_file)
        return config
    # Exception handling
    except FileNotFoundError:
        print(f"The configuration file{filename} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error reading the configuration file {filename}.")
        return None

# a function to identify the most prominent color in an image
def detect_dominant_color(image_path):
    # loading the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Counting the number of black and white pixels
    total_pixels = image.size
    white_pixels = cv2.countNonZero(image)
    black_pixels = total_pixels - white_pixels

    # setting the return value in accordance to the results of the counting
    if white_pixels > black_pixels:
        return ["white", "black"]
    else:
        return ["black", "white"]

# a function to analyze the fragmentation in an image by identifying the contours anc comparing their mean surface area
# to the overall area in the image
def analyze_fragmentation(image_path, resample_size_x=1000, resample_size_y=1000):
    # Loading the image
    original_image = cv2.imread(image_path)

    # resample the original image to a larger number of pixels
    resampled_image = cv2.resize(original_image, (resample_size_x, resample_size_y))

    # convert the image into the right format
    resampled_image_gray = cv2.cvtColor(resampled_image, cv2.COLOR_BGR2GRAY)

    # finding a threshold
    _, binary_image = cv2.threshold(resampled_image_gray, 128, 255, cv2.THRESH_BINARY)

    # identifying the contours in the image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # drawing the contours
    #cv2.drawContours(resampled_image, contours, -1, (0, 0, 255), 2)

    # Calculating the area of the contours
    areas = [cv2.contourArea(contour) for contour in contours]

    # finding the largest area
    mean_area = statistics.mean(areas)
    #print("Largest area: ", mean_area)

    # calculating the total area
    total_area = np.sum(areas)
    #print("Total area: ", total_area)

    # Calculating a fragmentation score
    fragmentation_score = mean_area / total_area

    #print("Grad der Fragmentierung:", fragmentation_score)

    return fragmentation_score

# The main function of this file. it calls all the other functions in order to create a text prompt that is taylored
# specifically to the input image (patches)
def build_text_prompt(image_path, config_path):
    # loading the configuration file
    config = load_config(config_path)

    if config:
        resample_size_x = config.get('resample_size_x')
        resample_size_y = config.get('resample_size_y')
        threshold_fragmentation = config.get('threshold_fragmentation')
        threshold_symmetry = config.get('threshold_symmetry')
        num_angles = config.get("num_angles")
        unique_identifier = config.get("unique_identifier")
    else:
        resample_size_x = None
        resample_size_y = None
        threshold_fragmentation = None
        threshold_symmetry = None
        num_angles = None
        unique_identifier = None

    # obtaining the background colour
    colours = detect_dominant_color(image_path)
    #print("Dominant colour: ", colours[0])

    # obtaining the fragmentation score
    fragmentation_score = analyze_fragmentation(image_path, resample_size_x=resample_size_x, resample_size_y=resample_size_y)
    print("Fragmentation score: ", fragmentation_score)

    # defining the size of the patches
    if fragmentation_score < threshold_fragmentation:
        size = "small"
    else:
        size = "large"

    # identifying symmetry properties in the image
    ssim_indices = compare_axis_symmetry(image_path, num_angles=num_angles)
    # Calculating the statistical parameters
    median_value = statistics.median(ssim_indices)
    mean_value = statistics.mean(ssim_indices)
    standard_deviation = statistics.stdev(ssim_indices)
    variance = statistics.variance(ssim_indices)
    min_value = min(ssim_indices)
    max_value = max(ssim_indices)

    # print("Mean Value: ", mean_value)
    #print("Median Value: ", median_value)
    #print("Standard Deviation: ", standard_deviation)
    #print("Variance: ", variance)
    #print("Minimal Value: ", min_value)
    #print("Maximal Value: ", max_value)

    # Defining the threshold
    if max_value > float(threshold_symmetry):
        peak_threshold = mean_value + 1 * standard_deviation
    else:
        peak_threshold = mean_value + 3 * standard_deviation

    # indentifying the peaks
    peaks, _ = find_peaks(ssim_indices, height=peak_threshold)
    #print("Peaks: ", peaks)
    #print("Value: ", ssim_indices[peaks[0]])

    # Plot SSIM score vs. rotation angle
    #rotation_angles = np.arange(0, 181)

    #fig, ax = plt.subplots(figsize=(5,5))

    #ax.plot(rotation_angles, ssim_indices, marker='x', markersize=2, color='blue')
    #ax.plot(rotation_angles[peaks], np.array(ssim_indices)[peaks], 'ro')  # Plotting peaks as red dots
    #ax.axhline((mean_value + standard_deviation), color='red', linestyle='--',
    #            label='Standard deviation')  # Adding the standard deviation as a horizontal line
    #ax.axhline((mean_value - standard_deviation), color='red', linestyle='--',
    #            )
    #plt.axhline(peak_threshold, color='red', linestyle='--',
    #            label='Decision threshold')  # Adding decision threshold as a horizontal line
    #ax.axhline(mean_value, color='green', linestyle='--',
    #            label='Mean value')  # Adding mean value as a horizontal line
    #ax.set_xlabel('Rotation Angle (degrees)')
    #ax.set_ylabel('SSIM Score')
    #ax.set_title('SSIM Score vs. Rotation Angle')
    #ax.legend()
    #fig.savefig("/home/photolap/Documents/Symmetry_Plot.png")

    # analysing the results of the symmetry investigation
    if peaks is not None and max_value > 0.5:
        symmetry = " that are symmetric and consistent with the rest of the image"
    else:
        symmetry = ""

    # constructing the text prompt
    text_prompt = "an " + unique_identifier + " " + colours[0] + " background " + "with " + size + " " + colours[1] + " patches" + symmetry

    return text_prompt