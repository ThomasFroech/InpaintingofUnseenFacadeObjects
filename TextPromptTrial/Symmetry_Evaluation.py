import numpy as np
from skimage import io, color, transform
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import math
import matplotlib.pyplot as plt
import cv2


def evaluate_symmetry(image_path, num_angles=360):

    # Load the image
    image = io.imread(image_path)

    # Defining the new width and height
    new_width = 1000
    new_height = 1000

    # scaling the image to a higher resolution
    neues_bild = cv2.resize(image, (new_width, new_height))

    # Convert the image to grayscale for SSIM calculation
    gray_image = color.rgb2gray(neues_bild)

    ssim_scores = []
    image_height, image_width = gray_image.shape

    for angle in range(0, 360, 360 // num_angles):
        # Rotate the image
        rotated_image = np.array(Image.fromarray(image).rotate(angle, expand=True))
        rotated_gray_image = color.rgb2gray(rotated_image)

        # Resize the rotated image to the original image dimensions
        rotated_gray_image = transform.resize(rotated_gray_image, (image_height, image_width))

        # Calculate the SSIM score with data_range specified
        score = ssim(gray_image, rotated_gray_image, data_range=1.0)
        ssim_scores.append(score)

    # Find the top three axes with the highest SSIM scores
    top_axes = np.argsort(ssim_scores)[-4:][::-1]

    return ssim_scores, top_axes


def plot_top_axes_in_image(image_path, top_axes):
    # Load the image
    image = io.imread(image_path)

    # Get the image dimensions
    image_height, image_width = image.shape[:2]

    # Create a copy of the image to draw lines on
    image_with_lines = np.copy(image)

    for angle in top_axes:
        if angle % 90 == 0:
            # Handle vertical lines (tangent is undefined)
            x1 = image_width // 2
            y1 = 0
            x2 = image_width // 2
            y2 = image_height
        else:
            # Calculate the slope and intercept of the line
            theta = math.radians(angle)
            slope = -1 / math.tan(theta)
            intercept = image_height // 2 - slope * (image_width // 2)

            # Calculate points for the line
            x1 = 0
            y1 = int(slope * x1 + intercept)
            x2 = image_width
            y2 = int(slope * x2 + intercept)

        # Draw the line on the image
        cv2.line(image_with_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Display the image with the lines
    plt.imshow(image_with_lines)
    plt.show()

def compare_axis_symmetry(image_path, num_angles=180, n=1):
    initial_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    new_width = 1000
    new_height = 1000
    image = cv2.resize(initial_image, (new_width, new_height))

    if image is None:
        print("Error while loading the image.")
        return

    height, width = image.shape
    center = (width // 2, height // 2)

    ssim_indices = []

    for angle in range(0, 181):
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        half_width = width // 2
        left_half = rotated_image[:, :half_width]
        right_half = rotated_image[:, half_width:]
        ssim_index = ssim(left_half, right_half)
        ssim_indices.append(ssim_index)

    # Choosing the best axes
    best_indices = np.argsort(ssim_indices)[-n:]

    # Crating a copy of the original image
    image_with_axes = image.copy()
    bgr_image = cv2.cvtColor(image_with_axes, cv2.COLOR_GRAY2BGR)

    for index in best_indices:
        angle = (180 / num_angles) * index
        # BCalculating the coordinates of the mirror-axis
        x1 = int(center[0] - np.sin(np.radians(angle)) * width)
        y1 = int(center[1] - np.cos(np.radians(angle)) * width)
        x2 = int(center[0] + np.sin(np.radians(angle)) * width)
        y2 = int(center[1] + np.cos(np.radians(angle)) * width)
        # Plotting the new image
        cv2.line(bgr_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the image with axes
    cv2.imshow('Image with Axes', bgr_image)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

    return ssim_indices



