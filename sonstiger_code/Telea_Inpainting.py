import cv2
import os
from PIL import Image

image_dir = ""	# Specify you input directory containing images and masks
output_dir = ""  # Specify your output directory

for imagename in os.listdir(image_dir):
    if imagename.endswith(".png"):
        print(imagename)
        image_path = os.path.join(image_dir, imagename)  # Construct the full image path
        image = cv2.imread(image_path)
        image_path_without_png = imagename.replace(".png", "")
        mask_image_path = os.path.join(image_dir, "masks", image_path_without_png + "_mask.png")
        mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
        # Inpainting using the Telea algorithm
        inpaintRadius = 1
        result = cv2.inpaint(image, mask_image, inpaintRadius, cv2.INPAINT_TELEA)
        # Display the original and inpainted images
        #cv2.imshow('Original Image', image)
        #cv2.imshow('Inpainted Image', result)
        #cv2.waitKey(100000)
        #cv2.destroyAllWindows()
        # Save the inpainted image to the output directory
        output_image_path = os.path.join(output_dir, image_path_without_png + "_inpainted.png")
        cv2.imwrite(output_image_path, result)