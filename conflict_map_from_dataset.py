import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

source_database = "cmp_extended"
path_to_etrims = r"C:\Users\thoma\Documents\Master_GUG\Masterarbeit\Datasets\eTRIMS\etrims-db_v1\annotations\08_etrims-ds"
path_to_cmp = r"C:\Users\thoma\Documents\Master_GUG\Masterarbeit\Datasets\CMP_Dataset_extended\extended"

if source_database == "etrims":
    # Reading the data
    image_list = []
    image_names = []
    for annotatedImage in os.listdir(path_to_etrims):
        if annotatedImage.endswith(".png"):
            imagePath = os.path.join(path_to_etrims, annotatedImage)
            image = Image.open(imagePath)
            image_list.append(image)
            image_names.append(annotatedImage)
    print(len(image_list), "annotated images have been found.")
    counter = 0
    # iterating over all the loaded images
    for i, img in enumerate(image_list):
        img_array = np.array(img)  # Convert the image to a NumPy array
        print("Unique values before: ", np.unique(img_array))
        # defining the different maks
        mask_building = (img_array == 1)
        mask_car = (img_array == 2)
        mask_door = (img_array == 3)
        mask_pavement = (img_array == 4)
        mask_road = (img_array == 5)
        mask_sky = (img_array == 6)
        mask_vegetation = (img_array == 7)
        mask_window = (img_array == 8)
        # defining the role of the masks in the final conflict maps
        img_array[mask_building] = 0
        img_array[mask_car] = 255
        img_array[mask_door] = 255
        img_array[mask_pavement] = 255
        img_array[mask_road] = 255
        img_array[mask_sky] = 255
        img_array[mask_vegetation] = 255
        img_array[mask_window] = 255

        previous_name = image_names[counter].replace(".png", "")
        new_file_name = f"{previous_name}_conflictMap.png"
        # defining the path to the new conflict map
        new_image_path = os.path.join(r"C:\Users\thoma\Documents\Master_GUG\Masterarbeit\Code_Output\Subfolder",
                                      new_file_name)
        conflict_map = Image.fromarray(img_array)
        # saving the new conflict map
        conflict_map.save(new_image_path)

        # creating masks from vegetation and cars
        img_array[~mask_car] = 0
        img_array[mask_car] = 255
        img_array[mask_vegetation] = 255
        previous_name = image_names[counter].replace(".png", "")
        new_file_name = f"{previous_name}_conflictMap_mask.png"
        # defining the path to the new conflict map
        new_image_path = os.path.join(r"C:\Users\thoma\Documents\Master_GUG\Masterarbeit\Code_Output\Subfolder",
                                      new_file_name)
        conflict_map = Image.fromarray(img_array)
        # saving the new conflict map
        conflict_map.save(new_image_path)

        counter = counter + 1

elif source_database == "cmp_extended":
    # Reading the data
    image_list = []
    image_names = []
    for annotatedImage in os.listdir(path_to_cmp):
        if annotatedImage.endswith(".png"):
            imagePath = os.path.join(path_to_cmp, annotatedImage)
            image = Image.open(imagePath)
            image_list.append(image)
            image_names.append(annotatedImage)
    print(len(image_list), "annotated images have been found.")

    counter = 0
    for i, img in enumerate(image_list):
        img_array = np.array(img)  # Convert the image to a NumPy array
        print("Unique values before: ", np.unique(img_array))
        # Every value is set to 0 except for "2: facade", indicating that there is a conflict
        # with the LOD2 model when the value differs from 2
        # Attention: the Background is treated as conflicting, there might be another possibility, however
        # Create a mask where pixel values are 2
        mask = (img_array == 2)
        # Set pixels with value 2 to black (0)
        img_array[mask] = 0
        # Set all other pixels to white (255)
        img_array[~mask] = 255
        # defining the name of the new conflict map
        previous_name = image_names[counter].replace(".png", "")
        new_file_name = f"{previous_name}_conflictMap.png"
        # defining the path to the new conflict map
        new_image_path = os.path.join(r"C:\Users\thoma\Documents\Master_GUG\Masterarbeit\Code_Output\Subfolder",
                                      new_file_name)
        conflict_map = Image.fromarray(img_array)
        # saving the new conflict map
        conflict_map.save(new_image_path)
        counter = counter + 1

else:
    print("Invalid data source!")
