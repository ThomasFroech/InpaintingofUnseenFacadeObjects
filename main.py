import conflict_map_generator as cmg
import point_cloud_module as pcm
import semantic_city_model_module as scm
import os
import argparse
import glob
import time
import json

# Messung der Startzeit
starting_time = time.time()

# -- Parse command-line arguments
PARSER = argparse.ArgumentParser(description='Masters Thesis Fröch')
PARSER.add_argument('-i1', '--directory1',
                    help='Directory containing .pcd files.', required=True)

PARSER.add_argument('-i2', '--directory2',
                    help='Directory containing .gml files.', required=True)

PARSER.add_argument('-o', '--results',
                    help='Directory where the output should be written', required=True)

PARSER.add_argument('-gt', '--groundTruthFiles',
                    help='Directory where the ground truth LoD3 models are stored', required=False)

PARSER.add_argument('-mp', '--MeshPath',
                    help='Directory where the obj files are stored', required=False)

PARSER.add_argument('-rcp', '--RandCityOut',
                    help='Directory where Random3DCity puts the xml and gml files', required=False)

PARSER.add_argument('-ann', '--annotatedImgFolder',
                    help='Directory where the annotated images are stored', required=False)

ARGS = vars(PARSER.parse_args())
DIRECTORY_1 = os.path.join(ARGS['directory1'], '')
DIRECTORY_2 = os.path.join(ARGS['directory2'], '')
RESULT = os.path.join(ARGS['results'], '')
try:
    GROUNDTRUTH = os.path.join(ARGS['groundTruthFiles'], '')
except:
    print("No ground-truth LOD3 path given")
try:
    MESHPATH = os.path.join(ARGS['MeshPath'], '')
except:
    print("No mesh path given!")
try:
    RAND3DCOUT = os.path.join(ARGS['RandCityOut'], '')
except:
    print("No output-path for the Random3Dcity application given!")
try:
    ANNDIR = os.path.join(ARGS['--annotatedImgFolder'], '')
except:
    print("No path to a database of annotated image given!")

# read the settings.json file
file_path = 'config.json'

# Read the contents of the file
with open(file_path, 'r') as file:
    parameter_data = json.load(file)

# Access the values
n_div = parameter_data["n_div"]
tolerance = parameter_data["tolerance"]
paramRansac = parameter_data["RANSACParam"]
lod3Tolerance = parameter_data["LoD3Tolerance"]

# Print the values
print(f"n_div: {n_div}")
print(f"tolerance: {tolerance}")
print(f"RANSAC parameters {paramRansac}")
print(f"LoD3 surface rolerance {lod3Tolerance}")

# -- Find all .pcd files in the respective directory
# os.chdir(DIRECTORY_1)
# -- Supported extensions
# types = ('*.pcd')
# Empty python lists for storage
# files_found = []
# point_clouds = []
# Finding all the .pcd files
# for files in types:
#    files_found.extend(glob.glob(files))
# files_found.pop()
# Iterating through all the found files
# print(len(files_found), " point clouds have been found.")
# print("Name of Fist File: ", files_found[0])
# for f in files_found:
#    FILENAME = f[:f.rfind('.pcd')]
#    #print("Filename: ", FILENAME)
#    FULLPATH = os.path.join(DIRECTORY_1, f)
#    # Creating 'MLSPointCloud objects; for ech pcd file one
#    # and append them to a python list
#    point_cloud = pcm.MLSPointCloud(file_path=FULLPATH, description=FILENAME)
#    point_clouds.append(point_cloud)
## Create a conflict map generator object
# cMapGen_1 = cmg.ConflictMapGenerator(point_clouds=point_clouds, output_path=RESULT,
#                                     mesh_path=MESHPATH, n_div=n_div, tol=tolerance)
# cMapGen_1.create_conflict_map(spec='obj')

# Comment the following code block in for randomly generating conflict maps

# Erstellung von Random Conflict maps
## Step 1: Creating a randomly generated semantic city model
#lodspec = "LOD3_3.gml"  # modify this parameter to generate random conflict maps at different LoD Levels
#model_path = RAND3DCOUT + "/" + lodspec
#print("ModelPath test: ", model_path)
#random_city_model = scm.SemanticCityModel(lod_level='LOD_3', citygml_version='2.0', model_path=model_path,
#                                          random_city_output_path=RAND3DCOUT)
#random_city_model.create_random_city_model(10, 'LOD_3')
## Step 2 instantiate a ConflictMapGenerator object
#cMapGen_1 = cmg.ConflictMapGenerator(city_models=random_city_model, output_path=RESULT)
## Step 3: Call the respective function
#cMapGen_1.create_random_conflict_map()

# Comment in in case you want to derive a conflict map from an LOD3 model that has been converted to the .obj format before
# Finding all CityGML datasets
# cMapGen_1 = cmg.ConflictMapGenerator(output_path=RESULT, ground_truth_path=GROUNDTRUTH)

# cMapGen_1.create_conflict_map_from_LOD3()

# Comment this part in if you want to derive conflict maps from either the CMP-database of
# annotated images or from the ETrims database of annotated images
# Step 1: Create a ConflictMapGenerator object
cMapGen_1 = cmg.ConflictMapGenerator(output_path=RESULT, path_to_annot_images=ANNDIR)
cMapGen_1.derive_from_annotation('cmp')

# Duration measurement
time_end = time.time()

# Calculation of Runtime
duration = time_end - starting_time
print(f"The duration of the program was {duration:.6f} seconds")