import open3d as o3d
import numpy as np
from PIL import Image as im

# The code in this Python file is based on the research from the following paper: [Anonymized]
# Please note that the code implemented is fully experimental. It is not optimized in any way!

def normalize(pointArray):
    # this function is used to perform a pose normalization of the input
    # point data.

    # Step 1: Translation: move the whole point cloud to the Coordinates of the mean value

    # Calculation of the mean-values:
    xmean = np.mean(pointArray[:, 0])
    ymean = np.mean(pointArray[:, 1])
    zmean = np.mean(pointArray[:, 2])
    print("Mean Point: ", xmean, " ", ymean, " ", zmean)

    # Performing the actual translation
    length = len(pointArray[:, 0])
    point_array_translated = np.zeros((length, 3))

    a = 0
    for i in pointArray[:, 0]:
        point_array_translated[a, 0] = pointArray[a, 0] - xmean
        point_array_translated[a, 1] = pointArray[a, 1] - ymean
        point_array_translated[a, 2] = pointArray[a, 2] - zmean
        a = a + 1

    # Step 2: Scale invariance

    # finding the indices of the maximal coordinate values of the point cloud in each direction in order to find
    # A factor to scale the point cloud's furthest point to a standard distance
    ind_max_values_xyz = np.zeros((1, 3))
    ind_max_values_xyz[0, 0] = np.argmax(np.absolute(point_array_translated[:, 0]))
    ind_max_values_xyz[0, 1] = np.argmax(np.absolute(point_array_translated[:, 1]))
    ind_max_values_xyz[0, 2] = np.argmax(np.absolute(point_array_translated[:, 2]))

    # finding the actual values, and the largest one of these values
    a = int(ind_max_values_xyz[0, 0])
    b = int(ind_max_values_xyz[0, 1])
    c = int(ind_max_values_xyz[0, 2])
    max_values_xyz = np.ones((1, 3))
    max_values_xyz[0, 0] = point_array_translated[a, 0]
    max_values_xyz[0, 1] = point_array_translated[b, 1]
    max_values_xyz[0, 2] = point_array_translated[c, 2]

    # Calculation of tha factor that the whole point cloud should be scaled with
    # so that scale invariance is achieved
    ind_max_max_values_xyz = np.argmax(np.absolute(max_values_xyz))
    scaling_factor = 10 / (max_values_xyz[0, ind_max_max_values_xyz])
    print("Scaling factor: ", scaling_factor)

    # Applying of the scaling factor:
    pos = 0
    pointArrayTraRotSca = np.zeros((length, 3))
    for m in point_array_translated[:, 0]:
        pos2 = 0
        for n in point_array_translated[0, :]:
            pointArrayTraRotSca[pos, pos2] = scaling_factor * point_array_translated[pos, pos2]
            pos2 = pos2 + 1
        pos = pos + 1

    # write the nupy array to a o3d Pointcloud file
    # print("test: ", pointArrayTraRotSca)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointArrayTraRotSca)
    o3d.io.write_point_cloud("Test_normalized.ply", pcd)
    return pcd

# The code starts here

# Step 1: Defining the path to the point cloud
path_to_cloud = r"..."

# Spec must be either 'xz' or 'yz' according to the plane that should be chosen
spec = 'xz'

# Defining the size of the new image
number_of_rows = 339
number_of_columns = 453

# Step 2: loading the point cloud data
tree = o3d.io.read_point_cloud(path_to_cloud)
print("point cloud has been sucessfully loaded")

# Obtaining the tree-points
treepoints_tmp = np.asarray(tree.points)

# Normalize the point cloud
normalized_tree_pointcloud = normalize(treepoints_tmp)

# Convert to a numpy array
treepoints = np.asarray(normalized_tree_pointcloud.points)

# Step 3: Projecting the point cloud to one of the coordinate planes
if spec == 'xz':
    treepoints_projected = treepoints[:, 1:]
elif spec == 'yz':
    points1 = treepoints[:, 0]
    # print("Points 1: ", points1)
    points2 = treepoints[:, 2]
    # print("points 2: ", points2)
    counter = 0
    treepoints_projected = np.ones((len(points1), 2))
    for i in points1:
        treepoints_projected[counter, 0] = points1[counter]
        treepoints_projected[counter, 1] = points2[counter]
        counter = counter + 1
else:
    treepoints_projected = None

treepoints_projected = treepoints_projected

# Step 4: performing the image generation
startimage = np.zeros((number_of_rows, number_of_columns), dtype=np.uint8)

# Step 5 Calculating the maximal Extent of the point cloud
max_extent_x = max(abs(treepoints_projected[:, 0]))
max_extent_y = max(abs(treepoints_projected[:, 1]))
maxValues = [max_extent_x + 1, max_extent_y + 1]
gesMax = max(maxValues)
pixel_size = (2 * gesMax) / number_of_columns

# Step 6 : Create a Grid in order to sample the projected point cloud
grid = np.ceil(treepoints_projected / pixel_size)
# print("Grid: ", grid)

# Step 7: Translate the matrix to the center so that there are only positive values
shape = np.shape(grid)
grid_trans = np.zeros(shape)
counter1 = 0
for i in grid[:, 0]:
    grid_trans[counter1, 0] = grid[counter1, 0] + number_of_rows / 2
    counter1 = counter1 + 1
counter2 = 0
for j in grid[:, 1]:
    grid_trans[counter2, 1] = grid[counter2, 1] + number_of_columns / 2
    counter2 = counter2 + 1
# Assign the values to the respective raster cells
counter = 0
for i in grid_trans[:, 0]:
    a = int(grid_trans[counter, 0])
    # print("a: ", a)
    b = int(grid_trans[counter, 1])
    # print("b: ", b)
    startimage[a, b] = 250
    # print("yoo")
    counter = counter + 1

# Step 8 write the image to a test file in order to inspect the result
data = im.fromarray(startimage)
data.show()
