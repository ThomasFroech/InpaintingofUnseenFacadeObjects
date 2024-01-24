import open3d as o3d
import numpy as np
from PIL import Image as im

# The code in this Python file is based on the research from the following paper: [Anonymized]

def normalize(pointArray):
    # this function is used to perform a pose normalization of the input
    # point data.
    # Step 1: Translation: move the whole point cloud to the
    #         Coordinates of the mean value
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

    # Step 3: Scale invariance
    # finding the indices of the maximal coordinate values of the point cloud in each direction in order to find
    # A factor to scale the point cloud's furthest point to a standard distance
    ind_maxValuesXYZ = np.zeros((1, 3))
    ind_maxValuesXYZ[0, 0] = np.argmax(np.absolute(point_array_translated[:, 0]))
    ind_maxValuesXYZ[0, 1] = np.argmax(np.absolute(point_array_translated[:, 1]))
    ind_maxValuesXYZ[0, 2] = np.argmax(np.absolute(point_array_translated[:, 2]))
    # print("Maximal value indices: ", ind_maxValuesXYZ)
    # finding the actual values, and the largest one of these values
    a = int(ind_maxValuesXYZ[0, 0])
    b = int(ind_maxValuesXYZ[0, 1])
    c = int(ind_maxValuesXYZ[0, 2])
    maxValuesXYZ = np.ones((1, 3))
    maxValuesXYZ[0, 0] = point_array_translated[a, 0]
    maxValuesXYZ[0, 1] = point_array_translated[b, 1]
    maxValuesXYZ[0, 2] = point_array_translated[c, 2]
    # Calculation of tha factor that the whole point cloud should be scaled with
    # so that scale invariance is achieved
    ind_maxMaxValuesXYZ = np.argmax(np.absolute(maxValuesXYZ))
    scalingFactor = 10 / (maxValuesXYZ[0, ind_maxMaxValuesXYZ])
    print("Scaling factor: ", scalingFactor)
    # Applying of the scaling factor:
    pos = 0
    pointArrayTraRotSca = np.zeros((length, 3))
    # print("TEST: ", pointArrayTraRot)
    for m in point_array_translated[:, 0]:
        pos2 = 0
        for n in point_array_translated[0, :]:
            pointArrayTraRotSca[pos, pos2] = scalingFactor * point_array_translated[pos, pos2]
            pos2 = pos2 + 1
        pos = pos + 1
    # write the nupy array to a o3d Pointcloud file
    # print("test: ", pointArrayTraRotSca)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointArrayTraRotSca)
    o3d.io.write_point_cloud("Test_normalized.ply", pcd)
    return pcd


# Step 1: defining tha path to the point cloud data
path2cloud = r"/home/photolap/Documents/Masterarbeit_Froech/Baum_Philipp/nicetree23.pcd"
# Step 2: loading the point cloud data
tree = o3d.io.read_point_cloud(path2cloud)
print("point cloud has been sucessfully loaded")
# Step 3: projecting th point cloud to one of the coordinat eplanes
# Spec must be either 'xz' or 'yz' accoring to the plane that should be choses
spec = 'xz'
# Obtaining the tree-points
treepoints_tmp = np.asarray(tree.points)

normalized_tree_pointcloud = normalize(treepoints_tmp)

treepoints = np.asarray(normalized_tree_pointcloud.points)

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
print(treepoints_projected)
print("point cloud has been projected")
# Step 4: performing the image generation
#defining the size of the image
numberOfRows = 339
numberOfColumns = 453
startimage = np.zeros((numberOfRows, numberOfColumns), dtype=np.uint8)
print("The startimage has been created")


# Step 2 : Sample the projected point cloud
# Step 2.1: Calculate the necessary sampling distance
maxExtentX = max(abs(treepoints_projected[:, 0]))
# print("Ind_maxExtentX: ", maxExtentX)
maxExtentY = max(abs(treepoints_projected[:, 1]))
# print("Ind_maxExtentY: ", maxExtentY)
maxValues = [maxExtentX+1, maxExtentY+1]
gesMax = max(maxValues)
print("gesMax: ", gesMax)
pixelSize = (2 * gesMax) / numberOfColumns
print("Pixel Size: ", pixelSize)
# Step 3 : Create a Grid in order to sample the projected point cloud
grid = np.ceil(treepoints_projected / pixelSize)
# print("Grid: ", grid)
# Translate the matrix to the center so that there are only positive values
shape = np.shape(grid)
grid_trans = np.zeros(shape)
counter1 = 0
for i in grid[:, 0]:
    grid_trans[counter1, 0] = grid[counter1, 0] + (numberOfRows) / 2
    counter1 = counter1 + 1
counter2 = 0
for j in grid[:, 1]:
    grid_trans[counter2, 1] = grid[counter2, 1] + (numberOfColumns) / 2
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
# write the image to a test file in order to inspect the result
data = im.fromarray(startimage)
data.show()
