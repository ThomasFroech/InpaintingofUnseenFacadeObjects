import numpy as np
import open3d as o3d
from pyproj import Transformer


class MLSPointCloud:

    def __init__(self, file_path, pcd_cloud=None, description=None, relating_semantic_city_model=None):
        self._file_path = file_path
        if pcd_cloud == None:
            self._pcd_cloud = o3d.io.read_point_cloud(file_path)
        else:
            self._pcd_cloud = pcd_cloud
        self._description = description
        self._number_of_points = None
        self._relating_semantic_city_model = relating_semantic_city_model
        self._relating_conflict_maps = []

    # Getter methods
    def get_file_path(self):
        return self._file_path

    def get_description(self):
        return self._description

    def get_relating_semantic_city_model(self):
        return self._relating_semantic_city_model

    def get_relating_conflict_maps(self):
        return self._relating_conflict_maps

    def get_pcd_cloud(self):
        return self._pcd_cloud

    # Setter methods

    def set_file_path(self, new_file_path):
        self._file_path = new_file_path

    def set_description(self, new_description):
        self._description = new_description

    def set_pcd_cloud(self, new_pcdCloud):
        self._pcd_cloud = new_pcdCloud

    # Adding and removing conflict maps
    def add_relating_conflict_map(self, new_conflict_map):
        self._relating_conflict_maps.append(new_conflict_map)

    def remove_relating_conflict_map(self, conflict_map_to_be_removed):
        self._relating_conflict_maps.remove(conflict_map_to_be_removed)

    def get_Viewpoint(self):
        # Obtaining the file path
        file_path = self.get_file_path()
        #print("File Path: ", file_path)
        try:
            # Opening the file
            with open(file_path, "r") as file:
                pcd_content = file.read()
        except:
            return 0

        viewpoint_line = None
        for line in pcd_content.split('\n'):
            # Extracting the line that contains the information on the viewpoint
            if line.startswith("VIEWPOINT"):
                viewpoint_line = line
                break

        if viewpoint_line:
            viewpoint_values = viewpoint_line.split(' ')[1:4]  # Ignoring the "Viewpoint" Keyword
            #print("viewpoint_values: ", viewpoint_values)
            viewpoint_array = np.array(viewpoint_values, dtype=np.float32)
        else:
            viewpoint_array = None
        return viewpoint_array

    # This function was not necessary in the end
    def enu_2_ecef(self, param, output_dir):
        # This function is used to convert the points from a ENU to an ECEF according to a 4x4 transformation matrix
        # It makes use of homogenous coordinates to apply the transformation
        cloud_enu = self.get_pcd_cloud()
        points_enu = np.asarray(cloud_enu.points)
        # Add homogeneous coordinate (1) to each point
        points_homogeneous = np.hstack((points_enu, np.ones((points_enu.shape[0], 1))))
        # print(points_homogeneous)
        # Transform points from ENU to ECEF
        points_ecef_4d = np.dot(points_homogeneous, param.T[:, :])
        # print(points_ecef_4d)
        points_ecef = points_ecef_4d[:, :3]
        # print(points_ecef)
        # print(points_ecef)
        # Create a new point cloud with the transformed points
        pcd_ecef = o3d.geometry.PointCloud()
        pcd_ecef.points = o3d.utility.Vector3dVector(points_ecef)
        # Save the new poibnt cloud that was created from the transformed points
        # o3d.io.write_point_cloud(output_dir + self.get_description() + "_ecef.pcd", pcd_ecef, write_ascii=True)
        # Update the pointcloud
        o3d.io.write_point_cloud(output_dir + self.get_description() + "_transformed.pcd", pcd_ecef,
                                 write_ascii=True)
        self.set_pcd_cloud(pcd_ecef)
        self.set_description(self.get_description() + "_ecef")
        return 0

    # This function was not necessary in the end
    def apply_coord_transform(self, inputEPSG, outputEPSG, output_dir):
        # This function is going to be used to convert the coordinates from the specified
        # input coordinated system to the specified output coordinate system.
        # Obtaining the points from the point cloud as a python array
        cloud_input = self.get_pcd_cloud()
        points_input = np.asarray(cloud_input.points)
        # Converting the point array into a list of individual points for the processing
        # Creating an empty list for the coordinate tupels to be stored in
        input_point_list = []
        # Iteration over all the points in the point cloud
        for index in range(len(points_input[:, 0])):
            # Creating a tupel for the point
            point_tupel = (points_input[index, 0], points_input[index, 1], points_input[index, 2])
            # Appending the new tupel to the list
            input_point_list.append(point_tupel)
        # Definition of the source coordinate system
        input_crs = inputEPSG
        target_crs = outputEPSG
        # Setting up a transformer
        transformer = Transformer.from_crs(input_crs, target_crs)
        # Transforming the coordinates with the transformer
        transformed_points = [transformer.transform(x, y, z) for x, y, z in input_point_list]
        # Transforming the points back into a numpy array
        transformed_points_array = np.array(transformed_points)
        #print(transformed_points_array)
        # Create a new point cloud with the transformed points
        transformed_pcd = o3d.geometry.PointCloud()
        transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points_array)
        # Save the new point cloud that was created from the transformed points
        o3d.io.write_point_cloud(output_dir + self.get_description() + "_transformed.pcd", transformed_pcd, write_ascii=True)
        # Update the pointcloud
        self.set_pcd_cloud(transformed_pcd)
        self.set_description(self.get_description() + "_transformed")
        return 0
