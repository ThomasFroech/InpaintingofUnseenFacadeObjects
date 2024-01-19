from CityGML_Translation_Files import markup3dmodule as m3dm, polygon3dmodule as p3dm
import open3d as o3d
from lxml import etree
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import os
from PIL import Image
import numpy as np
import math
import pyransac3d as pyrsc
from scipy.linalg import svd


class ConflictMapGenerator:

    def __init__(self, point_clouds=None, city_models=None, output_path=None, mesh_path=None, ground_truth_path=None,
                 n_div=0, tol=0, ransac_param=None, lod3_tolerance=None, path_to_annot_images=None):
        self._point_clouds = point_clouds
        self._city_models = city_models
        self._output_path = output_path
        self._mesh = None
        self._mesh_path = mesh_path
        self._ground_truth_path = ground_truth_path
        self._n_div = n_div
        self._tol = tol
        self._ransac_param = ransac_param
        self._lod3_tolerance = lod3_tolerance
        self._path_to_annot_images = path_to_annot_images

    # Getter methods
    def get_point_clouds(self):
        return self._point_clouds

    def get_city_models(self):
        return self._city_models

    def get_output_path(self):
        return self._output_path

    def get_mesh_path(self):
        return self._mesh_path

    def get_mesh(self):
        return self._mesh

    def get_ground_truth_path(self):
        return self._ground_truth_path

    def get_n_div(self):
        return self._n_div

    def get_tol(self):
        return self._tol

    def get_ransac_param(self):
        return self._ransac_param

    def get_lod3_tolerance(self):
        return self._lod3_tolerance

    def get_path_to_annot_images(self):
        return self._path_to_annot_images

    # Setter methods
    def set_point_clouds(self, new_pointClouds):
        self._point_clouds = new_pointClouds

    def set_city_models(self, new_clityModels):
        self._city_models = new_clityModels

    def set_output_path(self, new_output_path):
        self._output_path = new_output_path

    def set_mesh_path(self, newMeshPath):
        self._mesh_path = newMeshPath

    def set_mesh(self, new_mesh):
        self._mesh = new_mesh

    def set_ground_truth_path(self, new_ground_truth_path):
        self._ground_truth_path = new_ground_truth_path

    def set_n_div(self, new_n_div):
        self._n_div = new_n_div

    def set_tol(self, new_tol):
        self._tol = new_tol

    def set_ransacParam(self, new_ransacParam):
        self._ransac_param = new_ransacParam

    def set_lod3Tolerance(self, new_lod3Tolerance):
        self._lod3_tolerance = new_lod3Tolerance

    def set_path_to_annot_images(self, new_path_to_annot_images):
        self._path_to_annot_images = new_path_to_annot_images

    # Further Methods
    # This function is copied from the CityGML2OBJ functionality
    def remove_reccuring(self, list_vertices):
        # """Removes recurring vertices, which messes up the triangulation.
        # Inspired by http://stackoverflow.com/a/1143432"""
        # last_point = list_vertices[-1]
        list_vertices_without_last = list_vertices[:-1]
        found = set()
        for item in list_vertices_without_last:
            if str(item) not in found:
                yield item
                found.add(str(item))

    # A small function to load the .obj files and create a list of  open3D meshes from them
    def load_mesh(self):
        ppath = self.get_mesh_path()
        mesh_list = []  # Liste, in der die Mesh-Objekte gespeichert werden

        # Durchsuchen des Verzeichnisses nach Dateien
        files = os.listdir(ppath)
        supported_formats = [".obj"]  # Unterstützte Dateiformate

        for file in files:
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in supported_formats:
                file_path = os.path.join(ppath, file)
                mesh = o3d.io.read_triangle_mesh(file_path)
                if not mesh.is_empty():  # Überprüfen, ob das Laden erfolgreich war
                    mesh_list.append(mesh)

        # Überprüfen, ob Mesh-Objekte gefunden und geladen wurden
        if len(mesh_list) == 0:
            print("No supported mesh-files have been found!")
            return
        print("Mesh_list has length ", len(mesh_list))
        self.set_mesh(mesh_list)

    # This function takes the CityGML file and traverses through it in order to create an open3D mesh
    # The code in this function is basen on the CityGML2OBJs application (https://github.com/tudelft3d/CityGML2OBJs)
    # and (https://github.com/tum-gis/CityGML2OBJv2)
    def create_mesh_from_CityGML(self):
        # The code in this function is based on the CityGML2OBJFunctionality
        # Get the City model
        cityModel = self.get_city_models()

        # Reading and parsing the CityGML file(s)
        tree = etree.parse(cityModel.get_model_path())
        print("Tree:", tree)
        root = tree.getroot()

        # Initilaize empty arrays
        city_objects = []
        buildings = []

        # get the namespaces
        namespaces = self.get_city_models().get_namespaces()

        # Find all instances of cityObjectMember and put them in a list
        for obj in root.getiterator('{%s}cityObjectMember' % namespaces[0]):
            city_objects.append(obj)

        if len(city_objects) > 0:
            # Report the progress and contents of the CityGML file
            print("\tThere are", len(city_objects), "city_object(s) in this CityGML file.")

            # -- Store each building separately
            for city_object in city_objects:
                for child in city_object.getchildren():
                    if child.tag == '{%s}Building' % namespaces[1]:
                        buildings.append(child)

            print("\tThere are", len(buildings), "building(s) in this CityGML file.")
            b_counter = 0

            vertexlist = []
            length_vertices = 0
            for b in buildings:

                # Build the local list of vertices to speed up the indexing
                local_vertices = {}
                local_vertices['All'] = []

                # Increment the building counter
                b_counter += 1

                # OBJ with all surfaces in the same bin
                polys = m3dm.polygonFinder(b)

                # Process each surface
                polycounter = 0
                for poly in polys:

                    # Decompose the polygon into exterior and interior
                    e, i = m3dm.polydecomposer(poly)

                    # Points forming the exterior LinearRing
                    epoints = m3dm.GMLpoints(e[0])

                    # Clean recurring points, except the last one
                    last_ep = epoints[-1]
                    epoints_clean = list(self.remove_reccuring(epoints))
                    epoints_clean.append(last_ep)
                    # print("Länge der äußeren : ", len(epoints_clean))

                    # LinearRing(s) forming the interior
                    irings = []
                    for iring in i:
                        ipoints = m3dm.GMLpoints(iring)

                        # Clean them in the same manner as the exterior ring
                        last_ip = ipoints[-1]
                        ipoints_clean = list(self.remove_reccuring(ipoints))
                        ipoints_clean.append(last_ip)
                        irings.append(ipoints_clean)

                    try:
                        # Try to perform the triangulation
                        t = p3dm.triangulation(epoints_clean, irings)
                        # print("t: ", t)
                        # Iterate over all the triangles in the triangulated polygon
                        for triangle in t:
                            # Iterate over all the vertices in the triangle
                            for vertex in triangle:
                                # Append the vertices to the vertex-list
                                vertexlist.append(vertex)
                                length_vertices = length_vertices + 1
                        print("Ende erreicht!")
                    except:
                        print("An error has occurred!")
            print("VertexList: ", vertexlist)
            # Create a list that indicates which points belong to a triangle
            triangles = [[i, i + 1, i + 2] for i in range(0, length_vertices - 2, 3)]
            print("Triangles: ", triangles)
            # Create an open3D point cloud from the vertices
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(vertexlist)
            num_points = len(point_cloud.points)
            print("Number of points in the point cloud:", num_points)
            # Create the open3D mesh
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = point_cloud.points
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            # Visualize
            o3d.visualization.draw_geometries([mesh])
            self.set_mesh(mesh)

        return 0

    def project_Polygon_to_2D(self, polygon, projection_direction=None, return_projection_direction=False):
        # The code in this function is basen on the CityGML2OBJs application (https://github.com/tudelft3d/CityGML2OBJs)
        # and (https://github.com/tum-gis/CityGML2OBJv2)
        vertical = True
        if vertical:

            projected_polygon = [[point[0], point[2]] for point in polygon]

            # Check if the projected polygon is a line (all x-coordinates or all z-coordinates are the same)
            all_x = all(x == projected_polygon[0][0] for x, _ in projected_polygon)
            all_z = all(z == projected_polygon[0][1] for _, z in projected_polygon)

            if projection_direction is None:
                if all_x or all_z:
                    # Switch the projection direction to the other coordinate plane
                    if all_x:
                        projected_polygon = [[point[1], point[2]] for point in polygon]
                    elif all_z:
                        projected_polygon = [[point[0], point[1]] for point in polygon]

            if return_projection_direction and projection_direction is None:
                if all_x:
                    return projected_polygon, 'all_x'
                if all_z:
                    return projected_polygon, 'all_z'
                else:
                    return projected_polygon, 'correct'

            if projection_direction == 'all_x':
                projected_polygon = [[point[1], point[2]] for point in polygon]
                return projected_polygon

            if projection_direction == 'all_z':
                projected_polygon = [[point[0], point[1]] for point in polygon]
                return projected_polygon

            if projection_direction == "correct":
                projected_polygon = [[point[0], point[2]] for point in polygon]
                return projected_polygon

            if projection_direction is None and return_projection_direction is False:
                return projected_polygon
        else:
            return None

    def subdivide_meshes(self, n_iterations):
        # subdivide the triangle in the meshes for a higher resolution
        # get the meshes
        mesh_list = self.get_mesh()
        new_mesh_list = []
        for mesh in mesh_list:
            refined_mesh = mesh.subdivide_midpoint(number_of_iterations=n_iterations)
            new_mesh_list.append(refined_mesh)
        self.set_mesh(new_mesh_list)

    # just a little helper function to define the colors in the graphics according to the evaluated hit distance
    def color_by_distance(self, diff):
        tolerance = self.get_tol()
        if diff > tolerance:  # and diff < 3.00:
            return 'black'
        elif diff < -tolerance:  # and diff > -3.00:
            return 'red'
        elif abs(diff) <= tolerance:  # and abs(diff) < 3.00:
            return 'green'
        # elif diff > 3.00:
        #    return 'yellow'
        # elif diff < -3.00:
        #     return 'white'

    # just a little helper function that is used to obtain a dictionary that contains triangle indices and their
    # corresponding average hitting distance
    def calculate_average_distance_dictionary(self, triangle_indices_dict, differences):
        triangle_differences = {}
        triangle_count = {}
        average_distances = {}
        for triangle_index, index in triangle_indices_dict.items():
            average_distances[triangle_index] = np.mean(differences[index])

        return average_distances

    def create_conflict_map(self, spec):

        # The parameter 'spec' specifies which method should be used
        # The following specifications can be invoked:
        #
        #   'cgml' : the mesh is created from the CityGML file
        #   'obj'  : the mesh ios created from an .obj file

        if spec == 'cgml':
            print("Mesh is generated from CityGML file directly!")
            # Not implemented in this work due to problems with the conversion. thi might be adressed in the future
        elif spec == 'obj':
            print("Mesh is generated from .obj file!")
            # loading the mesh file by calling the respective function
            self.load_mesh()
            # retrieving the mesh object
            self.subdivide_meshes(self.get_n_div())
            print("Triangle subdivision process with", self.get_n_div(), "subdivision iterations is completed!")
            mesh_list = self.get_mesh()

            # retrieving the viewpoints for the ray casting
            viewpoints = []
            for pointcloud in self.get_point_clouds():
                viewpoint = pointcloud.get_Viewpoint()
                viewpoints.append(viewpoint)
            viewpoints.pop()

            # Calculating the directions and the distances
            directions = []
            pointss = []
            distances = []
            rayTensor = []
            for index in range(len(self.get_point_clouds()) - 1):  # Iterate over all the point clouds
                # Get all the points of a point cloud
                allPoints = self.get_point_clouds()[index].get_pcdCloud().points
                for point in allPoints:  # Iterate over all the points in the point cloud

                    # Berechne die quadrierten Differenzen der x-, y- und z-Koordinaten
                    diff_x = point[0] - viewpoints[index][0]
                    diff_y = point[1] - viewpoints[index][1]
                    diff_z = point[2] - viewpoints[index][2]

                    # Berechne die quadratische Summe der Differenzen
                    distance = math.sqrt(diff_x ** 2 + diff_y ** 2 + diff_z ** 2)
                    distances.append(distance)
                    # print(distance)

                    ray_directions = [diff_x, diff_y, diff_z]
                    normalized_vector = [ray_directions[0] / distance, ray_directions[1] / distance,
                                         ray_directions[2] / distance]

                    directions.append(normalized_vector)
                    pointss.append(point)
                    rayTensor.append(
                        [viewpoints[index][0], viewpoints[index][1], viewpoints[index][2], normalized_vector[0],
                         normalized_vector[1], normalized_vector[2]])

            # defining the open3D rays
            rays = o3d.core.Tensor(rayTensor, dtype=o3d.core.Dtype.Float32)
            print("Here: ", type(rays))

            # Converting the meshs into the legacy datatype and adding the to the scene
            scene = o3d.t.geometry.RaycastingScene()
            converted_meshes = []
            mesh_ids = []
            for mesh in mesh_list:
                converted_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
                mesh_id = scene.add_triangles(converted_mesh)
                mesh_ids.append(mesh_id)
                converted_meshes.append(converted_mesh)

            ans = scene.cast_rays(rays)

            # calculation of the intersection points
            intersection = ans['t_hit'].isfinite()
            intersection_points = rays[intersection][:, :3] + rays[intersection][:, 3:] * ans['t_hit'][
                intersection].reshape((-1, 1))
            intersection_points_np = intersection_points.numpy()
            # Creating a point cloud from the calculated intersection points for visualization purposes
            intersection_pointcloud = o3d.geometry.PointCloud()
            intersection_pointcloud.points = o3d.utility.Vector3dVector(intersection_points_np)
            # o3d.io.write_point_cloud("AAA_Punktwolke.pcd", intersection_pointcloud)

            # obtaining the hit distances as a numpy array
            hit_distances_tmp = (rays[:, 3:] * ans['t_hit'].reshape((-1, 1))).numpy()
            hit_distances = np.linalg.norm(hit_distances_tmp, axis=1)

            # obtaining the hit triangles
            hit_triangles = ans['primitive_ids'].numpy()

            # obtaining the differences between distance to the points and the distance to the intersection points
            diff = np.asarray(distances) - hit_distances

            # Zähler für Elemente, die nicht 'inf' sind
            anzahl_nicht_inf = 0

            # obtaining  the hit geometries
            hit_geometries = ans['geometry_ids'].numpy()
            unique_hit_geometries = np.unique(hit_geometries)

            # iterating over the different hit geometries
            for geometry_index in unique_hit_geometries[:-1]:
                # find the indices of the rays that hit the respective geometry
                indices_hit = np.where(hit_geometries == geometry_index)

                # get the corresponding mesh:
                mesh = mesh_list[geometry_index]

                # get all the indices of the hit triangles and append them to a list:
                # in this list, a triangle can appear more than just once
                triangle_indices_dict = {}  # Initialize an empty dictionary to store triangle indices and their positions

                for ind_hit in indices_hit[0]:
                    triangle_index = hit_triangles[ind_hit]

                    if triangle_index in triangle_indices_dict:
                        triangle_indices_dict[triangle_index].append(ind_hit)
                    else:
                        triangle_indices_dict[triangle_index] = [ind_hit]

                dictionary = self.calculate_average_distance_dictionary(triangle_indices_dict, diff)

                count = 0
                for triangle_index, avg_distance in dictionary.items():
                    if count < 5:
                        print(f"Triangle {triangle_index}: Average Distance = {avg_distance}")
                        count += 1
                    else:
                        break

                # First, we iterate over all triangles and plot them
                patches = []

                try:
                    for triangle_index, tri_ind in enumerate(mesh.triangles):
                        try:
                            triangle_vertices = np.asarray(mesh.vertices)[tri_ind]
                            # print("Triangle vertices: ", triangle_vertices)
                            projected_triangle = self.project_Polygon_to_2D(triangle_vertices)
                            # print("projected triangle: ", projected_triangle)
                            triangle_patch = Polygon(projected_triangle, closed=True, edgecolor='blue',
                                                     facecolor='blue')
                            patches.append(triangle_patch)
                        except:
                            print(
                                "a: The normal of the polygon has no magnitude. Check the polygon. The most common cause for this are two identical sequential points or collinear points.")

                    # iterate over the dictionary
                    for triangle_index, avg_distance in dictionary.items():
                        try:
                            triangle = np.asarray(mesh.triangles)[triangle_index]
                            triangle_vertices = np.asarray(mesh.vertices)[triangle]
                            # print("Hier")
                            # obtain the projected coordinates of the vertices
                            projected_triangle = self.project_Polygon_to_2D(triangle_vertices)
                            # plot the triangle according to its average distance
                            triangle_patch = Polygon(projected_triangle, closed=True,
                                                     edgecolor=self.color_by_distance(avg_distance),
                                                     facecolor=self.color_by_distance(avg_distance))
                            patches.append(triangle_patch)
                        except:
                            print(
                                "b: The normal of the polygon has no magnitude. Check the polygon. The most common cause for this are two identical sequential points or collinear points.")
                            print("")
                    # Erstelle die PatchCollection
                    collection = PatchCollection(patches, match_original=True)

                    # PÜLot all triangles in blue color, first
                    fig, ax = plt.subplots(figsize=(15, 15))
                    ax.axis('off')
                    ax.add_collection(collection)
                    ax.axis('off')
                    ax.autoscale_view()

                    name = self.get_mesh_path()

                    output_filename = "conflict_map_" + str(geometry_index)
                    output_filepath = os.path.join(self.get_output_path(), output_filename)

                    # saving the figure
                    plt.savefig(output_filepath, dpi=300)
                    plt.close(fig)

                    # Removing the white borders around the actual conflict maps
                    image = Image.open(os.path.join(self.get_output_path(), output_filename + ".png"))

                    # Convert the image to a NumPy array
                    np_image = np.array(image)

                    # Find the bounding box of the actual conflict map
                    non_white_pixels = np.any(np_image != [255, 255, 255, 255], axis=-1)
                    bbox = np.array([[
                        np.min(np.where(non_white_pixels)[1]),
                        np.min(np.where(non_white_pixels)[0]),
                        np.max(np.where(non_white_pixels)[1]),
                        np.max(np.where(non_white_pixels)[0])
                    ]])

                    # Crop the image using the bounding box
                    cropped_image = Image.fromarray(np_image[bbox[0][1]:bbox[0][3] + 1, bbox[0][0]:bbox[0][2] + 1, :])

                    # Save the edited image
                    cropped_image.save(os.path.join(self.get_output_path(), output_filename + ".png"))

                except Exception as e:
                    print("An unknown error has occurred: ", e)

        else:
            print("Please select a valid specification")

        return 0

    # this little helper function is inspired by the CityGML2OBJs functionality

    def create_random_conflict_map(self):
        # The code in this function is basen on the CityGML2OBJs application (https://github.com/tudelft3d/CityGML2OBJs)
        # and (https://github.com/tum-gis/CityGML2OBJv2)
        # obtaining the wall surfaces
        model = self.get_city_models()
        wall_surfaces = model.get_wall_surfaces()

        # Iterating over all the wall sufaces
        # This code is heavilý inspired by the citygml2objs application
        cf_map_counter = 0
        for wall_surface in wall_surfaces:

            # -- Build the local list of vertices to speed up the indexing
            local_vertices = {}
            local_vertices['All'] = []

            # -- OBJ with all surfaces in the same bin
            polys = m3dm.polygonFinder(wall_surface)

            # -- Process each surface
            for poly in polys:

                # -- Decompose the polygon into exterior and interior
                e, i = m3dm.polydecomposer(poly)

                # -- Points forming the exterior LinearRing
                epoints = m3dm.GMLpoints(e[0])

                # -- Clean recurring points, except the last one
                last_ep = epoints[-1]
                epoints_clean = list(self.remove_reccuring(epoints))
                epoints_clean.append(last_ep)
                # print("Länge der äußeren : ", len(epoints_clean))

                # -- LinearRing(s) forming the interior
                irings = []
                for iring in i:
                    ipoints = m3dm.GMLpoints(iring)

                    # -- Clean them in the same manner as the exterior ring
                    last_ip = ipoints[-1]
                    ipoints_clean = list(self.remove_reccuring(ipoints))
                    ipoints_clean.append(last_ip)
                    irings.append(ipoints_clean)

                # print("EpointsClean: ", epoints_clean)
                # print("Irings: ", irings)
                projected_exterior = self.project_Polygon_to_2D(epoints_clean)
                # print("Projected Exterior: ", projected_exterior)
                inner_polygons = []
                for inner in irings:
                    inner_polygons.append(self.project_Polygon_to_2D(inner))
                # print("Inner polygons: ", inner_polygons)

                if len(inner_polygons) > 0:
                    polygon_patch = Polygon(projected_exterior, closed=True, edgecolor='black', linewidth=2,
                                            facecolor='black')

                    hole_patches = [Polygon(hole, closed=True, edgecolor='white', linewidth=2, facecolor='white') for
                                    hole
                                    in
                                    inner_polygons]

                    patches = [polygon_patch] + hole_patches
                    collection = PatchCollection(patches, match_original=True)

                    # Setting up the plot
                    fig, ax = plt.subplots()
                    ax.add_collection(collection)

                    # Specifications of the axes
                    ax.autoscale()
                    ax.axis('off')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_title('')
                    # ax.grid(True)

                    output_filename = "conflict_map_" + str(cf_map_counter)
                    output_filepath = os.path.join(self.get_output_path(), output_filename)

                    # saving the figure
                    plt.savefig(output_filepath)

                    plt.close(fig)
                    cf_map_counter = cf_map_counter + 1
                    print("counter: ", cf_map_counter)

                    # Removing the white borders around the actual conflict maps
                    image = Image.open(os.path.join(self.get_output_path(), output_filename + ".png"))

                    # Convert the image to a NumPy array
                    np_image = np.array(image)

                    # Find the bounding box of the actual conflict map
                    non_white_pixels = np.any(np_image != [255, 255, 255, 255], axis=-1)
                    bbox = np.array([[
                        np.min(np.where(non_white_pixels)[1]),
                        np.min(np.where(non_white_pixels)[0]),
                        np.max(np.where(non_white_pixels)[1]),
                        np.max(np.where(non_white_pixels)[0])
                    ]])

                    # Crop the image using the bounding box
                    cropped_image = Image.fromarray(np_image[bbox[0][1]:bbox[0][3] + 1, bbox[0][0]:bbox[0][2] + 1, :])

                    # Save the edited image
                    cropped_image.save(os.path.join(self.get_output_path(), output_filename + ".png"))

        return 0

    def are_points_on_line_2d(self, point1, point2, point3):
        # Calculate the cross product of the vectors formed by the points
        cross_product = (point2[0] - point1[0]) * (point3[1] - point1[1]) - (point3[0] - point1[0]) * (
                point2[1] - point1[1])

        # Check if the cross product is very close to zero (within a small tolerance)
        return abs(cross_product) < 1e-10

    def is_polygon_horizontal_with_tolerance(self, polygon, tolerance=0.03):
        # Extract the y-coordinates of all vertices
        y_coordinates = [vertex[1] for vertex in polygon]

        # Calculate the maximum difference among y-coordinates
        max_difference = max(y_coordinates) - min(y_coordinates)

        # Check if the maximum difference is within the tolerance
        return max_difference <= tolerance

    def create_conflict_map_from_LOD3(self):
        # identify all the obj files in the folder
        lod3GroundTruthPath = self._ground_truth_path
        cfMapCounter = 0
        for filename in os.listdir(lod3GroundTruthPath):
            filenamewithoutobj = filename.replace(".ply", "")
            print("filename: ", filename)
            original_mesh = o3d.io.read_triangle_mesh(os.path.join(lod3GroundTruthPath, filename))
            mesh = original_mesh.subdivide_midpoint(number_of_iterations=self.get_n_div())
            # Access the vertices and triangles
            vertices = mesh.vertices
            vertices_array = np.asarray(vertices)
            print("Vertices: ", vertices_array)
            # centering the points
            centroid = np.mean(vertices_array, axis=0)
            centered_points = vertices_array - centroid
            print("Centroid")
            # identify the pricipal exes
            _, _, vh = svd(centered_points.T)
            print("svd")
            # the largest eigenvalue
            main_component = vh[0]
            print("Main_Components")
            # calculating the rotational angles
            angle = np.arctan2(main_component[1], main_component[0])

            # setting upt the rotation matrix
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                        [np.sin(angle), np.cos(angle), 0],
                                        [0, 0, 1]])

            # executing the rotation
            rotated_points = np.dot(centered_points, rotation_matrix.T)
            print("Rotated points")
            rotated_points += centroid
            # print("vertices: ", np.asarray(vertices))
            # calculation of the main facade plane
            # calculation of the main facade plane with RANSAC
            plane1 = pyrsc.Plane()
            print(rotated_points.shape)
            best_eq, best_inliers = plane1.fit(rotated_points, self.get_ransac_param()["thresh"], self.get_ransac_param()["minPoints"], self.get_ransac_param()["maxIteration"])
            a = best_eq[0]
            b = best_eq[1]
            c = best_eq[2]
            d = best_eq[3]
            print("Bis hier")
            triangles = mesh.triangles
            # Create a list to store projected triangles and their properties
            collection = []
            # Iterate over the triangles and get their vertices
            for i in range(len(triangles)):
                triangle = triangles[i]
                vertex_indices = triangle.tolist()  # Convert to a list of vertex indices
                triangle_vertices = [rotated_points[idx] for idx in vertex_indices]
                # triangle_vertices now contains the 3D coordinates of the triangle's vertices
                # print(f"Triangle {i} vertices: {triangle_vertices}")
                # calculate the distance to the previously defined plane
                # and check if all the points of the triangle are within this plane
                confirming = True
                for point in triangle_vertices:
                    distance = abs(a * point[0] + b * point[1] + c * point[2] + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)

                    if distance > self.get_lod3_tolerance():
                        confirming = False
                        # project the triangle to 2D
                        # if confirming == True:
                        # print("Confirming: ", confirming)

                projectedTriangle = self.project_Polygon_to_2D(triangle_vertices, projection_direction='all_x')

                if confirming == True:
                    triangle_patch = Polygon(projectedTriangle, closed=True, edgecolor='green', linewidth=0.1,
                                             facecolor='green')
                else:
                    triangle_patch = Polygon(projectedTriangle, closed=True, edgecolor='red', linewidth=0.1,
                                             facecolor='red')
                collection.append(triangle_patch)

            # Create a PatchCollection
            patch_collection = PatchCollection(collection, match_original=True)
            # Setting up the plot
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.add_collection(patch_collection)

            # Specifications of the axes
            ax.autoscale()
            ax.axis('off')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('')
            # ax.grid(True)

            output_filename = str(filenamewithoutobj) + "_conflict_map_" + str(cfMapCounter)
            output_filepath = os.path.join(self.get_output_path(), output_filename)

            # saving the figure
            plt.savefig(output_filepath, dpi=300)

            plt.close(fig)
            cfMapCounter = cfMapCounter + 1
            print("counter: ", cfMapCounter)

        return 0
    def derive_from_annotation(self, source_database):
        # This function can be applied ion order to derive conflict maps from annotated images of facades
        # There is functionalities to deploy it for the cmp-database and the etrims database
        # Source database can take two different parameter:
        #  1. "cmp"
        #  2. "etrims"
        # The mapping of the annotations that these databases feature are dataset-specific and can not be applied
        # for another dataset
        path_to_etrims = self.get_path_to_annot_images()
        path_to_cmp = self.get_path_to_annot_images()
        print(f"Searching for annotated images in {self.get_path_to_annot_images()}")

        if source_database == "etrims":
            # Reading the data
            image_list = []
            image_names = []
            for annotated_image in os.listdir(path_to_etrims):
                if annotated_image.endswith(".png"):
                    image_path = os.path.join(path_to_etrims, annotated_image)
                    image = Image.open(image_path)
                    image_list.append(image)
                    image_names.append(annotated_image)
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
                new_image_path = os.path.join(self.get_output_path(),
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
                new_image_path = os.path.join(self.get_output_path(),
                                              new_file_name)
                conflict_map = Image.fromarray(img_array)
                # saving the new conflict map
                conflict_map.save(new_image_path)

                counter = counter + 1

        elif source_database == "cmp":
            # Reading the data
            image_list = []
            image_names = []
            for annotated_image in os.listdir(path_to_cmp):
                if annotated_image.endswith(".png"):
                    image_path = os.path.join(path_to_cmp, annotated_image)
                    image = Image.open(image_path)
                    image_list.append(image)
                    image_names.append(annotated_image)
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
                new_image_path = os.path.join(self.get_output_path(),
                                              new_file_name)
                conflict_map = Image.fromarray(img_array)
                # saving the new conflict map
                conflict_map.save(new_image_path)
                counter = counter + 1

        else:
            print("Invalid data source!")
        return 0