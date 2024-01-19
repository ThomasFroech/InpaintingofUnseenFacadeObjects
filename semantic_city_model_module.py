from lxml import etree
from CityGML_Translation_Files import CityGMLTranslation as cgt
from decimal import Decimal
import subprocess

class SemanticCityModel:
    # Constructor
    def __init__(self, name=None, lod_level=None, model_path=None, description=None, citygml_version=None, randomCityOutputPath=None):
        self._name = name
        # the LOD-Level can only be specified according to the following list
        lod_values = ["LOD_0", "LOD_1", "LOD_2", "LOD_3", "LOD_4"]
        if lod_level in lod_values:
            self._lod_level = lod_level
        else:
            self._lod_level = None
            print("Invalid parameter value!")
            print("Valid LOD levels are: ", lod_values)
        self.model_path = model_path
        self.description = description
        self._citygml_version = citygml_version
        # The CityGML-version can only be specified according to the following list
        citygml_version_values = ['1.0', '2.0', '3.0']
        if citygml_version in citygml_version_values:
            self._lod_level = citygml_version
        else:
            self._lod_level = None
            print("Invalid parameter value!")
            print("Valid versions are: ", citygml_version_values)

        self._bounding_box = None
        self._relating_conflict_maps = []
        self._relating_point_clouds = []
        self._randomCityOutputPath = randomCityOutputPath

    # Getter methods
    def get_name(self):
        return self._name

    def get_lod_level(self):
        return self._lod_level

    def get_model_path(self):
        return self.model_path

    def get_description(self):
        return self.description

    def get_citygml_version(self):
        return self._citygml_version

    def get_bounding_box(self):
        return self._bounding_box

    def get_relating_conflict_maps(self):
        return self._relating_conflict_maps

    def get_relating_point_clouds(self):
        return self._relating_point_clouds

    def get_randomCityOutputPath(self):
        return self._randomCityOutputPath

    # Setter Methods
    def set_name(self, new_name):
        self._name = new_name

    def set_lod_level(self, new_lod_level):
        # the LOD-levels can only be specified according to the following list
        lod_values = ["LOD_0", "LOD_1", "LOD_2", "LOD_3", "LOD_4"]
        if new_lod_level in lod_values:
            self._lod_level = new_lod_level
        else:
            print("Invalid parameter value!")
            print("Valid LOD levels are: ", lod_values)

    def set_citygml_version(self, new_version):
        # the CityGML-version can only be specified according to the following list
        citygml_version_values = ['1.0', '2.0', '3.0']
        if new_version in citygml_version_values:
            self._lod_level = new_version
        else:
            print("Invalid parameter value!")
            print("Valid LOD levels are: ", citygml_version_values)

    def set_model_path(self, new_model_path):
        self.model_path = new_model_path

    def set_description(self, new_description):
        self._description = new_description

    def set_randomCityOutputPath(self, new_randomCityOutputPath):
        self._randomCityOutputPath = new_randomCityOutputPath

    # Adding or removing referencing to conflict maps
    def add_relating_conflict_map(self, new_relating_conflict_map):
        self._relating_conflict_maps.append(new_relating_conflict_map)

    def remove_relating_conflict_map(self, conflict_map_to_be_removed):
        self._relating_conflict_maps.remove(conflict_map_to_be_removed)

    # Adding or removing referencing to point clouds
    def add_relating_point_cloud(self, new_relating_point_cloud):
        self._relating_point_clouds.append(new_relating_point_cloud)

    def remove_relating_point_cloud(self, point_cloud_to_be_removed):
        self._relating_point_clouds.remove(point_cloud_to_be_removed)

    def get_namespaces(self):
        # this function is based on  the CityGML2OBJs package
        #
        # # Copyright (c) 2014
        # # Filip Biljecki
        # # Delft University of Technology
        # # fbiljecki@gmail.com

        tree = etree.parse(self.model_path)
        root = tree.getroot()
        # -- Determine CityGML version
        # If 1.0
        if root.tag == "{http://www.opengis.net/citygml/1.0}CityModel":
            # -- Name spaces
            ns_citygml = "http://www.opengis.net/citygml/1.0"
            ns_gml = "http://www.opengis.net/gml"
            ns_bldg = "http://www.opengis.net/citygml/building/1.0"
            ns_tran = "http://www.opengis.net/citygml/transportation/1.0"
            ns_veg = "http://www.opengis.net/citygml/vegetation/1.0"
            ns_gen = "http://www.opengis.net/citygml/generics/1.0"
            ns_xsi = "http://www.w3.org/2001/XMLSchema-instance"
            ns_xAL = "urn:oasis:names:tc:ciq:xsdschema:xAL:1.0"
            ns_xlink = "http://www.w3.org/1999/xlink"
            ns_dem = "http://www.opengis.net/citygml/relief/1.0"
            ns_frn = "http://www.opengis.net/citygml/cityfurniture/1.0"
            ns_tun = "http://www.opengis.net/citygml/tunnel/1.0"
            ns_wtr = "http://www.opengis.net/citygml/waterbody/1.0"
            ns_brid = "http://www.opengis.net/citygml/bridge/1.0"
            ns_app = "http://www.opengis.net/citygml/appearance/1.0"

            # Erstelle ein leeres Wörterbuch
            ns_dict = {}

            # Füge die Schlüssel-Wert-Paare zum Wörterbuch hinzu
            ns_dict['ns_citygml'] = ns_citygml
            ns_dict['ns_gml'] = ns_gml
            ns_dict['ns_bldg'] = ns_bldg
            ns_dict['ns_tran'] = ns_tran
            ns_dict['ns_veg'] = ns_veg
            ns_dict['ns_gen'] = ns_gen
            ns_dict['ns_xsi'] = ns_xsi
            ns_dict['ns_xAL'] = ns_xAL
            ns_dict['ns_xlink'] = ns_xlink
            ns_dict['ns_dem'] = ns_dem
            ns_dict['ns_frn'] = ns_frn
            ns_dict['ns_tun'] = ns_tun
            ns_dict['ns_wtr'] = ns_wtr
            ns_dict['ns_brid'] = ns_brid
            ns_dict['ns_app'] = ns_app

        # -- Else probably means 2.0
        else:
            # -- Name spaces
            ns_citygml = "http://www.opengis.net/citygml/2.0"

            ns_gml = "http://www.opengis.net/gml"
            ns_bldg = "http://www.opengis.net/citygml/building/2.0"
            ns_tran = "http://www.opengis.net/citygml/transportation/2.0"
            ns_veg = "http://www.opengis.net/citygml/vegetation/2.0"
            ns_gen = "http://www.opengis.net/citygml/generics/2.0"
            ns_xsi = "http://www.w3.org/2001/XMLSchema-instance"
            ns_xAL = "urn:oasis:names:tc:ciq:xsdschema:xAL:2.0"
            ns_xlink = "http://www.w3.org/1999/xlink"
            ns_dem = "http://www.opengis.net/citygml/relief/2.0"
            ns_frn = "http://www.opengis.net/citygml/cityfurniture/2.0"
            ns_tun = "http://www.opengis.net/citygml/tunnel/2.0"
            ns_wtr = "http://www.opengis.net/citygml/waterbody/2.0"
            ns_brid = "http://www.opengis.net/citygml/bridge/2.0"
            ns_app = "http://www.opengis.net/citygml/appearance/2.0"

        nsmap = {
            None: ns_citygml,
            'gml': ns_gml,
            'bldg': ns_bldg,
            'tran': ns_tran,
            'veg': ns_veg,
            'gen': ns_gen,
            'xsi': ns_xsi,
            'xAL': ns_xAL,
            'xlink': ns_xlink,
            'dem': ns_dem,
            'frn': ns_frn,
            'tun': ns_tun,
            'brid': ns_brid,
            'app': ns_app
        }
        return [ns_citygml, ns_bldg, ns_gml, ns_frn, ns_veg]

    def get_wall_surfaces(self):
        # this function is based on  the CityGML2OBJs package
        #
        # # Copyright (c) 2014
        # # Filip Biljecki
        # # Delft University of Technology
        # # fbiljecki@gmail.com
        # CityGML-Datei laden
        self.set_model_path(r"C:\Users\thoma\Documents\Master_GUG\Masterarbeit\Random_City_Output\LOD3_3.gml")
        tree = etree.parse(self.model_path)
        print("Tree:", tree)
        root = tree.getroot()
        print("Root: ", root)
        # Definieren der namespaces
        namespaces = self.get_namespaces()
        # Finding all the building elements
        cityObjects = []
        buildings = []
        # Find all instances of cityObjectMember and put them in a list
        for obj in root.getiterator('{%s}cityObjectMember' % namespaces[0]):
            cityObjects.append(obj)
        if len(cityObjects) > 0:
            for cityObject in cityObjects:
                for child in cityObject.getchildren():
                    if child.tag == '{%s}Building' % namespaces[1]:
                        buildings.append(child)
        # Find all the "boundedBy" objects and put them in a list
        boundedBy = []
        for building in buildings:
            for child in building.getchildren():
                if child.tag == '{%s}boundedBy' % namespaces[1]:
                    boundedBy.append(child)
        wallSurfaces = []
        for bby in boundedBy:
            for child in bby:
                if child.tag == '{%s}WallSurface' % namespaces[1]:
                    wallSurfaces.append(child)

        return wallSurfaces

    # This function has not been used in the end
    def apply_global_shift(self, path_to_shift_file, output_directory):
        # load the global shift data
        global_shift = []
        with open(path_to_shift_file, "r") as file:
            # Read the whole content of the documents
            inhalt = file.read()
            inhalt_splitted = inhalt.split('\n')
            for i in range(len(inhalt_splitted) - 1):
                global_shift.append(Decimal(float(inhalt_splitted[i])))

        # Get the namespaces
        namespaces = self.get_namespaces()
        ns_citygml = namespaces[0]
        ns_bldg = namespaces[1]
        ns_gml = namespaces[2]
        ns_frn = namespaces[3]
        ns_veg = namespaces[4]

        # Defining the filename
        filename = self.get_description()

        # Reading and parsing the CityGML file(s)
        CITYGML = etree.parse(self.model_path)

        # Getting the root of the XML tree
        root = CITYGML.getroot()

        # apply the translation
        cgt.translateToLocalCRS(CITYGML, file, root, ns_bldg, ns_gml, ns_citygml, ns_frn, ns_veg, output_directory,
                                global_shift, filename, write2file=False)

        # Updating the semantic model
        self.set_description(self.get_description() + "_local_")
        self.set_model_path(self.get_description() + ".gml" )
        return 0

    def create_random_city_model(self, number=1, lodSpec='LOD3_3'):
        # This function makes use of the following repository: https://github.com/tudelft3d/Random3Dcity
        path_part_1 = r"Random_City_Module\randomiseCity.py"
        path_part_2 = r"Random_City_Module\generateCityGML.py"
        arg_outputfile = r"-o"
        arg_outputfile_value = self._randomCityOutputPath + "file.xml"
        arg_number = r"-n"
        arg_number_value = str(number)
        arg_folder = r"-o"
        arg_folder_value = self._randomCityOutputPath
        arg_filename = r"-i"

        # Execute the first part of the random citygml creation
        subprocess.run(["python", path_part_1, arg_outputfile, arg_outputfile_value, arg_number, arg_number_value])

        # Execute the second part of the random citygml creation
        subprocess.run(["python", path_part_2, arg_filename, arg_outputfile_value, arg_folder, arg_folder_value])

        # Update the City Model and the related parameters
        self.set_description("Random_City_Model")
        path2model = arg_folder_value + '/' + lodSpec + ".gml"
        self.set_model_path(path2model)

