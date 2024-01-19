class ConflictMap:

    # Constructor
    def __init__(self, image_file=None, m_mask=None, assoc_semantic_city_model=None, assoc_point_cloud=None,
                 description=None, file_path=None):
        self._image_file = image_file
        self._m_mask = m_mask
        self._assoc_semantic_city_model = assoc_semantic_city_model
        self._assoc_point_cloud = assoc_point_cloud
        self._description = description
        self._file_path = file_path

    # Getter methods
    def get_image_file(self):
        return self._image_file

    def get_m_mask(self):
        return self._m_mask

    def get_assoc_semantic_city_model(self):
        return self._assoc_semantic_city_model

    def get_assoc_point_cloud(self):
        return self._assoc_point_cloud

    def get_description(self):
        return self._description

    def get_file_path(self):
        return self._file_path

    # Setter methods
    def set_image_file(self, new_image_file):
        self._image_file = new_image_file

    def set_m_mask(self, new_m_mask):
        self._m_mask = new_m_mask

    def set_assoc_semantic_city_model(self, new_assoc_semantic_city_model):
        self._assoc_semantic_city_model = new_assoc_semantic_city_model

    def set_assoc_point_cloud(self, new_assoc_point_cloud):
        self._assoc_point_cloud = new_assoc_point_cloud

    def set_description(self, new_description):
        self._description = new_description

    def set_file_path(self, new_file_path):
        self._file_path = new_file_path

