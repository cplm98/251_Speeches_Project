import numpy as np

class FileHandler:
    def __init__(self, file_name, type): # None type for mixed, str, int, float
        self.file_name = file_name
        self.data_type = type
    
    def read_csv(self):
        self.data = np.genfromtxt(self.file_name, delimiter = ',', dtype=self.data_type)
    
    def get_data(self):
        return self.data
