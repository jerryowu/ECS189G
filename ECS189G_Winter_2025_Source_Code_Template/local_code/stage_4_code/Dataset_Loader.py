'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import os


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')

        if self.dataset_source_file_name == "text_classification":
            X_train = []
            y_train = []
            X_test = []
            y_test = []
            for filename in os.listdir(self.dataset_source_folder_path + "/" + self.dataset_source_file_name + "/train/pos/"):
                file_path = os.path.join(self.dataset_source_folder_path  + "/" + self.dataset_source_file_name + "/train/pos/", filename)
                if os.path.isfile(file_path):
                    with open(file_path, 'r') as f:
                        contents = f.read()
                        X_train.append(contents)
                        y_train.append(1)

            for filename in os.listdir(self.dataset_source_folder_path + "/" + self.dataset_source_file_name + "/train/neg/"):
                file_path = os.path.join(self.dataset_source_folder_path + "/" + self.dataset_source_file_name + "/train/neg/", filename)
                if os.path.isfile(file_path):
                    with open(file_path, 'r') as f:
                        contents = f.read()
                        X_train.append(contents)
                        y_train.append(0)

            for filename in os.listdir(self.dataset_source_folder_path + "/" + self.dataset_source_file_name + "/test/pos/"):
                file_path = os.path.join(self.dataset_source_folder_path + "/" + self.dataset_source_file_name + "/test/pos/", filename)
                if os.path.isfile(file_path):
                    with open(file_path, 'r') as f:
                        contents = f.read()
                        X_test.append(contents)
                        y_test.append(1)

            for filename in os.listdir(self.dataset_source_folder_path  +  "/" + self.dataset_source_file_name + "/test/neg/"):
                file_path = os.path.join(self.dataset_source_folder_path + "/" + self.dataset_source_file_name + "/test/neg/", filename)
                with open(file_path, 'r') as f:
                    contents = f.read()
                    X_test.append(contents)
                    y_test.append(0)

            self.data = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
            return self.data

        elif self.dataset_source_file_name == "text_generation":
            X = []
            f = open(self.dataset_source_folder_path + "/" + self.dataset_source_file_name + "/data", 'r')
            for line in f:
                line = line.strip('\n')
                X.append(line)
            f.close()
            self.data = {'X': X}
            return self.data

        ## if invalid input
        self.data = {}
        return self.data