from local_code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from local_code.stage_5_code.Method_GNN import Method_GNN

# cora, citeseer, pubmed
DATASET_NAME = 'citeseer'

loader = Dataset_Loader(dName=DATASET_NAME)
loader.dataset_name = DATASET_NAME # replace cora with whatever dataset
loader.dataset_source_folder_path = '../../data/stage_5_data/' + loader.dataset_name
data = loader.load()

method = Method_GNN(mName='GCN', mDescription='GCN for node classification')
method.data = data
method.train()