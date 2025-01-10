import torch
from torch_geometric.data.dataset import Dataset
from torch_geometric.data.data import Data
import json
import numpy as np

class KinshipDataset(Dataset):
    def __init__(self,mode="train",max_nodes=21):
        with open("/Users/kunqixu/Desktop/Kinship/Emergent-Language-For-Kinship-Terms/my_code/"+mode+"_set.json", 'r') as f:
            data = json.load(f)
        cut_ids=(len(data)//32)*32
        data=data[:cut_ids]
        self.data_list=[]
        self.max_nodes = max_nodes
        for item in data:
            generations = np.array(item["generations"])
            genders = np.array(item["genders"])
            ids = np.array(item["ids"])
            edges_feature = item["edges_feature"]
            edge_attr = []
            edge_index = []
            edge_num = len(edges_feature)
            for e in edges_feature:
                edge_index.append([e[0], e[1]])
                edge_attr.append(e[2])
            edge_index = np.array(edge_index).reshape((2,edge_num))
            edge_attr = np.array(edge_attr).reshape((edge_num))

            caller_id = item["caller_id"]
            listener_id = item["listener_id"]

            caller_listener_vector = np.zeros_like(ids)
            caller_listener_vector[caller_id] = 1
            caller_listener_vector[listener_id] = -1

            test_listener = item["test_listener"]
            test_listener_vector = np.zeros_like(ids)
            test_listener_vector[test_listener] = -1

            labels = np.array(item["labels"])
            node_num = len(ids)
            generations = np.pad(generations, (0, self.max_nodes - len(generations)), mode='constant', constant_values=0)
            genders = np.pad(genders, (0, self.max_nodes - len(genders)), mode='constant', constant_values=0)
            
            labels = np.pad(labels, (0, self.max_nodes - len(ids)), mode='constant', constant_values=0)
            caller_listener_vector = np.pad(caller_listener_vector, (0, self.max_nodes - len(ids)), mode='constant', constant_values=0)
            test_listener_vector = np.pad(test_listener_vector, (0, self.max_nodes - len(ids)), mode='constant', constant_values=0)
            ids = np.pad(ids, (0, self.max_nodes - len(ids)), mode='constant', constant_values=0)
            node_feature = np.concatenate([generations.reshape(-1, 1), genders.reshape(-1, 1)], axis=1)
            
            itemdata=Data(
                x=torch.tensor(node_feature,dtype=torch.float32),
                edge_index=torch.tensor(edge_index,dtype=torch.int64),
                edge_attr=torch.tensor(edge_attr,dtype=torch.int64),
                y=torch.tensor(labels,dtype=torch.float32),
                caller_listener_vector=torch.tensor(caller_listener_vector),
                test_listener_vector=torch.tensor(test_listener_vector),
                node_num = node_num
            )
            self.data_list.append(itemdata)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
        
        #return  caller_listener_vector, labels, test_listener_vector, (node_feature, edge_index, edge_attr)

if __name__ == "__main__":
    ks = KinshipDataset(mode='test')
    print(ks.__getitem__(1))