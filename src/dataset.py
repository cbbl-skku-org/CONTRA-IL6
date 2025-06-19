from torch.utils.data import Dataset
import torch
import os
from torch.nn import functional as F

class IL6_Dataset(Dataset):
    def __init__(self, dataset_config, is_training=True):
        self.pos_feat_paths = []
        self.neg_feat_paths = []
        self.dataset_config = dataset_config
        self.embedding_types = 'mean_representations' if dataset_config.mean else 'representations'
        self.max_length = dataset_config.max_length
        
        if is_training:
            for feature_name in dataset_config.feature_list:
                self.pos_feat_paths.append(
                    os.path.join(dataset_config.data_root, feature_name, 'train_pos')
                )
                self.neg_feat_paths.append(
                    os.path.join(dataset_config.data_root, feature_name, 'train_neg')
                )
        else:
            for feature_name in dataset_config.feature_list:
                self.pos_feat_paths.append(
                    os.path.join(dataset_config.data_root, feature_name, 'test_pos')
                )
                self.neg_feat_paths.append(
                    os.path.join(dataset_config.data_root, feature_name, 'test_neg')
                )
        
        self._setup_keys()
        self._preload_data()
    
    def get_pep_keys(self):
        return self.pep_keys
        
    def _setup_keys(self):
        self.pep_keys = [f'Negative_{i+1}' for i in range(len(os.listdir(self.neg_feat_paths[0])))] + \
            [f'Positive_{i+1}' for i in range(len(os.listdir(self.pos_feat_paths[0])))]
        self.labels = [0] * len(os.listdir(self.neg_feat_paths[0])) + [1] * len(os.listdir(self.pos_feat_paths[0]))
        
    def _preload_data(self):
        self.data = {}
        
        for key in self.pep_keys:
            if key.startswith('Negative'):    
                self.data[key] = [
                    torch.load(os.path.join(path, f'{key}.pt'))[self.embedding_types]
                    for path in self.neg_feat_paths
                ]
            else:
                self.data[key] = [
                    torch.load(os.path.join(path, f'{key}.pt'))[self.embedding_types]
                    for path in self.pos_feat_paths
                ]
        
    
    def __len__(self):
        return len(self.pep_keys)
    
    def __getitem__(self, idx):
        pep_key = self.pep_keys[idx]
        X = self.data[pep_key]
        
        len_tokens = [x.shape[0] if not self.dataset_config.mean else 1 for x in X]
        
        masks_X = [None for l in len_tokens]
        
        X = [x.detach() for x in X]
        
        if not self.dataset_config.mean:
            for i in range(len(X)):
                X[i] = F.pad(X[i], (0, 0, 0, self.max_length - X[i].size(0)), value=0)
                masks_X[i] = torch.ones(self.max_length, dtype=torch.bool)
                masks_X[i][:len_tokens[i]] = False
        
        return X, masks_X, self.labels[idx]