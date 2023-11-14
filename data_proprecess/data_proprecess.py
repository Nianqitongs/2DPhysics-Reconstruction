# 院校：南京信息工程大学
# 院系：自动化学院
# 开发时间：2023/11/14 13:09
import numpy as np
from torch.utils.data import Dataset
def read_npy_data(path_data, path_label):
    data_ral = np.load(path_data)  # (5790,64,64)
    label_ral = np.load(path_label)  # (5790,2)u,t

    indices = np.arange(5790)
    np.random.shuffle(indices)

    data_ral_shuffle = data_ral[indices]
    label_ral_shuffle = label_ral[indices]
    return data_ral_shuffle[:, None, ...], label_ral_shuffle

def create_data(data):
    keys = list(data.keys())
    values = list(data.values())
    data = [{key: value for key, value in zip(keys, values)} for values in zip(*values)]
    return data

class KeyDataset(Dataset):
    def __init__(self, dict_data):
        self.data = dict_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]