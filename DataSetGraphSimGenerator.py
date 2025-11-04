import os, json, pickle, configparser, logging
import numpy as np
import torch
from torch.utils.data import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _read_config():
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(os.path.dirname(__file__), "config_graph_sim.ini"), encoding="utf-8")
    return cfg

class CustomDataset(Dataset):
    def __init__(self, data_set_id: int, dataset_version, max_node_num: int, mode: str,
                 repeat_pos_data: int = 1, resplit: bool = False):
        cfg = _read_config()
        self.dataset_dir = cfg.get("data", "dataset_dir")
        self.max_node_num = max_node_num
        self.input_dim = cfg.getint("model", "input_dim")
        self.support = cfg.getint("model", "support")

        file_map = {"train": cfg.get("data", "train_file"),
                    "val":   cfg.get("data", "valid_file"),
                    "test":  cfg.get("data", "test_file")}
        list_path = os.path.join(self.dataset_dir, file_map[mode])
        with open(list_path, "r") as f:
            self.rows = json.load(f)

        self._pos = sum(1 for _, _, y in self.rows if int(y) == 1)
        self._neg = len(self.rows) - self._pos
        self.pickle_dir = os.path.join(self.dataset_dir, "pickle_data")

    def __len__(self): 
        return len(self.rows)

    def _load_graph(self, fname: str):
        with open(os.path.join(self.pickle_dir, fname), "rb") as f:
            d = pickle.load(f)
        x = d["fetures"]  # keep repos key
        adjs = d["adj"]
        n = min(self.max_node_num, x.shape[0])
        feat = np.zeros((self.max_node_num, self.input_dim), dtype=np.float32)
        feat[:n, :self.input_dim] = x[:n, :self.input_dim]
        A = np.zeros((self.support, self.max_node_num, self.max_node_num), dtype=np.float32)
        for i in range(min(self.support, len(adjs))):
            Ai = adjs[i]
            A[i, :n, :n] = Ai[:n, :n]
        return torch.from_numpy(feat), torch.from_numpy(A)

    def __getitem__(self, idx: int):
        o, k, y = self.rows[idx]
        fo, Ao = self._load_graph(o)
        fk, Ak = self._load_graph(k)
        y = torch.tensor(int(y), dtype=torch.long)
        return {
            "graph_online_feature": fo,
            "graph_online_adj": Ao,
            "graph_kb_feature": fk,
            "graph_kb_adj": Ak,
            "label": y,
        }

    def pos_neg_num(self):
        return self._pos, self._neg

    def graph_class_data(self):
        return []

    def print_data_set_info(self):
        logging.error(f"dataset_dir:{self.dataset_dir} mode_len:{len(self)} pos:{self._pos} neg:{self._neg} input_dim:{self.input_dim} support:{self.support} max_node_num:{self.max_node_num}")
