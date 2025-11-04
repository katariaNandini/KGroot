from DataSetGraphSimGenerator import CustomDataset
import configparser
import json
import os
import sys
import logging

from collections import defaultdict
from functools import partial
from typing import Set, List, Any, Optional

import time as pytime
from datetime import datetime
import socket
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from Radm import RAdam

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import pandas as pd
from tensorboard_logger import TensorBoardWritter
from model_batch import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_rank(y_true: Set[Any], y_pred: List[Any], max_rank: Optional[int] = None) -> List[float]:
    rank_dict = defaultdict(lambda: len(y_pred) + 1 if max_rank is None else (max_rank + len(y_pred)) / 2)
    for idx, item in enumerate(y_pred, start=1):
        if item in y_true:
            rank_dict[item] = idx
    return [rank_dict[_] for _ in y_true]

def MAR(y_true: List[Set[Any]], y_pred: List[List[Any]], max_rank: Optional[int] = None):
    return np.mean([np.mean(get_rank(a, b, max_rank)) for a, b in zip(y_true, y_pred)])

class ModelInference:
    def __init__(self):
        logging.error("device:{}".format(device))
        torch.set_printoptions(linewidth=120)
        torch.set_grad_enabled(True)
        np.random.seed(5)
        torch.manual_seed(0)

        self.config = configparser.ConfigParser()
        self.config_file_path = os.path.join(os.path.dirname(__file__), "config_graph_sim.ini")
        self.config.read(self.config_file_path, encoding='utf-8')

        self.__load_super_paras()
        self.cross_weight_auto = None

        self.model = None
        self.model_saved_path = None
        self.model_saved_dir = None

        self.tb_comment = self.data_set_name
        self.tb_logger = None

    def __load_super_paras(self):
        self.data_set_id = self.config.getint("data", "DATASET")
        self.data_set_name = "train_ticket" if self.data_set_id==1 else "sock_shop"
        self.input_dim = self.config.getint("model", "input_dim")
        self.gcn_hidden_dim = self.config.getint("model", "gcn_hidden_dim")
        self.linear_hidden_dim = self.config.getint("model", "linear_hidden_dim")
        self.num_bases = self.config.getint("model", "num_bases")
        self.dropout = self.config.getfloat("model", "dropout")
        self.support = self.config.getint("model", "support")
        self.max_node_num = self.config.getint("model", "max_node_num")
        self.pool_step = self.config.getint("model", "pool_step")
        self.lr = self.config.getfloat("train", "LR")
        self.weight_decay = self.config.getfloat("train", "l2norm")
        self.resplit = self.config.getboolean("data", "resplit")
        self.batch_size = self.config.getint("data", "batch_size")
        self.resplit_each_time = self.config.getboolean("data", "resplit_each_time")
        self.repeat_pos_data = self.config.getint("data", "repeat_pos_data")
        self.dataset_version = self.config.get("data", "dataset_version")

        self.epoch = self.config.getint("train", "NB_EPOCH")
        self.user_comment = self.config.get("train", "comment")
        self.criterion = F.cross_entropy

    def __start_tb_logger(self, time_str):
        self.tb_log_dir = os.path.join(os.path.dirname(__file__), 'runs/%s' % time_str).replace("\\", os.sep).replace("/", os.sep)
        self.tb_logger = TensorBoardWritter(log_dir="{}_{}{}".format(self.tb_log_dir, socket.gethostname(), self.tb_comment + self.user_comment), comment=self.tb_comment)

    def __stop_tb_logger(self):
        del self.tb_logger
        self.tb_logger = None

    def __print_paras(self, model):
        for name, param in model.named_parameters():
            logging.warning("name:{} param:{}".format(name, param.requires_grad))

    def generate_labeled_data(self):
        return

    def __new_model_obj(self):
        return GraphSimilarity(input_dim=self.input_dim,
                               gcn_hidden_dim=self.gcn_hidden_dim,
                               linear_hidden_dim=self.linear_hidden_dim,
                               pool_step=self.pool_step,
                               num_bases=self.num_bases,
                               dropout=self.dropout,
                               support=self.support,
                               max_node_num=self.max_node_num)

    def __print_data_info(self):
        train_data = CustomDataset(self.data_set_id, self.dataset_version, self.max_node_num, "train", self.repeat_pos_data, False)
        test_data = CustomDataset(self.data_set_id, self.dataset_version, self.max_node_num, "test", self.repeat_pos_data, False)
        val_data = CustomDataset(self.data_set_id, self.dataset_version, self.max_node_num, "val", self.repeat_pos_data, False)
        if hasattr(train_data, "print_data_set_info"): train_data.print_data_set_info()
        if hasattr(test_data, "print_data_set_info"): test_data.print_data_set_info()
        if hasattr(val_data, "print_data_set_info"): val_data.print_data_set_info()
        for datas in [train_data, test_data, val_data]:
            for index, data in enumerate(datas):
                adj_1 = np.array(data["graph_online_adj"].cpu())[0]
                f_1 = np.array(data["graph_online_feature"].cpu())
                adj_2 = np.array(data["graph_kb_adj"].cpu())[0]
                f_2 = np.array(data["graph_kb_feature"].cpu())
                self.tb_logger.writer.add_histogram("graph_online/adj", adj_1, index)
                self.tb_logger.writer.add_histogram("graph_online/feature", f_1, index)
                self.tb_logger.writer.add_histogram("graph_kb/adj", adj_2, index)
                self.tb_logger.writer.add_histogram("graph_kb/feature", f_2, index)

    def crossentropy_loss(self, output, label, num_list):
        num_list.reverse()
        weight_ = torch.as_tensor(num_list, dtype=torch.float32, device=device)
        weight_ = weight_ / torch.sum(weight_)
        self.cross_weight_auto = np.array(weight_.cpu())
        return self.criterion(output, label, weight=weight_)

    def train_model(self):
        start_time_train = str(datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.__start_tb_logger(time_str= start_time_train)

        self.model = self.__new_model_obj()
        self.__print_paras(self.model)
        self.model = self.model.to(device)

        criterion = self.crossentropy_loss
        optimizer = RAdam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                               patience=8, threshold=1e-4, threshold_mode="rel",
                                                               cooldown=0, min_lr=0, eps=1e-8)

        train_data = CustomDataset(self.data_set_id, self.dataset_version, self.max_node_num, "train", self.repeat_pos_data, self.resplit)
        self.__print_data_info()
        pos_train_num, neg_train_num = train_data.pos_neg_num()

        for epoch in range(self.epoch):
            if self.resplit_each_time:
                train_data = CustomDataset(self.data_set_id, self.dataset_version, self.max_node_num, "train", self.repeat_pos_data, self.resplit)
            train_loader = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True)
            loss_all = 0
            accuary_all_num = 0
            preds_all_num = 0
            FN, FP, TN, TP = 0, 0, 0, 0
            outputs_all = list()
            for batch in train_loader:
                graphs_online = (batch["graph_online_feature"], batch["graph_online_adj"])
                graphs_offline = (batch["graph_kb_feature"], batch["graph_kb_adj"])
                labels = batch["label"]
                outputs = self.model(graphs_online, graphs_offline)
                outputs_all.append(outputs)
                loss = criterion(outputs, labels, num_list=[neg_train_num, pos_train_num])

                preds = torch.argmax(outputs, dim=1)
                accuary_all_num += torch.sum(preds == labels)
                preds_all_num += torch.as_tensor(labels.shape[0])
                FN += int(torch.sum(preds[labels == 1] == 0))
                FP += int(torch.sum(preds[labels == 0] == 1))
                TN += int(torch.sum(preds[labels == 0] == 0))
                TP += int(torch.sum(preds[labels == 1] == 1))

                loss_all += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step(loss_all)

            accuary_train = accuary_all_num.item() / max(1, preds_all_num.item())
            if accuary_train >= 0.99:
                self.save_model(time_str=start_time_train)

            _ = self.test_val_model(mode="val")
            _ = self.test_val_model(mode="test")

        self.test_val_model(mode="test")
        self.save_model(time_str=start_time_train)
        self.__stop_tb_logger()

    @torch.no_grad()
    def test_val_model(self, mode):
        test_data = CustomDataset(self.data_set_id, self.dataset_version, self.max_node_num, mode, self.repeat_pos_data, False)
        test_loader = DataLoader(dataset=test_data, batch_size=self.batch_size, shuffle=True)
        accuary_all_num = 0
        preds_all_num = 0
        FN, FP, TN, TP = 0, 0, 0, 0
        loss_all = 0.0
        pred_class = list()
        pred_class_t3 = list()
        pred_class_t2 = list()
        pred_class_t5 = list()
        label_class = list()
        pos_num, neg_num = test_data.pos_neg_num()
        for batch in test_loader:
            graphs_online = (batch["graph_online_feature"], batch["graph_online_adj"])
            graphs_offline = (batch["graph_kb_feature"], batch["graph_kb_adj"])
            labels = batch["label"]
            start_time = pytime.perf_counter()
            outputs = self.model(graphs_online, graphs_offline)
            end_time = pytime.perf_counter()
            loss = self.criterion(outputs, labels, weight=torch.as_tensor(self.cross_weight_auto, dtype=torch.float32, device=device))
            loss_all += loss

            outputs_sm = torch.nn.functional.softmax(outputs, dim=1)
            outputs_sq = torch.squeeze(outputs_sm.narrow(1, 1, 1))  # class-1 probabilities

            if outputs_sq.dim() == 0:
                top_1_idx = torch.tensor([0], device=outputs_sq.device)
                top_2_idx = top_1_idx
                top_3_idx = top_1_idx
                top_5_idx = top_1_idx
            else:
                n = outputs_sq.size(0)
                k1 = min(1, n); k2 = min(2, n); k3 = min(3, n); k5 = min(5, n)
                top_1_idx = torch.topk(outputs_sq, k=k1)[1]
                top_2_idx = torch.topk(outputs_sq, k=k2)[1]
                top_3_idx = torch.topk(outputs_sq, k=k3)[1]
                top_5_idx = torch.topk(outputs_sq, k=k5)[1]

            outputs_max_index = top_1_idx[0]
            if torch.argmax(labels) in top_3_idx: pred_class_t3.append(torch.argmax(labels))
            else: pred_class_t3.append(top_3_idx[0])
            if torch.argmax(labels) in top_2_idx: pred_class_t2.append(torch.argmax(labels))
            else: pred_class_t2.append(top_2_idx[0])
            if torch.argmax(labels) in top_5_idx: pred_class_t5.append(torch.argmax(labels))
            else: pred_class_t5.append(top_5_idx[0])

            pred_class.append(outputs_max_index)
            label_class.append(torch.argmax(labels))

            preds = torch.argmax(outputs, dim=1)
            accuary_all_num += torch.sum(preds == labels)
            preds_all_num += torch.as_tensor(labels.shape[0])
            FN += int(torch.sum(preds[labels==1]==0))
            FP += int(torch.sum(preds[labels==0]==1))
            TN += int(torch.sum(preds[labels==0]==0))
            TP += int(torch.sum(preds[labels==1]==1))

        time_ms = (end_time - start_time)*1000 if 'end_time' in locals() else 0.0
        precision = TP / (TP + FP) if (TP + FP) else 0
        recall = TP / (TP + FN) if (TP + FN) else 0
        F1 = ((2 * precision * recall) / (precision + recall)) if (precision and recall) else 0

        if len(pred_class) == 0:
            logging.error("{}_data : empty dataset; returning zeros".format(mode))
            base_ac = [0.0, 0.0]
            return 0.0, torch.tensor(0.0), base_ac, 0.0, 0.0, 0.0

        pred_class_s = torch.stack(pred_class)
        label_class_s = torch.stack(label_class)
        pred_class_t3_s = torch.stack(pred_class_t3)
        pred_class_t2_s = torch.stack(pred_class_t2)
        pred_class_t5_s = torch.stack(pred_class_t5)

        y_true_list = [{int(l.item())} for l in label_class]
        y_pred_list = [[int(p.item())] for p in pred_class]
        MAR_res = MAR(y_true_list, y_pred_list)

        top1_a = torch.sum(pred_class_s == label_class_s).item() / pred_class_s.size()[0]
        top3_a = torch.sum(pred_class_t3_s == label_class_s).item() / pred_class_t3_s.size()[0]
        top2_a = torch.sum(pred_class_t2_s == label_class_s).item() / pred_class_t2_s.size()[0]
        top5_a = torch.sum(pred_class_t5_s == label_class_s).item() / pred_class_t5_s.size()[0]

        logging.error("{}_data : accuracy2:{} accuracy3:{} accuracy5:{} accuracy1:{}/{}={} MAR={} precision:{}/{}={}  recall:{}/{}={}  F1:{} time:{}".format(
            mode, top2_a, top3_a, top5_a, accuary_all_num, preds_all_num, int(accuary_all_num) / int(preds_all_num), MAR_res,
            TP, (TP + FP), precision, TP, (TP + FN), recall, F1, time_ms
        ))
        pos, neg = test_data.pos_neg_num()
        base_ac = [pos/(pos+neg), neg/(pos+neg)]
        return int(accuary_all_num) / int(preds_all_num), loss_all, base_ac, precision, recall, F1

    def save_model(self, time_str):
        dir_name = "{}_{}".format(time_str, socket.gethostname()+self.user_comment)
        save_path_dir = os.path.join(os.path.dirname(__file__), "..", "data", "graph_sim_model_parameters", self.data_set_name, dir_name)
        os.makedirs(save_path_dir, exist_ok=True)
        self.model_saved_path = os.path.join(save_path_dir, "model.pth")
        self.model_saved_dir = save_path_dir
        torch.save(self.model.state_dict(), self.model_saved_path)
        shutil.copy(self.config_file_path, os.path.join(save_path_dir, "config_graph_sim.ini"))

    def load_model(self, model_saved_dir):
        if model_saved_dir:
            self.model_saved_dir = model_saved_dir
            self.model_saved_path = os.path.join(model_saved_dir, "model.pth")
            self.config_file_path = os.path.join(model_saved_dir, "config_graph_sim.ini")
            self.config.read(self.config_file_path, encoding='utf-8')
            self.__load_super_paras()
        self.model = self.__new_model_obj()
        self.model = self.model.to(device)
        self.model.load_state_dict(torch.load(self.model_saved_path, map_location=device))
        self.model.eval()

def get_error_summary(did = 1):
    pass

def save_json_data(save_path, pre_save_data):
    with open(save_path, 'w', encoding='utf-8') as file_writer:
        raw_data = json.dumps(pre_save_data, indent=4)
        file_writer.write(raw_data)

if __name__ == '__main__':
    minf = ModelInference()
    minf.train_model()
