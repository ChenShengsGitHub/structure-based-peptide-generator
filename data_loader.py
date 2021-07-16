import pdb
import random

import torch.utils.data as data
import json
import numpy as np
import torch
import cfg


class PeptideInfo(data.Dataset):
    def __init__(self, data_path, process_mode='train'):
        with open(data_path, 'r') as load_f:
            json_data = json.load(load_f)
        self.sequeces = []
        self.eigen_CMs = []
        self.process_mode = process_mode
        self.validate_start = 0
        self.test_start = 0


        if cfg.train_mode == 'pretrain':
            self.ex_data = json.load(open(cfg.pepbdb_processed))

            self.total_sample = len(json_data)
            self.validate_start = int(round(self.total_sample * 4 / 5))
            self.test_start = int(round(self.total_sample * 99 / 100))
            for peptide in json_data:
                self.sequeces.append(np.array(peptide))  # peptide is sequence
            for key in self.ex_data:
                for eigen_CM in self.ex_data[key]['eigen_CM']:
                    self.eigen_CMs.append(np.array(eigen_CM) / cfg.EGCM_max_value)
        else:#cfg.train_mode ==finetune or z_gen
            now = 0
            self.total_sample = 0
            for peptide in json_data:
                self.total_sample += len(json_data[peptide]['name'])
            for key in json_data:
                for eigen_CM in json_data[key]['eigen_CM']:
                    self.sequeces.append(np.array(json_data[key]['peptide'][0]))
                    self.eigen_CMs.append(np.array(eigen_CM) / cfg.EGCM_max_value)
                    now += 1
                if float(now) / self.total_sample >= 4 / 5 and self.validate_start == 0:
                    self.validate_start = now
                if float(now) / self.total_sample >= 9 / 10 and self.test_start == 0:
                    self.test_start = now


    def __getitem__(self, index):
        if self.process_mode == 'train':
            ind = index
        elif self.process_mode == 'validate':
            ind = index + self.validate_start
        elif self.process_mode == 'test':
            ind = index + self.test_start
        if cfg.train_mode == 'pretrain':
            eigen_CM = random.choice(self.eigen_CMs)
        else:#cfg.train_mode ==finetune or z_gen
            eigen_CM = self.eigen_CMs[ind]
        return self.sequeces[ind], torch.FloatTensor(eigen_CM)




    def __len__(self):
        if self.process_mode == 'train':
            return self.validate_start
        elif self.process_mode == 'validate':
            return self.test_start - self.validate_start
        elif self.process_mode == 'test':
            return self.total_sample - self.test_start


