import os
import os.path as osp
import numpy as np
import torchvision
import utils.transforms as tf
from datasets.base_dataset import BaseDataset
from datasets.registry import DATASETS
import cv2
import torch


@DATASETS.register_module
class CULane(BaseDataset):
    def __init__(self, img_path, data_list, cfg=None):
        super().__init__(img_path, data_list, cfg=cfg)
        self.ori_imgh = 590
        self.ori_imgw = 1640

    def init(self):
        with open(osp.join(self.list_path, self.data_list)) as f:
            for line in f:
                line_split = line.strip().split(" ")
                self.img_name_list.append(line_split[0])
                self.full_img_path_list.append(self.img_path + line_split[0])
                if not self.is_training:
                    continue
                self.label_list.append(self.img_path + line_split[1])
                self.exist_list.append(
                    np.array([int(line_split[2]), int(line_split[3]),
                              int(line_split[4]), int(line_split[5])]))

    def transform_train(self):
        train_transform = torchvision.transforms.Compose([
            tf.GroupRandomRotation(degree=(-2, 2)),
            tf.GroupRandomHorizontalFlip(),
            tf.SampleResize((self.cfg.img_width, self.cfg.img_height)),
            tf.GroupNormalize(mean=(self.cfg.img_norm['mean'], (0, )), std=(
                self.cfg.img_norm['std'], (1, ))),
        ])
        return train_transform

    def probmap2lane(self, probmaps, exists, pts=18):  # probmap表示概率图，pts表示车道线上点数量
        coords = []
        probmaps = probmaps[1:, ...]  # 取出除第一维以外的所有数据
        exists = exists > 0.5  # 将车道线是否存在的概率进行二值化,>0.5置为true
        for probmap, exist in zip(probmaps, exists):
            if exist == 0:
                continue
            probmap = cv2.blur(probmap, (9, 9), borderType=cv2.BORDER_REPLICATE)
            thr = 0.3
            coordinate = np.zeros(pts)  # 创建18个初始值为0的元素
            cut_height = self.cfg.cut_height
            for i in range(pts):
                line = probmap[round(  # 根据车道线上当前点的位置计算概率图上的一行，并将其存储在line中
                    self.cfg.img_height-i*20/(self.ori_imgh-cut_height)*self.cfg.img_height)-1]

                if np.max(line) > thr:
                    coordinate[i] = np.argmax(line)+1
            if np.sum(coordinate > 0) < 2:
                continue
    
            img_coord = np.zeros((pts, 2))
            img_coord[:, :] = -1
            for idx, value in enumerate(coordinate):
                if value > 0:
                    # 根据车道线上的点的相对坐标，将其映射到原始图像上的横坐标
                    img_coord[idx][0] = round(value*self.ori_imgw/self.cfg.img_width-1)
                    img_coord[idx][1] = round(self.ori_imgh-idx*20-1)
    
            img_coord = img_coord.astype(int)
            coords.append(img_coord)
    
        return coords  # 返回所有车道线的坐标列表
