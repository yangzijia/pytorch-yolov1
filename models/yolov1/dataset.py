import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class YoloDataset(Dataset):

    def __init__(self, transforms=None, lines=None, bbox_num=2, classes_num=2):
        print('data init')
        self.transforms = transforms
        self.bbox_num = bbox_num
        self.classes_num = classes_num
        self.image_path_list = []
        self.boxes = []
        self.labels = []
        self.image_size = 448
        for line in lines:
            splited = line.strip().split(" ")
            self.image_path_list.append(splited[0])
            box = []
            label = []
            for bbox in splited[1:]:
                x1, y1, x2, y2, c = bbox.split(",")
                box.append([float(x1), float(y1), float(x2), float(y2)])
                label.append(int(c) + 1)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        self.num_samples = len(self.boxes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        img = cv2.imread(os.path.join(image_path))
        boxes = self.boxes[idx].clone()
        h, w, _ = img.shape
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)
        labels = self.labels[idx].clone()
        target = self.convert_bbox2labels(boxes, labels)  # 7x7x30
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = self.transforms(img)

        return img, target

    def convert_bbox2labels(self, boxes, labels):
        """将bbox的(cls,x,y,w,h)数据转换为训练时方便计算Loss的数据形式(7,7,5*B+cls_num)
        注意，输入的bbox的信息是(xc,yc,w,h)格式的，转换为labels后，bbox的信息转换为了(px,py,w,h)格式
            boxes (tensor) [[x1,y1,x2,y2],[]]
            labels (tensor) [...]
            return (7,7,5*B+cls_num)
        """
        grid_size = 1.0 / 7
        target = np.zeros((7, 7, 5 * self.bbox_num + self.classes_num))  # 注意，此处需要根据不同数据集的类别个数进行修改
        # boxes[:, 2:] 表示x2,y2的矩阵，boxes[:, :2] 表示x1,y1的矩阵
        wh = boxes[:, 2:] - boxes[:, :2]  # 生成宽高的矩阵
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2  # 生成中心点的矩阵
        for i in range(cxcy.size()[0]):  # 循环获取每一个box
            cxcy_sample = cxcy[i]
            ij = (cxcy_sample / grid_size).ceil() - 1  # 向上取整-1     找出中心点坐落的网格位置

            for j in range(self.bbox_num):
                # 匹配到的网格的左上角相对坐标
                xy = ij * grid_size
                # 计算出obj中心点相对网格左上角的位置坐标
                delta_xy = (cxcy_sample - xy) / grid_size
                target[int(ij[1]), int(ij[0]), (5 * j):(5 * j + 2)] = delta_xy  # 0:2
                target[int(ij[1]), int(ij[0]), (5 * j + 2):(5 * j + 4)] = wh[i]  # 2:4
                # 预测的分数confidence
                target[int(ij[1]), int(ij[0]), 5 * (j + 1) - 1] = 1  # 5

            # 类别
            target[int(ij[1]), int(ij[0]), int(labels[i]) + (5 * self.bbox_num - 1)] = 1

        return target


def load_dataset(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
    return lines


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms

    annotation_path = 'shape_voc/trainval.txt'

    train_loader = DataLoader(yoloDataset(transforms=transforms.ToTensor(), lines=load_dataset(annotation_path)),
                              batch_size=4, shuffle=True)
    for i, (img, target) in enumerate(train_loader):
        print(img.shape, target.shape)
    # train_iter = iter(train_loader)
    # for i in range(100):
    #     img, target = next(train_iter)
    #     print(img, target)
