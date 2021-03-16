import os
import os.path

import torch.utils.data as data
from PIL import Image
import torch
from utils.util import cal_subitizing

NO_LABEL = -1

def make_union_dataset(root, edge=False):
    img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'ShadowImages')) if f.endswith('.jpg')]
    label_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'ShadowMasks')) if f.endswith('.png')]
    data_list = []
    if edge:
        edge_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'EdgeMasks')) if f.endswith('.png')]
        for img_name in img_list:
            if img_name in label_list:  # filter labeled data by seg label
            # if img_name in edge_list: # filter labeled data by edge label
                data_list.append((os.path.join(root, 'ShadowImages', img_name + '.jpg'),
                                  os.path.join(root, 'ShadowMasks', img_name + '.png'),
                                 os.path.join(root, 'EdgeMasks', img_name + '.png')))
            else: # we set label=-1 when comes to unlebaled data
                data_list.append((os.path.join(root, 'ShadowImages', img_name + '.jpg'), -1, -1))
    else:
        for img_name in img_list:
            if img_name in label_list:  # filter labeled data by seg label
            # if img_name in edge_list:  # filter labeled data by edge label
                data_list.append((os.path.join(root, 'ShadowImages', img_name + '.jpg'),
                                  os.path.join(root, 'ShadowMasks', img_name + '.png')))
            else:  # we set label=-1 when comes to unlebaled data
                data_list.append((os.path.join(root, 'ShadowImages', img_name + '.jpg'), -1))

    return data_list

def make_labeled_dataset(root, edge=False):
    img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'ShadowImages')) if f.endswith('.jpg')]
    label_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'ShadowMasks')) if f.endswith('.png')]
    data_list = []
    if edge:
        # for img_name in img_list:
        for img_name in label_list:
            data_list.append((os.path.join(root, 'ShadowImages', img_name + '.jpg'),
                              os.path.join(root, 'ShadowMasks', img_name + '.png'),
                              os.path.join(root, 'EdgeMasks', img_name + '.png')))
    else:
        # for img_name in img_list:
        for img_name in label_list:
            data_list.append((os.path.join(root, 'ShadowImages', img_name + '.jpg'),
                                  os.path.join(root, 'ShadowMasks', img_name + '.png')))
    return data_list


class SBU(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None, mod='union', subitizing=False, subitizing_threshold=8, subitizing_min_size_per=0.005, edge=False):
        assert (mod in ['union', 'labeled'])
        self.root = root
        self.mod = mod
        if self.mod == 'union':
            self.imgs = make_union_dataset(root, edge)
        else:
            self.imgs = make_labeled_dataset(root, edge)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
        self.edge = edge
        self.subitizing = subitizing
        # 8, 0.005
        self.subitizing_threshold = subitizing_threshold
        self.subitizing_min_size_per = subitizing_min_size_per



    def __getitem__(self, index):
        if self.edge:
            img_path, gt_path, edge_path = self.imgs[index]
        else:
            img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        if gt_path == -1: #unlabeled data
            if self.joint_transform is not None:
                img = self.joint_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            target = torch.zeros((img.shape[1], img.shape[2])).unsqueeze(0) #fake label to make sure pytorch inner check
            if self.subitizing and self.edge:
                edge = torch.zeros((img.shape[1], img.shape[2])).unsqueeze(0)
                number_per = torch.ones(1)
                sample = {'image': img, 'label': target, 'number_per': number_per, 'edge': edge}
            elif self.subitizing:
                number_per = torch.ones(1)
                sample = {'image': img, 'label': target, 'number_per': number_per}
            elif self.edge:
                edge = torch.zeros((img.shape[1], img.shape[2])).unsqueeze(0)
                sample = {'image': img, 'label': target, 'edge': edge}
            else:
                sample = {'image': img, 'label': target}
        else: #labeled data
            target = Image.open(gt_path).convert('L')
            #output subitizing knowledge for multi task learning
            if self.multi_task:
                number_per, percentage = cal_subitizing(target, threshold=self.subitizing_threshold, min_size_per=self.subitizing_min_size_per)
                number_per = torch.Tensor([number_per]) #to Tensor

            if self.joint_transform is not None:
                if self.edge:
                    edge = Image.open(edge_path).convert('L')
                    img, target, edge = self.joint_transform(img, target, edge)
                else:
                    img, target = self.joint_transform(img, target)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
                # target = target.long().squeeze(0)
                if self.edge:
                    edge = self.target_transform(edge)
                    # edge = edge.long().squeeze(0)
            if self.subitizing and self.edge:
                sample = {'image': img, 'label': target, 'number_per': number_per, 'edge': edge}
            elif self.subitizing:
                sample = {'image': img, 'label': target, 'number_per': number_per}
            elif self.edge:
                sample = {'image': img, 'label': target, 'edge': edge}
            else:
                sample = {'image': img, 'label': target}
        return sample

    def __len__(self):
        return len(self.imgs)

def relabel_dataset(dataset, edge_able=False):
    unlabeled_idxs = []
    for idx in range(len(dataset.imgs)):
        if not edge_able:
            path, label = dataset.imgs[idx]
        else:
            path, label, edge = dataset.imgs[idx]
        if label == -1:
            unlabeled_idxs.append(idx)
    labeled_idxs = sorted(set(range(len(dataset.imgs))) - set(unlabeled_idxs))

    return labeled_idxs, unlabeled_idxs

