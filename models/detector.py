import torch.nn as nn
from utils.utils_depth import depth_from_dict
from utils.utils_models import extraction_resnet


class Extract_Depth(nn.Module):
    def __init__(self, model, layers_dict_train, y_train, layers, num_classes, K=10000, bs=100, dataset='cifar10'):
        super(Extract_Depth, self).__init__()
        self.model = model
        self.y_train = y_train
        self.layers = layers
        self.num_classes = num_classes
        self.K = K
        self.layers_dict_train = layers_dict_train
        self.bs = bs
        self.dataset = dataset

    def forward(self, x):
        layers_dic_test = extraction_resnet(x, self.model, bs=self.bs, dataset=self.dataset)
        depth_dic = depth_from_dict(self.model, self.layers_dict_train, self.y_train, layers_dic_test, self.K, self.layers, self.num_classes, self.dataset)
        return depth_dic
