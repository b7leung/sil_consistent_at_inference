
import torch
import torch.nn as nn
import torchvision.models as models

class MultiviewSemanticDiscriminatorNetwork(nn.Module):

    # MVCNN inspired discirminator
    # based on https://github.com/jongchyisu/mvcnn_pytorch/blob/master/models/MVCNN.py
    def __init__(self, cfg):
        super().__init__()

        self.num_views = len(cfg["semantic_dis_training"]["dis_mv_azims"])
        self.batch_size = cfg["semantic_dis_training"]["batch_size"]
        self.img_size = cfg["semantic_dis_training"]["dis_mv_img_size"]
        self.cnn_name = cfg["semantic_dis_training"]["dis_mv_backbone"]
        self.use_resnet = self.cnn_name.startswith('resnet')
        self.pretraining = True

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512,40)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512,40)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048,1)
        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features
                self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11(pretrained=self.pretraining).features
                self.net_2 = models.vgg11(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features
                self.net_2 = models.vgg16(pretrained=self.pretraining).classifier
            
            self.net_2._modules['6'] = nn.Linear(4096,1)


    
    # input should be a tensor of shape [b, M, c, H, W]
    # b is batch size, M is number of MV images, c is channel (1 for silhouette, 3 for rgb)
    # H, W should be 224
    # output is a [b, 1] tensor of logits
    def forward(self, input):
        #print(input.shape)
        input = input.reshape(-1, input.shape[-3], input.shape[-2], input.shape[-1])
        #print(input.shape)
        feature_maps = self.net_1(input)
        #print(feature_maps.shape)
        feature_maps = feature_maps.reshape(self.batch_size, self.num_views, feature_maps.shape[-3], feature_maps.shape[-2], feature_maps.shape[-1]) # [b, M, 512, 7, 7]
        #print(feature_maps.shape)
        # elementwise max pooling across views
        feature_maps = torch.max(feature_maps, 1)[0]
        #print(feature_maps.shape)
        logits = self.net_2(feature_maps.reshape(self.batch_size, -1))
        #print(logits.shape)

        return logits

