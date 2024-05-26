import torch
import torch.nn as nn
import torchvision
from math import sqrt
import numpy as np

def get_model(cfg):
    """
    args:
        cfg: configuration dict for model_params
    
    Calls the appropriate model based on the name in the configuration file.
    """
    if cfg.name == "1layer":
        return LinearModel(cfg)
    elif cfg.name == "combresnet":
        return CombRenset18(cfg)
    elif cfg.name == "resnet18":
        return Resnet18(cfg)
    elif cfg.name == "resnet34":
        return Resnet34(cfg)
    elif cfg.name == "resnet18_pm":
        return Resnet18_PM(cfg)
    elif cfg.name == "portfolio":
        return portfolioModel(cfg)
    else:
        raise NotImplementedError("Model {} not implemented".format(cfg.model))

class LinearModel(nn.Module):
    def __init__(self, cfg):
        super(LinearModel, self).__init__()
        """
        args:
            cfg: configuration dict for model_params
        
        This function defines a linear model with input_dim and output_dim. If extra flags, 
        softmax, abs, softplus are set, then the model will have those layers.
        """
        self.input_dim = cfg.input_dim
        self.output_dim = cfg.output_dim
        self.cfg = cfg
        model = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim, bias=True),
            # nn.ReLU(),
            # nn.Linear(self.output_dim, self.output_dim),
        )
        if cfg.softmax:
            model.add_module("softmax", nn.Softmax(dim=1))
        self.model = model
    
    def forward(self, x):
        x = self.model(x)
        if self.cfg.abs:
            x = torch.abs(x)
        if self.cfg.softplus:
            x = nn.functional.softplus(x)
        return x

class CombRenset18(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        """
        args: 
            cfg: configuration dict for model_params
        
        This function defines a resnet18 model with input_dim and output_dim. It uses the same
        architecture from the Identity codebase. 
        
        Two differences: 
        1) last layer output_dim  is higher than identity as number_edges we are considering 
                is higher than number_nodes.
        2) adaptive maxpool is replaced with a linear layer.


        """
        out_features = cfg.output_dim
        in_channels = cfg.input_dim
        self.resnet_model = torchvision.models.resnet18(pretrained=False, num_classes=out_features)
        del self.resnet_model.conv1
        self.resnet_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        output_shape = (int(sqrt(out_features)), int(sqrt(out_features)))
        # self.pool = nn.AdaptiveMaxPool2d(output_shape)
        self.pool = nn.Linear(64*24, out_features, bias=True) ###changed in our code compared to identity
        #self.last_conv = nn.Conv2d(128, 1, kernel_size=1,  stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        args:
            x: input tensor of shape (batch_size, input_dim)
        
        This function takes in the input tensor and returns the output tensor of shape
        (batch_size, output_dim)
        """
        x = self.resnet_model.conv1(x)
        x = self.resnet_model.bn1(x)
        x = self.resnet_model.relu(x)
        x = self.resnet_model.maxpool(x)
        # print(x.shape)
        x = self.resnet_model.layer1(x)
        x = self.resnet_model.layer2(x)
        x = self.resnet_model.layer3(x)
        # x = self.last_conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.pool(x)
        # x = self.relu(x)
        # x = torch.abs(x)
        x = nn.functional.softplus(x)
        # print(x.shape)
        # x = x.mean(dim=1)
        return x


class Resnet18(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        """
        args: 
            cfg: configuration dict for model_params
        
        Using Resnet to train model
        """
        self.cfg = cfg
        out_features = cfg.output_dim
        in_channels = cfg.input_dim
        self.resnet_model = torchvision.models.resnet18(pretrained=False, num_classes=out_features)
        
    def forward(self, x):
        """
        args:
            x: input tensor of shape (batch_size, input_dim)
        
        This function takes in the input tensor and returns the output tensor of shape
        (batch_size, output_dim)
        """
        x = x.transpose(1,3)
        x = self.resnet_model(x)
        if self.cfg.softplus:
            x = nn.functional.softplus(x)
        return x

class Resnet18_PM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        """
        args: 
            cfg: configuration dict for model_params
        
        Using Resnet to train model
        """
        self.cfg = cfg
        out_features = cfg.output_dim
        in_channels = cfg.input_dim
        self.resnet_model = torchvision.models.resnet18(num_classes=out_features, pretrained=False)
        self.resnet_model.conv1 = torch.nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)

        self.M = None
        # if "load_e2v" in cfg:
        #     data_path = "../dataset/mnist_matching/12x12_mnist_nonunique/ver_2_edge.npy"
        #     self.M = torch.from_numpy(np.load(data_path)).float()
        #     self.M = self.M.to(self.resnet_model.conv1.weight.device)
        #     self.M.requires_grad = False

    def forward(self, x):
        """
        args:
            x: input tensor of shape (batch_size, input_dim)
        
        This function takes in the input tensor and returns the output tensor of shape
        (batch_size, output_dim)
        """
        # x = x.transpose(1,3)
        x = self.resnet_model(x)
        if self.cfg.softplus:
            x = nn.functional.softplus(x)
        if self.M is not None:
            self.M = self.M.to(x.device)
            x = torch.matmul(x, self.M)
        return x

class Resnet34(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        """
        args: 
            cfg: configuration dict for model_params
        
        Using Resnet to train model
        """
        self.cfg = cfg
        out_features = cfg.output_dim
        in_channels = cfg.input_dim
        self.resnet_model = torchvision.models.resnet34(pretrained=False, num_classes=out_features)
        
    def forward(self, x):
        """
        args:
            x: input tensor of shape (batch_size, input_dim)
        
        This function takes in the input tensor and returns the output tensor of shape
        (batch_size, output_dim)
        """
        x = x.transpose(1,3)
        x = self.resnet_model(x)
        if self.cfg.softplus:
            x = nn.functional.softplus(x)
        return x

class portfolioModel(nn.Module):
    def __init__(self, cfg):
        super(portfolioModel, self).__init__()
        """
        args:
            cfg: configuration dict for model_params
        
        This function defines a linear model with input_dim and output_dim. If extra flags, 
        softmax, abs, softplus are set, then the model will have those layers.
        """
        self.input_dim = cfg.input_dim
        self.output_dim = cfg.output_dim
        self.qp_output_dim = cfg.qp_output_dim
        self.cfg = cfg
        if self.cfg.convex:
            model = nn.Sequential(
                nn.Conv1d(self.input_dim, self.qp_output_dim, self.qp_output_dim, bias=True),
                )
            # print("running the convex model", self.cfg.convex)
            # exit()
        else:
            model = nn.Sequential(
                    nn.Conv1d(self.input_dim, 4*self.input_dim, 1, bias=True),
                    nn.ReLU(),
                    nn.Conv1d(4*self.input_dim, self.output_dim, 1, bias=True),
                )
            # print("Not running the convex model", self.cfg.convex)
            # exit()
            
    
        self.model = model

        qp_model = nn.Sequential(
            # nn.Conv1d(self.input_dim, self.qp_output_dim, 1, bias=True),
            # nn.ReLU(),
            # nn.Conv1d(self.qp_output_dim, self.qp_output_dim*self.qp_output_dim, self.qp_output_dim, bias=True),
            nn.Conv1d(self.input_dim, self.qp_output_dim*self.qp_output_dim, self.qp_output_dim, bias=True),
        )
        self.qp_model = qp_model
    
    def forward(self, x):
        x = x.transpose(1,2)
        c = self.model(x)
        c = c.squeeze()
        # c = nn.functional.softplus(c)
        
        q = self.qp_model(x)
        q = q.squeeze(2)
        q = q.reshape(-1, self.qp_output_dim, self.qp_output_dim)
        # q = nn.functional.softplus(q)
        q_T = torch.transpose(q, 1, 2)
        q = torch.bmm(q_T, q) + 1e-4*torch.eye(q.size(1))
        return [c, q]
        # else:
        #     return [c, None]