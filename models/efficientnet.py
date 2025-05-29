import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetV2_s(nn.Module):
    """
    FPS:
        - 730 : 139.43
        - 630 : 30.49
    """
    def __init__(self, dataset, pretrained=True):
            super(EfficientNetV2_s, self).__init__()
            num_classes = 12
            self.model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
            # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.classifier[1] = torch.nn.Linear(1280, num_classes, bias=True)
    
    def forward(self, x):
            output = self.model(x)
            return output

class EfficientNet_b3(nn.Module):
    """
    FPS:
        - 730 : 165.68
        - 630 : 29.42
    """
    def __init__(self, dataset, pretrained=True):
            super(EfficientNet_b3, self).__init__()
            num_classes = 12
            self.model = models.efficientnet_b3(weights='IMAGENET1K_V1')
            # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.classifier[1] = torch.nn.Linear(1536, num_classes, bias=True)
    
    def forward(self, x):
            output = self.model(x)
            return output

class EfficientNet_b1(nn.Module):
    """
    FPS:
        - 730 : 276.30
        - 630 : 48.95
    """
    def __init__(self, dataset, pretrained=True):
            super(EfficientNet_b1, self).__init__()
            num_classes = 12
            self.model = models.efficientnet_b1(weights='IMAGENET1K_V1')
            # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.classifier[1] = torch.nn.Linear(1280, num_classes, bias=True)
    
    def forward(self, x):
            output = self.model(x)
            print(output.shape)
            return output

class EfficientNet_b0(nn.Module):
    """
    FPS:
        - 730 : 374.53
        - 630 : 69.03
    """
    def __init__(self, dataset, pretrained=True):
            super(EfficientNet_b0, self).__init__()
            num_classes = 12
            self.model = models.efficientnet_b0(weights='IMAGENET1K_V1')
            # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.classifier[1] = torch.nn.Linear(1280, num_classes, bias=True)
    
    def forward(self, x):
            output = self.model(x)
            print(output.shape)
            return output

if __name__ == "__main__":
    model = EfficientNet_b0("baby_crying")
    x = torch.randn(1, 3, 128, 250, requires_grad=False)
    # print(model)
    # Export the model
    torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "efficientnet_b0.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'])
       