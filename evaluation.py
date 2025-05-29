import os
import argparse

import torch

import utils
import models.resnet
import models.efficientnet
from dataloaders import dataloader

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)

def evaluate(model, model_dir, device, test_loader):
    correct = 0
    total = 0
    model.eval()
    
    save_dir =  os.path.join(model_dir, 'scores')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, 'score.txt')
    with open(save_path, 'w') as f:
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                inputs = data[0].to(device)
                target = data[1].squeeze(1).to(device)
                outputs = model(inputs)
                outputs = torch.nn.Softmax()(outputs)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                top_k = list(predicted)[-1]
                score = list(outputs.data)[-1][top_k]

                f.write('{} {} {}\n'.format(list(predicted)[0], list(target)[0], float(score)))
            acc = 100*correct/total
            f.write('Accuracy:{}%'.format(acc))
    return acc

def main(params):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_dir = params.checkpoint_dir

    val_loader = dataloader.fetch_dataloader(params.datapath, params.evallist, 1, params.num_workers, False)
    if params.model=="densenet":
        model = models.densenet.DenseNet(params.dataset_name, params.pretrained).to(device)
    elif params.model=="resnet":
        model = models.resnet.ResNet(params.dataset_name, params.pretrained).to(device)
    elif params.model=="inception":
        model = models.inception.Inception(params.dataset_name, params.pretrained).to(device) 
    elif params.model=="resnet18":
        model = models.resnet.AVENet(params.dataset_name).to(device)
        checkpoint = torch.load('H.pth.tar')
        for name, param in model.named_parameters():
            if 'audnet.fc' not in name:
                print(name)
                param = checkpoint['model_state_dict'][name]

    elif params.model=="resnet34":
        model = models.resnet.ResNet34(params.dataset_name).to(device)
    elif params.model=="resnet50":
        model = models.resnet.ResNet50(params.dataset_name).to(device)
    elif params.model=="efficientnetv2_s":
        model = models.efficientnet.EfficientNetV2_s(params.dataset_name).to(device)

    # load pretrained models
    best_model = os.path.join(model_dir, 'model_best.pth.tar')
    print('load model:{}'.format(best_model))
    checkpoint = torch.load(best_model)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    print('load pretrained model.')

    acc = evaluate(model, model_dir, device, val_loader)
    print('Accuracy:{:.2f}'.format(acc))

if __name__ == "__main__":
    args = parser.parse_args()
    params = utils.Params(args.config_path)
    main(params)
