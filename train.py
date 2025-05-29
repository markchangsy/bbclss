import torch
import torch.nn as nn
import utils
import validate
import argparse
import models.densenet
import models.resnet
import models.inception
import models.resnet18
import models.efficientnet

from dataloaders import dataloader
from tqdm import tqdm
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)


def train(model, device, data_loader, optimizer, loss_fn):
    model.train()
    loss_avg = utils.RunningAverage()

    with tqdm(total=len(data_loader)) as t:
        for batch_idx, data in enumerate(data_loader):
            inputs = data[0].to(device)
            target = data[1].squeeze(1).to(device)

            outputs = model(inputs)

            loss = loss_fn(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
    return loss_avg()


def train_and_evaluate(model, device, train_loader, val_loader, optimizer, loss_fn, writer, params, scheduler=None):
    best_acc = 0.0

    for epoch in range(params.epochs):
        avg_loss = train(model, device, train_loader, optimizer, loss_fn)

        acc = validate.evaluate(model, device, val_loader)
        print("Epoch {}/{} Loss:{} Valid Acc:{}".format(epoch, params.epochs, avg_loss, acc))

        is_best = (acc > best_acc)
        if is_best:
            best_acc = acc
        if scheduler:
            scheduler.step()

        utils.save_checkpoint({"epoch": epoch + 1,
                               "model": model.state_dict(),
                               "optimizer": optimizer.state_dict()}, is_best, "{}".format(params.checkpoint_dir))
        writer.add_scalar("trainingLoss", avg_loss, epoch)
        writer.add_scalar("valLoss", acc, epoch)
    writer.close()
    return best_acc

if __name__ == "__main__":
    args = parser.parse_args()
    params = utils.Params(args.config_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_acc = 0
            
    train_loader = dataloader.fetch_dataloader(params.datapath, params.trainlist, params.batch_size, params.num_workers, True)
    val_loader = dataloader.fetch_dataloader(params.datapath, params.vallist, params.batch_size, params.num_workers, False)

    writer = SummaryWriter(params.summaries_dir, comment=params.dataset_name)
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

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    if params.scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)
    else:
        scheduler = None

    acc = train_and_evaluate(model, device, train_loader, val_loader, optimizer, loss_fn, writer, params, scheduler)
    print('validation accuracy: {} %'.format(acc))

