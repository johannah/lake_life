import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
import os
import sys
from ecotaxa_dataloader import EcotaxaDataset
from uvp_dataloader import UVPDataset
from IPython import embed

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def set_model_mode(model_dict, phase):
    for name in model_dict.keys():
        if phase == 'train':
            model_dict[name].train()
        else:
            model_dict[name].eval()
    return model_dict

def get_model(model_path, num_classes, num_last_layer_features, device):
    from config import adaptive_sm_cutoffs
    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    # feature_extract = False
    # print(rmodel)
    # last layer: (fc): Linear(in_features=2048, out_features=1000, bias=True)
    # need to reshape
    # need to reshape
    model = models.resnet50(pretrained=True)
    #model.fc = nn.Linear(num_last_layer_features, num_classes)
    model.fc = nn.Linear(2048, num_last_layer_features)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    criterion = nn.AdaptiveLogSoftmaxWithLoss(in_features=num_last_layer_features, n_classes=num_classes, cutoffs=adaptive_sm_cutoffs)
    model_dict = {'model':model, 'criterion':criterion}
    for name in model_dict.keys():
        model_dict[name].to(device)
    if model_path != '':
        print('--------------------loading from %s'%model_path)
        save_dict = torch.load(model_path)
        for name in model_dict.keys():
            model_dict[name].load_state_dict(save_dict[name])
        cnt_start = int(save_dict['cnt'])
        print("starting from cnt", cnt_start)
        cnt_start = save_dict['cnt']
        all_accuracy = save_dict['accuracy']
        all_losses = save_dict['loss']
        epoch_cnt = len(all_losses['train'])
        print("have seen %s epochs"%epoch_cnt)
    else:
        all_accuracy = {'train':[]}
        all_losses = {'train':[]}
        cnt_start = 0
        epoch_cnt = 0
    return model_dict, cnt_start, epoch_cnt, all_accuracy, all_losses

def get_dataset(dataset_base_path, batch_size, num_workers=4, evaluation=False, limit=1e6, img_size=225):
    train_ds = UVPDataset(csv_file=os.path.join(dataset_base_path, 'train.csv'), seed=34, valid=False, limit=limit, img_size=img_size)
    class_names = train_ds.classes
    #class_weights = train_ds.weights
    #valid_ds = UVPDataset(csv_file=os.path.join(dataset_base_path, 'valid.csv'), seed=334, valid=True, classes=class_names, weights=class_weights)

    #train_weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.FloatTensor(train_ds.img_weights), len(train_ds), replacement=True)
    #valid_weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.FloatTensor(valid_ds.img_weights), len(valid_ds), replacement=True)
    # when evaluation - i want replacement = False
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.FloatTensor(train_ds.img_weights), len(train_ds), replacement=not evaluation)
    #valid_sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.FloatTensor(valid_ds.img_weights), len(valid_ds), replacement=not evaluation)
    #train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=np.arange(len(train_ds)))
    #valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=np.arange(len(valid_ds)))

    train_dl = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=train_sampler,
        )
    #valid_dl = torch.utils.data.DataLoader(
    #        valid_ds,
    #        batch_size=batch_size,
    #        num_workers=max([1, num_workers//2]),
    #        sampler=valid_sampler,
    #    )

    #dataloaders = {'train':train_dl, 'valid':valid_dl}
    #dataloaders = {'train':train_dl, 'valid':valid_dl}
    dataloaders = {'train':train_dl}
    return dataloaders, class_names

def save_model(write_dir, cnt, model_dict, optimizer, all_accuracy, all_losses, epoch_cnt):
    print("starting cnt sequence", cnt)
    pp = {
          'opt':optimizer.state_dict(),
          'loss':all_losses,
          'accuracy':all_accuracy,
          'cnt':cnt,
          'epoch_cnt': epoch_cnt,
           }
    for name in model_dict.keys():
        pp[name] = model_dict[name].state_dict()

    cpath = os.path.join(write_dir, 'ckptwt_eval%05d.pt'%len(all_losses['train']))
    print("saving model", cpath)
    torch.save(pp, cpath)


