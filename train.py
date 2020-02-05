
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import sys
import copy
from ecotaxa_dataloader import EcotaxaDataset
from uvp_dataloader import UVPDataset
from IPython import embed

torch.manual_seed(44)
random_state = np.random.RandomState(394)

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

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

def train_model(model_dict, dataloaders, optimizer, criterion, num_epochs=25, device='gpu'):
    since = time.time()

    accuracies = {'train':[], 'valid':[]}
    losses = {'train':[], 'valid':[]}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        #for phase in ['train', 'valid']:
        for phase in ['valid', 'train']:
            print('starting %s'%phase)
            model_dict = set_model_mode(model_dict, phase)
            large_running_loss = small_running_loss = 0.0
            large_running_corrects = small_running_corrects =  0
            n_batches = 0
            # Iterate over data.
            if phase == 'train':
                small_class_idx = train_ds.large_classes.index('wrong_class_size')
            else:
                small_class_idx = valid_ds.large_classes.index('wrong_class_size')
            for data in dataloaders[phase]:
                inputs, class_num, large_class_num, small_class_num, filepath, idx = data
                inputs = inputs.to(device)
                large_class_num = large_class_num.to(device)
                small_class_num = small_class_num.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    large_outputs = model_dict['large'](inputs)
                    pred_large = torch.argmax(large_outputs, dim=1)

                    pred_small = pred_large==small_class_idx
                    pred_small_inputs = inputs[pred_small]
                    pred_small_class_num = small_class_num[pred_small]
                    small_outputs = model_dict['small'](pred_small_inputs)
                    # determine where large model predicted the class was small
                    #small_inputs
                    large_loss = criterion(large_outputs, large_class_num)
                    small_loss = criterion(small_outputs, pred_small_class_num)
                    loss = large_loss + small_loss

                    _, small_preds = torch.max(small_outputs, 1)
                    large_seen += large_outputs.shape[0]
                    small_seen += small_outputs.shape[0]

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                n_batches += 1
                large_running_loss += large_loss.item() * inputs.shape[0]
                large_running_corrects += torch.sum(pred_large == large_class_num.data).cpu().numpy()
                small_running_loss += small_loss.item() * pred_small_inputs.shape[0]
                small_running_corrects += torch.sum(pred_small == small_class_num[pred_small].data).cpu().numpy()

            large_epoch_loss = large_running_loss.item() / large_seen
            large_epoch_acc = large_running_corrects.double().cpu().numpy() / large_seen
            small_epoch_loss = small_running_loss.item() / small_seen
            small_epoch_acc = small_running_corrects.double().cpu().numpy() / small_seen


            print('LARGE {} Loss: {:.4f} Acc: {:.4f}'.format(phase, large_epoch_loss, large_epoch_acc))
            print('SMALL {} Loss: {:.4f} Acc: {:.4f}'.format(phase, small_epoch_loss, small_epoch_acc))
            print('used %s batches for calculation'%n_batches)
            accuracies['large_'+phase].append(large_epoch_acc)
            losses['large_'+phase].append(large_epoch_loss)
            accuracies['small_'+phase].append(small_epoch_acc)
            losses['small_'+phase].append(small_epoch_loss)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    ## load best model weights
    return model_dict, optimizer, accuracies, losses

if __name__ == '__main__':
    load_model = ''
    #load_model = 'experiments/most_merged/checkpoints/ckptwt00120.pt
    """
     without rotate - seems to overfit badly
    """
    #name = 'uvp_big_1000small_noliving_rotate_other_bg0'
    small = True
    name = 'uvp_warmup'
    exp_dir = os.path.join('experiments', name)
    datadir = './'
    write_dir = os.path.join('experiments', name, 'checkpoints')
    batch_size = 64
    train_ds = UVPDataset(csv_file=os.path.join('experiments', name, 'train.csv'), seed=34, valid=False)
    class_names = train_ds.classes
    large_class_names = train_ds.large_classes
    small_class_names = train_ds.small_classes
    class_counts = train_ds.class_counts
    class_weights = train_ds.weights
    valid_ds = UVPDataset(csv_file=os.path.join('experiments', name, 'valid.csv'), seed=334, valid=True, classes=class_names, weights=class_weights)
    #train_weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.FloatTensor(train_ds.img_weights), len(train_ds), replacement=True)
    #valid_weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.FloatTensor(valid_ds.img_weights), len(valid_ds), replacement=True)
    train_weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.FloatTensor(train_ds.img_weights), len(train_ds), replacement=False)
    valid_weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.FloatTensor(valid_ds.img_weights), len(valid_ds), replacement=False)
    train_dl = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_weighted_sampler,
            num_workers=4,
        )
    valid_dl = torch.utils.data.DataLoader(
            valid_ds,
            batch_size=batch_size,
            sampler=valid_weighted_sampler,
            num_workers=2,
        )
    dataloaders = {'train':train_dl, 'valid':valid_dl}

    # Load the pretrained model from pytorch
    large_model = torchvision.models.resnet50(pretrained=True)
    small_model = torchvision.models.resnet50(pretrained=True)
    # Number of classes in the dataset
    large_num_classes = len(large_class_names)
    small_num_classes = len(small_class_names)

    # Number of epochs to train for
    num_epochs = 35

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    # feature_extract = False
    # print(rmodel)
    # last layer: (fc): Linear(in_features=2048, out_features=1000, bias=True)
    # need to reshape
    large_model.fc = nn.Linear(2048, large_num_classes)
    large_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    small_model.fc = nn.Linear(2048, small_num_classes)
    small_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using', device)
    model_dict = {'large':large_model, 'small':small_model}

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = list(large_model.parameters()) + list(small_model.parameters())
    # Observe that all parameters are being optimized
    optimizer = optim.Adam(params_to_update, lr=1e-4)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))
    num_epochs_bt_saves = 1

    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    all_accuracy = {'large_train':[], 'large_valid':[], 'small_train':[], 'small_valid':[]}
    all_losses = {'large_train':[], 'large_valid':[], 'small_train':[], 'small_valid':[]}
    cnt_start = 0
    epoch_cnt = 0
    if load_model != '':
        print('--------------------loading from %s'%load_model)
        save_dict = torch.load(load_model)
        for name in model_dict.keys():
            model_dict[name].load_state_dict[name]
        cnt_start = int(save_dict['cnt'])
        print("starting from cnt", cnt_start)
        cnt_start = save_dict['cnt']
        all_accuracy = save_dict['accuracy']
        all_losses = save_dict['loss']
        epoch_cnt = len(all_losses['large_train'])
        print("have seen %s epochs"%epoch_cnt)
        try:
            optimizer.load_state_dict(save_dict['opt'])
        except:
            print('could not load opt state dict')

    for name in model_dict.keys():
        model_dict[name].to(device)
    for cnt in range(cnt_start, cnt_start+1000):
        # Train and evaluate
        print("starting cnt sequence", cnt)
        pp = {
              'opt':optimizer.state_dict(),
              'accuracy':all_accuracy,
              'loss':all_losses,
              'cnt':cnt,
              'num_epochs_bt_saves':num_epochs_bt_saves,
              'epoch_cnt': epoch_cnt,
               }
        for name in model_dict.keys():
            pp[name] = model_dict[name].state_dict()

        cpath = os.path.join(write_dir, 'ckptwt_eval%05d.pt'%len(all_losses['large_train']))
        print("saving model", cpath)
        torch.save(pp, cpath)
        model_dict, optimizer, accuracies, losses = train_model(model_dict, dataloaders, optimizer, criterion, num_epochs=num_epochs_bt_saves, device=device)
        for phase in all_accuracy.keys():
            all_accuracy[phase].extend(accuracies[phase])
        for phase in all_losses.keys():
            all_losses[phase].extend(losses[phase])


