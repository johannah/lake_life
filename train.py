
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

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        #for phase in ['train', 'valid']:
        for phase in ['valid', 'train']:
            model_dict = set_model_mode(model_dict, phase)
            model.train()  # Set model to training mode
            running_loss = 0.0
            running_corrects = 0
            n_batches = 0
            # Iterate over data.
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
                    large_loss = criterion(large_outputs, large_class_num)
                    embed()

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                n_batches += 1
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == class_num.data)

                ncorr = torch.sum(preds == class_num.data).cpu().numpy()
                pcorr = ncorr/float(outputs.shape[0])

                #if pcorr < 0.6:
                #    print('---------------------------')
                #    print(phase, n_batches, pcorr)
                #    print(sorted(list((class_num.cpu().numpy()))))
                #    print(class_num.cpu().numpy())
                #    print(outputs.argmax(1).detach().cpu().numpy())
                #    print('correct %s/%s'%(ncorr, outputs.shape[0]))

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)


            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('used %s batches for calculation'%n_batches)
            accuracies[phase].append(epoch_acc)
            losses[phase].append(epoch_loss)
            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    ## load best model weights
    #model.load_state_dict(best_model_wts)
    return model, optimizer, accuracies, losses, best_model_wts

if __name__ == '__main__':
    load_model = ''
    #load_model = 'experiments/most_merged/checkpoints/ckptRDD00130.pth'
    # goes bad after 120
    #load_model = 'experiments/most_merged/checkpoints/ckptwt00120.pth'
    #load_model = 'experiments/most_merged/checkpoints/ckptwt_eval00245.pth'
    #load_model = 'experiments/most_merged/checkpoints/ckptwt00120.pth'
    """
     without rotate - seems to overfit badly
    """
    #name = 'uvp_big_1000small_noliving_rotate_other_bg0'
    small = True
    name = 'uvp_big_1000small_noliving_rotate_bg0_trim_combine_new'
    exp_dir = os.path.join('experiments', name)
    if small:
        small_name = name+'_small'
        small_exp_dir = os.path.join('experiments', small_name)
        if not os.path.exists(small_exp_dir):
            os.makedirs(small_exp_dir)
        for phase in ['train', 'valid']:
            cmd = 'cp %s %s' %(
                os.path.join(exp_dir, '%s.csv' %phase),
                os.path.join(small_exp_dir, '%s.csv'%phase))
            print(cmd)
            os.system(cmd)
        name = small_name
        exp_dir = small_exp_dir
    datadir = './'

    write_dir = os.path.join('experiments', name, 'checkpoints')
    batch_size = 64
    train_ds = UVPDataset(csv_file=os.path.join('experiments', name, 'train.csv'), seed=34, valid=False, small=small)
    class_names = train_ds.classes
    large_class_names = train_ds.large_classes
    small_class_names = train_ds.small_classes
    class_counts = train_ds.class_counts
    class_weights = train_ds.weights
    valid_ds = UVPDataset(csv_file=os.path.join('experiments', name, 'valid.csv'), seed=334, valid=True, classes=class_names, weights=class_weights, small=small)
    train_weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.FloatTensor(train_ds.img_weights), len(train_ds), replacement=True)
    valid_weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.FloatTensor(valid_ds.img_weights), len(valid_ds), replacement=True)
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
    params_to_update = list(large_model.parameters()) + list(small_model.paramters)
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
    best_model_wts = None
    if load_model != '':
        print('--------------------loading from %s'%load_model)
        save_dict = torch.load(load_model)
        for name in model_dict.keys():
            model_dict[name].load_state_dict[name]
        cnt_start = int(save_dict['cnt'])
        print("starting from cnt", cnt_start)
        best_model_wts = save_dict['best_model_wts']
        cnt_start = save_dict['cnt']
        all_accuracy = save_dict['accuracy']
        all_losses = save_dict['loss']
        epoch_cnt = len(all_losses['train'])
        print("have seen %s epochs"%epoch_cnt)
        try:
            optimizer.load_state_dict(save_dict['opt'])
        except:
            print('could not load opt state dict')

    for name in model_dict.keys():
        model_dict.to(device)
    for cnt in range(cnt_start, cnt_start+1000):
        # Train and evaluate
        print("starting cnt sequence", cnt)
        pp = {
              'opt':optimizer.state_dict(),
              'accuracy':all_accuracy,
              'loss':all_losses,
              'cnt':cnt,
              'best_model_wts':best_model_wts,
              'num_epochs_bt_saves':num_epochs_bt_saves,
              'epoch_cnt': epoch_cnt,
               }
        for name in model_dict.keys():
            state_dict[name] = model_dict[name].state_dict()

        cpath = os.path.join(write_dir, 'ckptwt_eval%05d.pth'%len(all_losses['train']))
        print("saving model", cpath)
        torch.save(pp, cpath)
        model_dict, optimizer, accuracies, losses, best_model_wts = train_model(model_dict, dataloaders, optimizer, criterion, num_epochs=num_epochs_bt_saves, device=device)
        for phase in all_accuracy.keys():
            all_accuracy[phase].extend(accuracies[phase])
        for phase in all_losses.keys():
            all_losses[phase].extend(losses[phase])


