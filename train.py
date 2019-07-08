
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
from IPython import embed

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")
random_state = np.random.RandomState(394)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def train_model(model, dataloaders, optimizer, criterion, num_epochs=25, device='gpu'):
    since = time.time()

    accuracies = {'train':[], 'valid':[]}
    losses = {'train':[], 'valid':[]}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0


            # Iterate over data.
            for inputs, labels, filepaths, classnames in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

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
    #load_model = 'experiments/most_and_balanced/checkpoints/ckpt00020.pth'
    load_model = ''
    name = 'most_and_balanced'
    datadir = './'

    write_dir = os.path.join('experiments', name, 'checkpoints')
    batch_size = 32
    train_ds = EcotaxaDataset(csv_file=os.path.join('experiments', name, 'train.csv'),seed=34)
    class_names = train_ds.classes
    class_counts = train_ds.class_counts
    class_weights = train_ds.weights
    valid_ds = EcotaxaDataset(csv_file=os.path.join('experiments', name, 'valid.csv'), seed=334, classes=class_names, weights=class_weights)
    for cn, cc, cw in zip(class_names, class_counts, class_weights):
        print('class:%s counts:%s weight:%.03f'%(cn, cc, cw))
    train_weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.FloatTensor(train_ds.img_weights), len(train_ds))
    valid_weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.FloatTensor(valid_ds.img_weights), len(valid_ds))
    train_dl = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_weighted_sampler,
            num_workers=2,
        )
    valid_dl = torch.utils.data.DataLoader(
            valid_ds,
            batch_size=batch_size,
            sampler=valid_weighted_sampler,
            num_workers=2,
        )
    dataloaders = {'train':train_dl, 'valid':valid_dl}


    # Load the pretrained model from pytorch
    rmodel = torchvision.models.resnet50(pretrained=True)
    num_epochs = 1000
    # Number of classes in the dataset
    num_classes = len(class_names)

    # Number of epochs to train for
    num_epochs = 15

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = False
    # print(rmodel)
    # last layer: (fc): Linear(in_features=2048, out_features=1000, bias=True)
    # need to reshape
    rmodel.fc = nn.Linear(2048, num_classes)
    rmodel.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using', device)
    rmodel = rmodel.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = rmodel.parameters()
    #print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in rmodel.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                #print("\t",name)
#    else:
#        for name,param in rmodel.named_parameters():
#            if param.requires_grad == True:
#                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(params_to_update, lr=1e-3)

    # Setup the loss fxn
    #criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))
    num_epochs_bt_saves = 10

    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    all_accuracy = {'train':[], 'valid':[]}
    all_losses = {'train':[], 'valid':[]}
    cnt_start = 1
    best_model_wts = None
    if load_model != '':
        print('loading from %s'%load_model)
        save_dict = torch.load(load_model)
        rmodel.load_state_dict(save_dict['state_dict'])
        rmodel.to(device)
        cnt_start = int(save_dict['cnt']/float(num_epochs_bt_saves))
        best_model_wts = save_dict['best_model_wts']
        try:
            all_accuracy = save_dict['accuracies']
            all_losses = save_dict['losses']
        except:
            print("could not load histories")
        try:
            optimizer.load_state_dict(save_dict['opt'])
            optimizer.to(device)
        except:
            print('could not load opt state dict')

    for cnt in range(cnt_start, cnt_start+1000):
        # Train and evaluate
        print("starting cnt sequence", cnt)
        pp = {'state_dict':rmodel.state_dict(),
              'opt':optimizer.state_dict(),
              'accuracy':all_accuracy,
              'loss':all_losses,
              'cnt':cnt*num_epochs_bt_saves,
              'best_model_wts':best_model_wts,
              'num_epochs_bt_saves':num_epochs_bt_saves,
               }
        cpath = os.path.join(write_dir, 'ckpt%05d.pth'%(cnt*num_epochs_bt_saves))
        print("saving model", cpath)
        torch.save(pp, cpath)

        rmodel, optimizer, accuracies, losses, best_model_wts = train_model(rmodel, dataloaders, optimizer, criterion, num_epochs=num_epochs_bt_saves, device=device)
        for phase in all_accuracy.keys():
            all_accuracy[phase].extend(accuracies[phase])
        for phase in all_losses.keys():
            all_losses[phase].extend(losses[phase])


