
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import sys
from IPython import embed
from config import exp_dir, exp_name, get_checkpoints_dir
from utils import get_model, get_dataset, save_model, set_model_mode

torch.manual_seed(44)
random_state = np.random.RandomState(394)

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

def forward_pass(model_dict, inputs, large_class_num, small_class_num, device, belongs_in_small_class_idx):
    inputs = inputs.to(device)
    large_class_num = large_class_num.to(device)
    small_class_num = small_class_num.to(device)
    # Get large model output
    large_outputs = model_dict['large'](inputs)
    # get predictions from softmax
    large_predictions = torch.argmax(large_outputs, dim=1)
    # determine where the large model predicted the label to
    # belong in a small class
    pred_small = [x for x in range(large_predictions.shape[0]) if large_predictions[x]==belongs_in_small_class_idx or large_class_num[x]==belongs_in_small_class_idx or random_state.rand()<.15]
    if not len(pred_small):
        # if no fit criteria  - put them all thru
        pred_small = range(large_predictions.shape[0])
    pred_small_inputs = inputs[pred_small]
    pred_small_class_num = small_class_num[pred_small]
    small_outputs = model_dict['small'](pred_small_inputs)
    # determine where large model predicted the class was small
    _, small_predictions = torch.max(small_outputs, 1)
    return model_dict, large_outputs, small_outputs, pred_small_class_num, large_predictions, small_predictions

def train_model(model_dict, dataloaders, all_accuracy, all_losses, optimizer, criterion, num_epochs=25, device='gpu'):
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        #for phase in ['train', 'valid']:
        for phase in ['valid', 'train']:
            large_seen =  small_seen = 0
            print('starting %s'%phase)
            model_dict = set_model_mode(model_dict, phase)
            large_running_loss = small_running_loss = 0.0
            large_running_corrects = small_running_corrects =  0
            n_batches = 0
            # Iterate over data.
            belongs_in_small_class_idx = dataloaders[phase+'_large_wrong_class_idx']
            for data in dataloaders[phase]:
                inputs, class_num, large_class_num, small_class_num, filepath, idx = data
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    out = forward_pass(model_dict, inputs, large_class_num, small_class_num, belongs_in_small_class_idx, device, belongs_in_small_class_idx)
                    model_dict, large_outputs, small_outputs, pred_small_class_num, large_predictions, small_predictions  = out
                    #small_inputs
                    large_loss = criterion(large_outputs, large_class_num)
                    small_loss = criterion(small_outputs, pred_small_class_num)
                    loss = large_loss + small_loss

                    large_seen += large_outputs.shape[0]
                    small_seen += small_outputs.shape[0]

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                n_batches += 1
                large_running_loss += large_loss * inputs.shape[0]
                large_running_corrects += torch.sum(large_predictions == large_class_num.data).cpu().numpy()
                small_running_loss += small_loss * pred_small_inputs.shape[0]
                small_running_corrects += torch.sum(small_predictions == small_class_num[pred_small].data).cpu().numpy()

            large_epoch_loss = large_running_loss.item() / large_seen
            large_epoch_acc = large_running_corrects / large_seen
            small_epoch_loss = small_running_loss.item() / small_seen
            small_epoch_acc = small_running_corrects / small_seen

            print('LARGE {} Loss: {:.4f} Acc: {:.4f}'.format(phase, large_epoch_loss, large_epoch_acc))
            print('SMALL {} Loss: {:.4f} Acc: {:.4f}'.format(phase, small_epoch_loss, small_epoch_acc))
            print('used %s batches for calculation'%n_batches)
            all_accuracy['large_'+phase].append(large_epoch_acc)
            all_losses['large_'+phase].append(large_epoch_loss)
            all_accuracy['small_'+phase].append(small_epoch_acc)
            all_losses['small_'+phase].append(small_epoch_loss)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    ## load best model weights
    return model_dict, optimizer, all_accuracy, all_losses

if __name__ == '__main__':
    """
     without rotate - seems to overfit badly
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = ''
    checkpoints_dir = get_checkpoints_dir()
    #model_path = 'experiments/most_merged/checkpoints/ckptwt00120.pt
    #name = 'uvp_big_1000small_noliving_rotate_other_bg0'
    name = 'uvp_warmup'
    datadir = './'
    batch_size = 64
    # Number of epochs to train for
    num_epochs = 35

    dataloaders, large_class_names, small_class_names = get_dataset(exp_dir, batch_size=batch_size)
    # Load the pretrained model from pytorch
    # Number of classes in the dataset
    large_num_classes = len(large_class_names)
    small_num_classes = len(small_class_names)
    model_dict, cnt_start, epoch_cnt, all_accuracy, all_losses = get_model(model_path, large_num_classes, small_num_classes, device)

    params_to_update = list(model_dict['large'].parameters()) + list(model_dict['small'].parameters())
    # Observe that all parameters are being optimized
    optimizer = optim.Adam(params_to_update, lr=1e-4)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))
    num_epochs_bt_saves = 1

    for cnt in range(cnt_start, cnt_start+1000):
        # Train and evaluate
        save_model(checkpoints_dir, cnt, model_dict, optimizer, all_accuracy, all_losses, num_epochs_bt_saves, epoch_cnt)
        model_dict, optimizer, all_accuracy, all_losses = train_model(model_dict, dataloaders, all_accuracy, all_losses, optimizer, criterion, num_epochs=num_epochs_bt_saves, device=device)


