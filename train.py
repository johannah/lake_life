
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
from config import exp_dir, exp_name, get_checkpoints_dir, adaptive_sm_cutoffs
from utils import get_model, get_dataset, save_model, set_model_mode

torch.manual_seed(44)
random_state = np.random.RandomState(394)

def forward_pass(model_dict, inputs, class_num, device):
    inputs = inputs.to(device)
    class_num = class_num.to(device)
    # Get large model output
    model_output = model_dict['model'](inputs)
    # get predictions from softmax
    _, loss = model_dict['criterion'](model_output, class_num)
    predictions = model_dict['criterion'].predict(model_output)
    return model_dict, class_num, predictions, loss

def train_model(model_dict, dataloaders, all_accuracy, all_losses, optimizer, device='gpu'):
    since = time.time()
    # Each epoch has a training and validation phase
    for phase in ['train']:
        num_seen = 0
        print('starting %s'%phase)
        model_dict = set_model_mode(model_dict, phase)
        running_loss = 0.0
        running_corrects = 0.0
        n_batches = 0
        # Iterate over data.
        for data in dataloaders[phase]:
            images, class_num, filepath, idx = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            with torch.set_grad_enabled(phase == 'train'):
                model_dict, class_num, predictions, loss = forward_pass(model_dict, images, class_num, device)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            bs = images.shape[0]
            n_batches += 1
            num_seen +=bs
            running_loss += loss * bs
            running_corrects += torch.sum(predictions == class_num.data).cpu().numpy()


        epoch_loss = running_loss.item() / num_seen
        epoch_acc = running_corrects / num_seen

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        print('used %s batches for calculation'%n_batches)
        all_accuracy[phase].append(epoch_acc)
        all_losses[phase].append(epoch_loss)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    ## load best model weights
    return model_dict, optimizer, all_accuracy, all_losses, num_seen

if __name__ == '__main__':
    """
     without rotate - seems to overfit badly
     TODO - copy files to folder
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = ''
    checkpoints_dir = get_checkpoints_dir()
    model_path = 'experiments/uvp_adaptive_np/checkpoints_01/ckptwt_eval00079.pt'
    #model_path = 'experiments/most_merged/checkpoints/ckptwt00120.pt
    #name = 'uvp_big_1000small_noliving_rotate_other_bg0'
    name = 'uvp_adaptive_sm'
    datadir = './'
    batch_size = 96
    # Number of epochs to train f
    num_last_layer_features = 512

    dataloaders, class_names = get_dataset(exp_dir, batch_size=batch_size)
    # Load the pretrained model from pytorch
    # Number of classes in the dataset
    num_classes = len(class_names)
    model_dict, cnt_start, epoch_cnt, all_accuracy, all_losses = get_model(model_path, num_classes, num_last_layer_features, device)

    params_to_update = list(model_dict['model'].parameters())
    # Observe that all parameters are being optimized
    optimizer = optim.Adam(params_to_update, lr=1e-4)

    # Setup the loss fxn
    #erion = nn.CrossEntropyLoss()
    #criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))

    cnt = cnt_start
    for epoch_cnt in range(epoch_cnt,epoch_cnt+250):
        # Train and evaluate

        print('Starting epoch {}, seen count {}'.format(epoch_cnt, cnt))
        print('-' * 10)
        model_dict, optimizer, all_accuracy, all_losses, n_seen = train_model(model_dict, dataloaders, all_accuracy, all_losses, optimizer, device=device)
        cnt += n_seen
        if not epoch_cnt % 2:
            save_model(checkpoints_dir, cnt, model_dict, optimizer, all_accuracy, all_losses, epoch_cnt)


