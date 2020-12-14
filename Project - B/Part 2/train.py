"""
https://github.com/SaoYan/LearnToPayAttention
"""

import os
import random
import argparse
import numpy as np
#from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.utils as utils
import torchvision.transforms as transforms
from model1 import AttnVGG_before
from model2 import AttnVGG_after
from utilities import *
from torch.utils.data.sampler import SubsetRandomSampler
from scipy.signal import max_len_seq
from pickle import load

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser(description="LearnToPayAttn-CIFAR10")

parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
parser.add_argument("--lr", type=float, default=0.1, help="initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--attn_mode", type=str, default="after", help='insert attention modules before OR after maxpooling layers')

parser.add_argument("--normalize_attn", action='store_true', help='if True, attention map is normalized by softmax; otherwise use sigmoid')
parser.add_argument("--no_attention", action='store_false', help='turn down attention')
parser.add_argument("--log_images", action='store_true', help='log images and (is available) attention maps')

opt = parser.parse_args()



#------------------------------------------------------------------------------

class NWPU(torch.utils.data.Dataset):
    
    def __init__(self, data_path='../../NWPU-Updated/', transforms=None):
        self.dataset = torchvision.datasets.ImageFolder(root=data_path)
        self.transforms = transforms
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if transforms is not None:
            image = self.transforms(image)
        return image, label
    
#------------------------------------------------------------------------------T


def main():
    # dataset split/size parameters
    num_aug = 1
    im_size = 32        # image will be resize to 32x32 for leanring task
    train_split = 0.8   # 80% image data will be used as training and 20% as testing
    num_classes = 759    # 45 different classes
    num_images_per_class = 700 # 700 images per class
    
    transform_train = transforms.Compose([
        transforms.Resize((im_size,im_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((im_size,im_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    #train_data = NWPU(data_path='../../NWPU-Updated/', transforms=transform_test)
    #test_data = NWPU(data_path='../../NWPU-Updated/', transforms=transform_test)

    #split = int(np.floor(train_split * num_images_per_class))
    #indices = np.arange(num_classes*num_images_per_class).reshape(-1, num_images_per_class)
    #train_indices, test_indices = indices[:,:split].flatten(), indices[:,split:].flatten()

    #trainloader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=False, num_workers=0, sampler=SubsetRandomSampler(train_indices))
    #testloader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False, num_workers=0, sampler=SubsetRandomSampler(test_indices))

    full_dataset = NWPU(data_path='../../NWPU-Updated/', transforms=transform_train)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False, num_workers=0)
    
   
    # use attention module?
    if not opt.no_attention:
        print('turn on attention ...\n')
    else:
        print('turn off attention ...\n')
    # (linear attn) insert attention befroe or after maxpooling?
    # (grid attn only supports "before" mode)
    if opt.attn_mode == 'before':
        print('pay attention before maxpooling layers...\n')
        net = AttnVGG_before(im_size=im_size, num_classes=num_classes,
            attention=not opt.no_attention, normalize_attn=opt.normalize_attn, init='xavierUniform')
    elif opt.attn_mode == 'after':
        print('pay attention after maxpooling layers...\n')
        net = AttnVGG_after(im_size=im_size, num_classes=num_classes,
            attention=not opt.no_attention, normalize_attn=opt.normalize_attn, init='xavierUniform')
    else:
        raise NotImplementedError("Invalid attention mode!")
    criterion = nn.CrossEntropyLoss()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device_ids = [0]#,1]
    #model = nn.DataParallel(net, device_ids=device_ids).to(device)
    model = net.to(device)
    criterion.to(device)

    ### optimizer
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    lr_lambda = lambda epoch : np.power(0.5, int(epoch/25))
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # training
    
    step = 0
    running_avg_accuracy = 0
    #writer = SummaryWriter(opt.outf)
    for epoch in range(opt.epochs):
        images_disp = []
        #writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        #print("\nepoch %d learning rate %f\n" % (epoch, optimizer.param_groups[0]['lr']))
        # run for one epoch
        for aug in range(num_aug):
            for i, data in enumerate(trainloader, 0):
                # warm up
                model.train()
                model.zero_grad()
                optimizer.zero_grad()
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # if (aug == 0) and (i == 0): # archive images in order to save to logs
                #     images_disp.append(inputs[0:36,:,:,:])
                # forward
                pred, __, __, __ = model(inputs)
                # backward
                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()
                # display results
                if i % 10 == 0:
                    model.eval()
                    pred, __, __, __ = model(inputs)
                    predict = torch.argmax(pred, 1)
                    total = labels.size(0)
                    correct = torch.eq(predict, labels).sum().double().item()
                    accuracy = correct / total
                    running_avg_accuracy = 0.9*running_avg_accuracy + 0.1*accuracy
                    # writer.add_scalar('train/loss', loss.item(), step)
                    # writer.add_scalar('train/accuracy', accuracy, step)
                    # writer.add_scalar('train/running_avg_accuracy', running_avg_accuracy, step)
                    print("[epoch %d][aug %d/%d][%d/%d] loss %.4f accuracy %.2f%% running avg accuracy %.2f%%"
                        % (epoch, aug, num_aug-1, i, len(trainloader)-1, loss.item(), (100*accuracy), (100*running_avg_accuracy)))
                step += 1
        # adjust learning rate
        scheduler.step()
        
        # the end of each epoch: test & log
        #print('\none epoch done, saving records ...\n')
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))
        if epoch == 5: #opt.epochs / 2:
            torch.save(model.state_dict(), os.path.join(opt.outf, 'net%d.pth' % epoch))
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            # log scalars
            for i, data in enumerate(testloader, 0):
                images_test, labels_test = data
                images_test, labels_test = images_test.to(device), labels_test.to(device)
                # if i == 0: # archive images in order to save to logs
                #     images_disp.append(inputs[0:36,:,:,:])
                pred_test, __, __, __ = model(images_test)
                predict = torch.argmax(pred_test, 1)
                total += labels_test.size(0)
                correct += torch.eq(predict, labels_test).sum().double().item()
            #writer.add_scalar('test/accuracy', correct/total, epoch)
            print("[epoch %d] accuracy on test data: %.2f%%" % (epoch, 100*correct/total))
            # log images
            # if opt.log_images:
            #     #print('\nlog images ...\n')
            #     I_train = utils.make_grid(images_disp[0], nrow=6, normalize=True, scale_each=True)
            #     #writer.add_image('train/image', I_train, epoch)
            #     if epoch == 0:
            #         I_test = utils.make_grid(images_disp[1], nrow=6, normalize=True, scale_each=True)
            #         #writer.add_image('test/image', I_test, epoch)
            # if opt.log_images and (not opt.no_attention):
            #     #print('\nlog attention maps ...\n')
            #     # base factor
            #     if opt.attn_mode == 'before':
            #         min_up_factor = 1
            #     else:
            #         min_up_factor = 2
            #     # sigmoid or softmax
            #     if opt.normalize_attn:
            #         vis_fun = visualize_attn_softmax
            #     else:
            #         vis_fun = visualize_attn_sigmoid
            #     # training data
            #     __, c1, c2, c3 = model(images_disp[0])
            #     if c1 is not None:
            #         attn1 = vis_fun(I_train, c1, up_factor=min_up_factor, nrow=6)
            #         #writer.add_image('train/attention_map_1', attn1, epoch)
            #     if c2 is not None:
            #         attn2 = vis_fun(I_train, c2, up_factor=min_up_factor*2, nrow=6)
            #         #writer.add_image('train/attention_map_2', attn2, epoch)
            #     if c3 is not None:
            #         attn3 = vis_fun(I_train, c3, up_factor=min_up_factor*4, nrow=6)
            #         #writer.add_image('train/attention_map_3', attn3, epoch)
            #     # test data
            #     __, c1, c2, c3 = model(images_disp[1])
            #     if c1 is not None:
            #         attn1 = vis_fun(I_test, c1, up_factor=min_up_factor, nrow=6)
            #         #writer.add_image('test/attention_map_1', attn1, epoch)
            #     if c2 is not None:
            #         attn2 = vis_fun(I_test, c2, up_factor=min_up_factor*2, nrow=6)
            #         #writer.add_image('test/attention_map_2', attn2, epoch)
            #     if c3 is not None:
            #         attn3 = vis_fun(I_test, c3, up_factor=min_up_factor*4, nrow=6)
            #         #writer.add_image('test/attention_map_3', attn3, epoch)

if __name__ == "__main__":
    main()
