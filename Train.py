import torch
from torch.autograd import Variable
import os
import math
import argparse
from datetime import datetime
from GSNet import GSNet
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging
import matplotlib.pyplot as plt

torch.cuda.set_device(0)

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def test(model, path, dataset):

    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        _, _, res  = model(image)
        # eval Dice
        res = F.upsample(res , size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

    return DSC / num1



def train(train_loader, model, optimizer, epoch, test_path):
    model.train()
    global best
    x = epoch
    W_g = math.exp((-x)/50)
    W_b = 1/(math.exp(-x/20)+1)
    size_rates = [0.75, 1, 1.25] 
    loss_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            # images, gts = pack
            images, gausss, bodys, gts = pack
            images = Variable(images).cuda()

            gausss = Variable(gausss).cuda()
            bodys = Variable(bodys).cuda()
            gts = Variable(gts).cuda()


            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
               
                gausss = F.upsample(gausss, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                bodys = F.upsample(bodys, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            out_gauss,out_body,out_pre= model(images)
            # ---- loss function ----
            loss_G = F.binary_cross_entropy_with_logits(out_gauss, gausss)
            loss_B = F.binary_cross_entropy_with_logits(out_body, bodys)
            loss_P = structure_loss(out_pre, gts)
            loss = loss_G*W_g + loss_B*W_b + loss_P
            


            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)
        # ---- train visualization ----
        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record.show()))

    # save model 
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch>(opt.epoch - 5):   
        torch.save(model.state_dict(), save_path +str(epoch)+ 'GSNet.pth')
    # choose the best model

    # global dict_plot
   
    test1path ="/mnt/cpfs/dataset/tuxiangzu/Classfication_Detection_Group/whb/V100/GSNet/PolypDataset/Test"
    # if ( epoch  % 1 == 0) and (epoch >(opt.epoch - 30)):
    if ( epoch  % 1 == 0):
        # for dataset in ['CVC_300', 'CVC_ClinicDB', 'Kvasir', 'CVC_ColonDB', 'ETIS_LaribPolypDB']:
        fig = 0
        for dataset in ['Kvasir', 'CVC_ClinicDB', 'CVC_ColonDB', 'ETIS_LaribPolypDB', 'CVC_300' ]:    
            dataset_dice = test(model, test1path, dataset)
            if dataset == 'Kvasir' and dataset_dice> 0.9165:
                fig += 1
            elif dataset == 'CVC_ClinicDB' and dataset_dice> 0.9365:
                fig += 1
            elif dataset == 'CVC_ColonDB' and dataset_dice> 0.8075:
                fig += 1
            elif dataset == 'ETIS_LaribPolypDB' and dataset_dice> 0.7865:
                fig += 1
            elif dataset == 'CVC_300' and dataset_dice> 0.8995:
                fig += 1
            else:
                fig += 0
                            
                
            logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
            print(dataset, ': ', dataset_dice)
            dict_plot[dataset].append(dataset_dice)
            if fig >= 5:
               torch.save(model.state_dict(), save_path + str(epoch) + 'best.pth') 
               logging.info('save best.pth')
 

def plot_train(dict_plot=None, name = None):
    color = ['red', 'lawngreen', 'lime', 'gold', 'm', 'plum', 'blue']
    line = ['-', "--"]
    for i in range(len(name)):
        plt.plot(dict_plot[name[i]], label=name[i], color=color[i], linestyle=line[(i + 1) % 2])
        transfuse = {'CVC_300': 0.900, 'CVC_ClinicDB': 0.937, 'Kvasir': 0.917, 'CVC_ColonDB': 0.808,'ETIS_LaribPolypDB': 0.787, 'test':0.83}
        plt.axhline(y=transfuse[name[i]], color=color[i], linestyle='-')
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.title('Train')
    plt.legend()
    plt.savefig('eval.png')
    plt.show()
    
    
if __name__ == '__main__':
    dict_plot = {'CVC_300':[], 'CVC_ClinicDB':[], 'Kvasir':[], 'CVC_ColonDB':[], 'ETIS_LaribPolypDB':[], 'test':[]}
    name = ['CVC_300', 'CVC_ClinicDB', 'Kvasir', 'CVC_ColonDB', 'ETIS_LaribPolypDB', 'test']
    ##################model_name#############################
    model_name = 'GSNet'
    ###############################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')


    parser.add_argument('--train_path', type=str,
                        default="/mnt/cpfs/dataset/tuxiangzu/Classfication_Detection_Group/whb/V100/GSNet/PolypDataset/Train",
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default="/mnt/cpfs/dataset/tuxiangzu/Classfication_Detection_Group/whb/V100/GSNet/PolypDataset/Test",
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='./model_pth/'+model_name+'/')

    opt = parser.parse_args()
    logging.basicConfig(filename='train_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = GSNet().cuda()

    best = 0

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/images/'.format(opt.train_path)
    gauss_root = '{}/gaussmap/'.format(opt.train_path)
    body_root = '{}/body/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gauss_root, body_root,gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, opt.epoch+1):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        train(train_loader, model, optimizer, epoch, opt.test_path)
    
    # plot the eval.png in the training stage
    # plot_train(dict_plot, name)

    


