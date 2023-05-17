import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from GSNet import GSNet
from utils.dataloader import test_dataset
import cv2
torch.cuda.set_device(1)


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default='./model_pth/GSNet/58best.pth')
    opt = parser.parse_args()
    model = GSNet()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()
    for _data_name in ['CVC_300', 'CVC_ClinicDB', 'Kvasir', 'CVC_ColonDB', 'ETIS_LaribPolypDB']:

        ##### put data_path here #####
        data_path = '/mnt/cpfs/dataset/tuxiangzu/Classfication_Detection_Group/whb/V100/GSNet/PolypDataset/Test/{}'.format(_data_name)
        ##### save_path #####
        save_path = './result_map/GSNet/{}/'.format(_data_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset(image_root, gt_root, 352)
        for i in range(num1):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            # pre,_,_ = model(image) #gauss
            # _,pre,_ = model(image) #body
            _,_,pre = model(image)
            res = F.upsample(pre, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(save_path+name, res*255)
        print(_data_name, 'Finish!')
if __name__ == '__main__':
    test()
    
    