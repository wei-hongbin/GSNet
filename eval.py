import numpy as np
import os
from utils.test_data import test_dataset
from utils.metric import cal_mae,cal_fm,cal_sm,cal_em,cal_wfm, cal_dice, cal_iou,cal_ber,cal_acc
from utils.config import te_data_list,te_data_list_2

dataset_rootpath_pre = 'result_map/GSNet' #预测图路径

def test():
    print("this is test")
    for name, path in te_data_list_2.items():
        sal_root = os.path.join(dataset_rootpath_pre,name) + '/'
        gt_root = path + '/'
        test_loader = test_dataset(sal_root, gt_root)
        mae,fm,sm,em,wfm, m_dice, m_iou,ber,acc= cal_mae(),cal_fm(test_loader.size),cal_sm(),cal_em(),cal_wfm(), cal_dice(), cal_iou(),cal_ber(),cal_acc()
        for i in range(test_loader.size):
            # print ('predicting for %d / %d' % ( i + 1, test_loader.size))
            sal, gt = test_loader.load_data()
            if sal.size != gt.size:
                x, y = gt.size
                sal = sal.resize((x, y))
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            gt[gt > 0.5] = 1
            gt[gt != 1] = 0
            res = sal
            res = np.array(res)
            if res.max() == res.min():
                res = res/255
            else:
                res = (res - res.min()) / (res.max() - res.min())
            mae.update(res, gt)
            sm.update(res,gt)
            fm.update(res, gt)
            em.update(res,gt)
            wfm.update(res,gt)
            m_dice.update(res,gt)
            m_iou.update(res,gt)
            ber.update(res,gt)
            acc.update(res,gt)

        MAE = mae.show()
        maxf,meanf,_,_ = fm.show()
        sm = sm.show()
        em = em.show()
        wfm = wfm.show()
        m_dice = m_dice.show()
        m_iou = m_iou.show()
        ber = ber.show()
        acc = acc.show()
        # print('dataset: {} MAE: {:.4f} maxF: {:.4f} avgF: {:.4f} wfm: {:.4f} Sm: {:.4f} Em: {:.4f} M_dice: {:.4f} M_iou: {:.4f} Ber: {:.4f}  Acc: {:.4f}'.format(dataset, MAE, maxf,meanf,wfm,sm,em, m_dice, m_iou,ber,acc))
        print('dataset: {} M_dice: {:.4f} M_iou: {:.4f}   wfm: {:.4f} Sm: {:.4f} Em: {:.4f} MAE: {:.4f}  Ber: {:.4f} maxF: {:.4f} avgF: {:.4f} Acc: {:.4f}'.format(name, m_dice, m_iou, wfm, sm, em, MAE, ber, maxf, meanf, acc))

if __name__ == "__main__":

    test()

