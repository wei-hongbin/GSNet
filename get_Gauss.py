
import os

import cv2
import mmcv
import numpy as np
from tqdm import tqdm

data_root = '/home/zxq/Downloads/datasets/miccai_segmentation/TrainDataset/TrainDataset/mask'
save_root = '/home/zxq/Downloads/datasets/miccai_segmentation/TrainDataset/TrainDataset/gaussmap'

def get_bbox(mask, show=False):
    """
    Get the bbox for a binary mask
    Args:
        mask: a binary mask

    Returns:
        bbox: (col_min, col_max, row_min, row_max)
    """
    area_obj = np.where(mask != 0)
    bbox = np.min(area_obj[0]), np.max(area_obj[0]), \
           np.min(area_obj[1]), np.max(area_obj[1])
    if show:
        cv2.rectangle(mask, (bbox[2], bbox[0]), (bbox[3], bbox[1]),
                      (255, 255, 255), 1)
        mmcv.imshow(mask, 'test', 10)
        exit()
    return bbox


def get_hotspot(mask, bbox):
    h, w = mask.shape
    obj_h, obj_w = bbox[1] - bbox[0], bbox[3] - bbox[2]
    R = min(obj_h, obj_w)
    center_yx = sum(bbox[:2]) // 2, sum(bbox[2:]) // 2
    x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))
    rel_y_map, rel_x_map = y_map - center_yx[0], x_map - center_yx[1]
    gauss_map = np.exp(-0.5 * np.sqrt(rel_y_map ** 2 + rel_x_map ** 2) / R)
    low_limit = gauss_map[mask != 0].min()
    hotspot_map = np.where(gauss_map > low_limit, gauss_map,
                           np.ones_like(mask) * low_limit)
    hotspot_map = (hotspot_map - hotspot_map.min()) / \
                  (hotspot_map.max() - hotspot_map.min())
    hotspot_map[hotspot_map<0.5] = 0
    return hotspot_map


def main():
    image_name_list = sorted(os.listdir(data_root))
    for image_name in tqdm(image_name_list, total=len(image_name_list)):
        image_path = os.path.join(data_root, image_name)
        save_path = os.path.join(save_root, image_name)
        img = mmcv.imread(image_path, flag='grayscale')
        # print(img.max(),img.min())
        img[img > 128] = 255
        img[img <= 128] = 0

        if img.max() != 0 :
            img = img.astype(np.uint8)
            mideanblured_img = cv2.medianBlur(img, 5)
            dilated_img = cv2.dilate(mideanblured_img,
                                     kernel=np.ones((7, 7), np.uint8),
                                     iterations=1)
            bbox = get_bbox(mask=dilated_img, show=False)
            hotspot = get_hotspot(mask=dilated_img, bbox=bbox) * 255
        else:
            hotspot = img
        mmcv.imwrite((hotspot ).astype(np.uint8), save_path,
                     auto_mkdir=True)


if __name__ == '__main__':
    # mask = np.zeros(shape=(1000, 1000))
    # mask[200:700, 400:900] = 1
    # bbox = get_bbox(mask=mask)
    # get_hotspot(mask=mask, bbox=bbox, )
    main()
