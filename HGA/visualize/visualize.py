import numpy as np
import cv2
import torch

## 353--48  063--88  211-1 -- 57
def main():
    # save_path = "/home/sjj/MMMSeg/MaskT/visual/region/HG_BraTS20_Training_353_t2.png"
    # save_path = "/home/sjj/MMMSeg/MaskT/visual/fusiontrans/fusiontrans_flairt1cet1t2_353.png"
    # save_path = "/home/sjj/MMMSeg/LongTail/exp/pic/baseline/flairt1.png"
    # save_path = "/home/sjj/MMMSeg/LongTail/exp/pic/ours/flairt1.png"
    # save_path = "/home/sjj/MMMSeg/LongTail/exp/pic/pmr/flairt1.png"
    # save_path = "/home/sjj/MMMSeg/LongTail/exp/pic/moddrop/flairt1.png"
    save_path = "/home/sjj/MMMSeg/LongTail/exp/pic/ours/t11.png"
    depth = 51

    blue   = [0,0,255]    # ET
    green  = [0,255,0]    # ED
    red    = [255,0,0]    # NCR/NET
    x=np.load("/home/sjj/MMMSeg/BraTS/BRATS2020_Training_none_npy/vol/HG_BraTS20_Training_019_vol.npy")

    print(x.shape)
    x=np.transpose(x,(3,1,0,2))
    C,H,W,Z=x.shape
    img_split = x[3]
    # original_img = img_split * 255.0
    original_img = ((img_split - np.min(img_split)) * 255.0 / (np.max(img_split) - np.min(img_split))).astype(np.uint8)
    # print(original_img.shape)
    original_img = original_img[:,:,depth]
    # original_img = np.zeros_like(original_img)
    original_img = cv2.cvtColor(original_img,cv2.COLOR_GRAY2BGR)

    y=np.load("/home/sjj/MMMSeg/BraTS/BRATS2020_Training_none_npy/seg/HG_BraTS20_Training_019_seg.npy")
    y=np.transpose(y,(1,0,2))
    gt = y[:,:,depth]
    gt = cv2.cvtColor(gt,cv2.COLOR_GRAY2BGR)
    # z=np.load("/home/sjj/MMMSeg/MaskT/visual/fusiontrans_flairt1cet1t2_353.npy")
    # z=np.load("/home/sjj/MMMSeg/LongTail/exp/visual/baseline/flairt1.npy")
    z=np.load("/home/sjj/MMMSeg/LongTail/exp/visual/ours/t11.npy")
    # z=np.load("/home/sjj/MMMSeg/LongTail/exp/visual/pmr/flairt1.npy")
    # z=np.load("/home/sjj/MMMSeg/LongTail/exp/visual/moddrop/flairt1.npy")

    z=np.transpose(z[0],(1,0,2)).astype(np.uint8)

    pred = z[:,:,depth]
    pred = cv2.cvtColor(pred,cv2.COLOR_GRAY2BGR)

    original_img = np.where(pred==1, np.full_like(original_img, red), original_img)
    original_img = np.where(pred==2, np.full_like(original_img, green), original_img)
    original_img = np.where(pred==3, np.full_like(original_img, blue), original_img)

    # original_img = np.where(gt==1, np.full_like(original_img, red), original_img)
    # original_img = np.where(gt==2, np.full_like(original_img, green), original_img)
    # original_img = np.where(gt==3, np.full_like(original_img, blue), original_img)
    original_img = cv2.cvtColor(original_img,cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_path, original_img,[int(cv2.IMWRITE_PNG_COMPRESSION),1])


if __name__ == '__main__':
    main()