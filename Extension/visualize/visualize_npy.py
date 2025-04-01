import torch
import numpy as np
from data.datasets_nii import Brats_loadall_test_nii
from utils.lr_scheduler import MultiEpochsDataLoader
from models import mmformer_pk

patch_size = 80
num_cls = 4

def main():
    save_path = "/home/sjj/MMMSeg/LongTail/exp/visual/ours/t11.npy"
    # masks_test = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
    #      [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
    #      [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
    #      [True, True, True, True]]
    # mask_name = ['t2', 't1c', 't1', 'flair',
    #         't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
    #         'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
    #         'flairt1cet1t2']
    feature_mask = [False, False, True, False]
    # mask_name = ['flairt1cet1t2']
    # resume = "/home/sjj/MMMSeg/LongTail/exp/out/2468_mmformer_baseline_Adam_nosplit_longtail_seed1037_batchsize1_epoch300_lr2e-4_temp4/model_last.pth"
    # resume = "/home/sjj/MMMSeg/LongTail/exp/out/2468_mmformer_pk_pp/model_last.pth"
    # resume = "/home/sjj/MMMSeg/LongTail/exp/out/2468_mmformer_pmr_AdamW_nosplit_longtail_seed1037_batchsize1_epoch300_lr2e-4_temp4/model_last.pth"
    # resume = "/home/sjj/MMMSeg/LongTail/exp/out/2468_mmformer_AdamW_split_longtail_seed1037_batchsize1_epoch300_lr2e-4_temp4/model_last.pth"
    resume = "/home/sjj/MMMSeg/LongTail/exp/out/2468_mmformer_ppramask_AdamW_nosplit_longtail_seed1037_batchsize1_epoch300_lr2e-4_temp4/model_last.pth"
    datapath = "/home/sjj/MMMSeg/BraTS/BRATS2020_Training_none_npy"
    test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
    test_file = '/home/sjj/MMMSeg/LongTail/visual.txt'
    test_set = Brats_loadall_test_nii(transforms=test_transforms, root=datapath, test_file=test_file)
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False)

    checkpoint = torch.load(resume)
    model = mmformer_pk.Model(num_cls=num_cls)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(checkpoint['state_dict'])
    print("load ok")
    with torch.no_grad():

        one_tensor = torch.ones(1, patch_size, patch_size, patch_size).float().cuda()

        H, W, T = 240, 240, 155
        model.eval()
        for i, data in enumerate(test_loader):
            target = data[1].cuda()
            x = data[0].cuda()
            names = data[-1]

            mask = torch.from_numpy(np.array(feature_mask))
            mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)

            mask = mask.cuda()
            _, _, H, W, Z = x.size()
            #########get h_ind, w_ind, z_ind for sliding windows
            h_cnt = np.int_(np.ceil((H - patch_size) / (patch_size * (1 - 0.5))))
            h_idx_list = range(0, h_cnt)
            h_idx_list = [h_idx * np.int_(patch_size * (1 - 0.5)) for h_idx in h_idx_list]
            h_idx_list.append(H - patch_size)

            w_cnt = np.int_(np.ceil((W - patch_size) / (patch_size * (1 - 0.5))))
            w_idx_list = range(0, w_cnt)
            w_idx_list = [w_idx * np.int_(patch_size * (1 - 0.5)) for w_idx in w_idx_list]
            w_idx_list.append(W - patch_size)

            z_cnt = np.int_(np.ceil((Z - patch_size) / (patch_size * (1 - 0.5))))
            z_idx_list = range(0, z_cnt)
            z_idx_list = [z_idx * np.int_(patch_size * (1 - 0.5)) for z_idx in z_idx_list]
            z_idx_list.append(Z - patch_size)

            weight1 = torch.zeros(1, 1, H, W, Z).float().cuda()
            for h in h_idx_list:
                for w in w_idx_list:
                    for z in z_idx_list:
                        weight1[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] += one_tensor
            weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

            pred = torch.zeros(len(names), num_cls, H, W, Z).float().cuda()
            model.module.is_training=False

            for h in h_idx_list:
                for w in w_idx_list:
                    for z in z_idx_list:
                        x_input = x[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size]
                        pred_part = model(x_input, mask)
                        pred[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] += pred_part
            pred = pred / weight

            pred = pred[:, :, :H, :W, :Z]
            pred = torch.argmax(pred, dim=1)

            npy = pred.data.cpu().numpy()
            np.save(save_path, npy)
        model.train()



if __name__ == '__main__':
    main()