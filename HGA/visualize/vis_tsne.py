import torch
import numpy as np
from data.datasets_nii import Brats_loadall_test_nii
from utils.lr_scheduler import MultiEpochsDataLoader
from models import mmformer_vis
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from alphashape import alphashape

patch_size = 80
num_cls = 4
alpha =1.0

def plot_embedding_2D(data, label, colors, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=colors[i],
                 fontdict={'weight': 'bold', 'size': 9})
    # hull = ConvexHull(data[::5])
    shape1 = alphashape(data[::5], alpha)
    x1, y1 = shape1.exterior.xy
    plt.plot(x1, y1, color='#D62728', linestyle='--', alpha=0.6)

    shape2 = alphashape(data[1::5], alpha)
    x2, y2 = shape2.exterior.xy
    plt.plot(x2, y2, color='#1F77B4', linestyle='--', alpha=0.6)

    shape3 = alphashape(data[2::5], alpha)
    x3, y3 = shape3.exterior.xy
    plt.plot(x3, y3, color='#FF7F0E', linestyle='--', alpha=0.6)

    shape4 = alphashape(data[3::5], alpha)
    x4, y4 = shape4.exterior.xy
    plt.plot(x4, y4, color='#2CA02C', linestyle='--', alpha=0.6)

    shape5 = alphashape(data[4::5], alpha)
    x5, y5 = shape5.exterior.xy
    plt.plot(x5, y5, color='#9467BD', linestyle='--', alpha=0.6)

    # for polygon in shape.geoms:
    #     x, y = polygon.exterior.xy
    #     plt.plot(x, y, color='#D62728')
    plt.xticks([])
    plt.yticks([])
    # plt.title(title)
    return fig

def main():
    # save_path = "/home/sjj/Visual/npy/ours/t1cflairt2.npy"
    # masks_test = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
    #      [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
    #      [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
    #      [True, True, True, True]]
    # mask_name = ['t2', 't1c', 't1', 'flair',
    #         't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
    #         'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
    #         'flairt1cet1t2']
    feature_mask = [True, True, True, True]
    # feature_mask = [True, True, False, True]
    # feature_mask = [True, False, False, True]
    # feature_mask = [False, False, False, True]
    # mask_name = ['flairt1cet1t2']
    resume = "/home/sjj/MMMSeg/LongTail/exp/out/1379_mmformer_Adam_nosplit_longtail_seed1037_batchsize1_epoch300_lr2e-4_temp4/model_last.pth"
    # resume = "/home/sjj/MMMSeg/LongTail/exp/out/2468_mmformer_baseline_Adam_nosplit_longtail_seed1037_batchsize1_epoch300_lr2e-4_temp4/model_last.pth"
    # resume = "/home/sjj/PASSION/code/outputs/idt_mr2468_meta_imb_ra_reptile_linear_prm_dce1e-4_kl5e-1_p2e-2_mmformer_passion_bs1_epoch300_lr2e-3_temp4/model_last.pth"
    # resume = "/home/sjj/PASSION/code/outputs/idt_mr2468_meta_imb_ra_reptile_linear_prm_dce1e-4_kl5e-1_p2e-2_mmformer_passion_bs1_epoch300_lr2e-3_temp4/model_last.pth"
    # resume = "/home/sjj/MMMSeg/LongTail/exp/out/2468_mmformer_pk_pp/model_last.pth"
    # resume = "/home/sjj/MMMSeg/LongTail/exp/out/2468_mmformer_pmr_AdamW_nosplit_longtail_seed1037_batchsize1_epoch300_lr2e-4_temp4/model_300.pth"
    # resume = "/home/sjj/MMMSeg/LongTail/exp/out/2468_mmformer_AdamW_split_longtail_seed1037_batchsize1_epoch300_lr2e-4_temp4/model_last.pth"
    # resume = "/home/sjj/MMMSeg/LongTail/exp/out/2468_mmformer_ppramask_AdamW_nosplit_longtail_seed1037_batchsize1_epoch300_lr2e-4_temp4/model_last.pth"
    datapath = "/home/sjj/MMMSeg/BraTS/BRATS2020_Training_none_npy"
    test_transforms = 'Compose([RandCrop3D((80,80,80)), NumpyType((np.float32, np.int64)),])'
    test_file = '/home/sjj/MMMSeg/BraTS/BRATS2020_Training_none_npy/test.txt'
    test_set = Brats_loadall_test_nii(transforms=test_transforms, root=datapath, test_file=test_file)
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False)

    checkpoint = torch.load(resume)
    model = mmformer_vis.Model(num_cls=num_cls)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(checkpoint['state_dict'])
    print("load ok")
    tsne_points = []
    tsne_labels = []
    tsne_colors = []
    with torch.no_grad():

        model.eval()
        for i, data in enumerate(test_loader):
            target = data[1].cuda()
            x = data[0].cuda()
            names = data[-1]

            mask = torch.from_numpy(np.array(feature_mask))
            mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)

            mask = mask.cuda()
            # _, _, H, W, Z = x.size()
            #########get h_ind, w_ind, z_ind for sliding windows

            vis_ft = model(x, mask)
            print(vis_ft.size())

            tsne_2D = TSNE(n_components=2, init='pca', random_state=308)
            tsne_points.append(vis_ft.cpu().detach().numpy())
            tsne_colors.append(['#D62728', '#1F77B4', '#FF7F0E', '#2CA02C', '#9467BD'])
            # tsne_labels.append([0, 1, 2, 3, 4, 5, 6, 7])
            tsne_labels.append(['x', 'o', 'o', 'o', 'o'])
            if len(tsne_points) == 50:  # actual visualization
                vis_tsne_points = np.concatenate(tsne_points)
                vis_tsne_labels = np.concatenate(tsne_labels)
                vis_tsne_colors = np.concatenate(tsne_colors)
                vis_tsne_2D = tsne_2D.fit_transform(vis_tsne_points)
                tsne_fig = plot_embedding_2D(vis_tsne_2D, vis_tsne_labels, vis_tsne_colors, 't-SNE of Features')
                plt.axis('off')
                # plt.show()
                plt.savefig('tsne50-base.png', format='png', dpi=300, bbox_inches='tight')
                quit()

        vis_tsne_points = np.concatenate(tsne_points)
        vis_tsne_labels = np.concatenate(tsne_labels)
        vis_tsne_colors = np.concatenate(tsne_colors)
        vis_tsne_2D = tsne_2D.fit_transform(vis_tsne_points)
        tsne_fig = plot_embedding_2D(vis_tsne_2D, vis_tsne_labels, vis_tsne_colors, 't-SNE of Our Features')
        # plt.show()
        plt.savefig('tsne1379.png')




if __name__ == '__main__':
    main()