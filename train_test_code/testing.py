import os
import json
import numpy as np
from PIL import Image
import time
import cv2
from matplotlib import pyplot as plt
from mmsegmentation.mmseg.apis import inference_segmentor, init_segmentor, single_gpu_test
import mmcv
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.core.evaluation import mean_iou, mean_fscore, mean_dice
import neptune.new as neptune


def inference_test(config_file: str, checkpoint_file: str):
    imgs_data = os.scandir('./dataset/img_dir/test/')
    imgs_name = [o.name for o in imgs_data]
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
    for img in imgs_name:
        path = './dataset/img_dir/test/' + img
        print(path)
        result = inference_segmentor(model, path)
        model.show_result(path, result, out_file='./train_test_code/images_processed/inference/' + img, opacity=0.5)


def plotAccuracyLoss(log_path):
    run = neptune.init(
        project="RECV/CV-DL",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1Yjg1YTE3Mi03NzUzLTRkMDUtYmU3ZC1hODQxOTI0ZGUwYWEifQ==",
    )
    # read log file
    log = []
    for line in open(log_path, 'r'):
        log.append(json.loads(line))
    loss = []
    acc = []
    i = 0
    while i < len(log):
        loss.append(log[i]['loss'])
        acc.append(log[i]['decode.acc_seg'])
        if i != 0:
            i += 20
        else:
            i += 19

    for j in range(len(loss)):
        run["train/loss"].log(loss[j])
        run['train/accuracy'].log(acc[j])
    run.stop()


def removeBackground(ground_t, masks, imgs):
    i = 0
    imgs_data = os.scandir('./dataset/ann_dir/test/')
    imgs_name = [o.name for o in imgs_data]
    print(imgs_name)
    for mask in masks:
        mask = Image.fromarray(mask.astype(np.uint8))
        mask = np.copy(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        ground = cv2.imread(ground_t[i])
        ground = cv2.cvtColor(ground, cv2.COLOR_BGR2RGB)

        img = cv2.imread(imgs[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cond_m = np.stack(mask) > 0
        cond_g = np.stack(ground) > 0

        # where condition is true yield img, otherwise yield mask
        seg_m = np.where(cond_m, img, mask)
        seg_g = np.where(cond_g, img, ground)

        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(25, 25)
        axs[0].imshow(seg_m)
        axs[1].imshow(seg_g)
        # plt.show()
        fig.savefig('./train_test_code/images_processed/background_removed/' + imgs_name[i], dpi=fig.dpi)

        i += 1


def results2Img(results):
    i = 0
    for result in zip(results):
        print(result)
        output = Image.fromarray(result[0].astype(np.uint8))
        output.show()
        i += 1
        if i > 4:
            break


def test(cnf, checkpoint):
    config = mmcv.Config.fromfile(cnf)
    config.seed = 42
    config.gpu_ids = range(1)
    loader_cfg = dict(
        num_gpus=len(config.gpu_ids),
        seed=config.seed,
        drop_last=True)
    loader_cfg.update({
        k: v
        for k, v in config.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_dataset = build_dataset(config.data.test, dict(test_mode=True))
    test_loader_cfg = {
        **loader_cfg,
        'samples_per_gpu': 1,
        'shuffle': False,  # Not shuffle by default
        **config.data.get('test_dataloader', {}),
    }
    test_dataloader = build_dataloader(test_dataset, **test_loader_cfg)

    model = build_segmentor(config.model)
    load_checkpoint(model, checkpoint, map_location='cpu')
    time.sleep(1)

    results = single_gpu_test(model, test_dataloader)

    imgs = os.scandir('./dataset/img_dir/test/')
    imgs_path = ['./dataset/img_dir/test/' + o.name for o in imgs]
    gr_t = os.scandir('./dataset/ann_dir/test/')
    gr_t_path = ['./dataset/ann_dir/test/' + o.name for o in gr_t]
    removeBackground(gr_t_path, results, imgs_path)
    # results2img(results)

    '''obj = os.scandir('./dataset/ann_dir/test/')
    seg_map = ['./dataset/ann_dir/test/' + o.name for o in obj]
    res = mean_iou(results, seg_map, 28, 255, reduce_zero_label=False)'''
    '''res2 = mean_fscore(results, seg_map, 27, 0, reduce_zero_label=True)
    res3 = mean_dice(results, seg_map, 27, 0, reduce_zero_label=True)'''
    #print(res)
    #print('Overall IoU (%): ', np.mean(res['IoU'], where=np.isfinite(res['IoU'])) * 100)
    '''print('Overall F1Score (%): ', np.mean(res2['Fscore'] * 100, where=np.isfinite(res2['Fscore'])))
    print('Overall Dice (%): ', np.mean(res3['Dice'] * 100, where=np.isfinite(res3['Dice'])))'''
