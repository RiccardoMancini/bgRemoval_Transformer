import numpy as np
from matplotlib import pyplot as plt
from mmsegmentation.mmseg.apis import train_segmentor, init_random_seed
import mmcv
from mmseg.models import build_segmentor
from mmseg.datasets import build_dataset, build_dataloader


def get_data_after_processing(dataset):
    i = 0
    for data in dataset[0]:
        print(data)
        plt.figure()
        plt.imshow(data['img'].data.numpy().transpose(1, 2, 0))
        plt.show()
        plt.figure()
        plt.imshow(data['gt_semantic_seg'].data.numpy().transpose(1, 2, 0).astype(np.uint8), cmap='Greys')
        plt.show()
        if i > 5:
            break
        i += 1


def train(cnf):
    config = mmcv.Config.fromfile(cnf)
    config.seed = init_random_seed()
    config.gpu_ids = range(1)
    # config.find_unused_parameters = True
    config.workflow = [('train', 1)]
    print(config.model.type, config.model.backbone.type)
    # print(config.workflow, config.data_root, config.load_from, init_random_seed(42))
    datasets = [build_dataset(config.data.train)]

    # get_data_after_processing(datasets)

    '''print(datasets.__getitem__(0)[0]['gt_semantic_seg'].data.shape)
    print(datasets.__getitem__(0)[0]['img_metas'].data['ori_filename'])'''
    model = build_segmentor(config.model)
    model.CLASSES = datasets[0].CLASSES

    train_segmentor(model, datasets, config, distributed=True, validate=False,
                    meta=dict())
