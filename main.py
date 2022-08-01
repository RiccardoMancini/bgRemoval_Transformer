from train_test_code.testing import inference_test, test, removeBackground
from train_test_code.training import train
from dataset.dataset_manipulation import test_data
import time

if __name__ == '__main__':
    cnf_file_segmenter = 'configs/config_segmenter_ViT_B_Mask.py'
    cnf_file_setr = 'configs/config_setr_ViT_mla.py'
    # chk_created = 'weights/segmenterB/epoch_10.pth'
    chk_created = 'weights/SETR_mla/epoch_5.pth'
    # test_data()
    # train(cnf_file_segmenter)
    test(cnf_file_setr, chk_created)
    # inference_test(cnf_file_segmenter, chk_created)
