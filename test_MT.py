import os
import argparse
import torch
from test_MT_util import test_all_case
from networks.MTMT import build_model

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/media/data/chenzhihao/datasets/ISIC_2018_Skin_Lesion', help='Name of Experiment')
parser.add_argument('--TI_path', type=str, default='ISIC2018_Task1-2_Test_Input', help='Test image path')
parser.add_argument('--TA_path', type=str, default='None', help='Test GroundTruth')
parser.add_argument('--model', type=str,  default='MTMT', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
# parser.add_argument('--base_lr', type=float,  default=0.005, help='base learning rate')
# parser.add_argument('--edge', type=float, default='10', help='edge learning weight')
# parser.add_argument('--epoch_name', type=str,  default='iter_7000.pth', help='choose one epoch/iter as pretrained')
# parser.add_argument('--consistency', type=float,  default=1.0, help='consistency')
parser.add_argument('--scale', type=int,  default=416, help='batch size of 8 with resolution of 416*416 is exactly OK')
# parser.add_argument('--subitizing', type=float,  default=5.0, help='subitizing loss weight')


FLAGS = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = '../models_ISIC2018/MTMT_baseline_v2/iter_7000.pth'
test_save_path = '../models_ISIC2018/MTMT_baseline_v2/prediction_crf'
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(snapshot_path)
num_classes = 1

img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(FLAGS.root_path, FLAGS.TI_path)) if f.endswith('.jpg')]
if FLAGS.TA_path == 'None':
    data_path = [(os.path.join(FLAGS.root_path, FLAGS.TI_path, img_name + '.jpg'), -1) #-1 is None path
                for img_name in img_list]
else:
    data_path = [(os.path.join(FLAGS.root_path, FLAGS.TI_path, img_name + '.jpg'),
                 os.path.join(FLAGS.root_path, FLAGS.TA_path, img_name + '_segmentation.png'))
                for img_name in img_list]
# data_path = [(os.path.join(FLAGS.root_path, 'ShadowImages', img_name + '.jpg'), '****') for img_name in img_list]


def test_calculate_metric():
    net = build_model('resnext101').cuda()
    net.load_state_dict(torch.load(snapshot_path))
    print("init weight from {}".format(snapshot_path))
    net.eval()

    avg_metric = test_all_case(net, data_path, num_classes=num_classes,
                               save_result=True, test_save_path=test_save_path, trans_scale=FLAGS.scale, GT_access=False)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    with open('record/test_record_EGNet_meanteacher.txt', 'a') as f:
        f.write(snapshot_path+' ')
        f.write(str(metric)+' --UCF\r\n')
    print('Test ber results: {}'.format(metric))
