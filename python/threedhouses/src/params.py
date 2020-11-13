
'''
params.py

Managers of all hyper-parameters

'''

import torch

epochs = 100
batch_size = 32
pos_weight = 30
soft_label = False
adv_weight = 0
d_thresh = 0.8
z_dim = 200
z_dis = "norm"
model_save_step = 5
g_lr = 0.001
d_lr = 0.00001
e_lr = 0.0005
kl_weight = 0.0001
beta = (0.5, 0.999)
cube_len = 32
leak_value = 0.2
bias = False
noise = [0] #list(range(-2, 3))
num_block_types = 255
use_block_type = False
use_block_part = False
only_popular_parts = False
part_augment = False
vae = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#data_dir = '/scr/kayburns/instance_segmentation_data/'
data_dir = '/craftassist/python/threedhouses/data'

from_weights = None
#output_dir = '/craftassist/python/threedhouses/models/ae_debug/'
#images_dir = '/craftassist/python/threedhouses/images/'
output_dir = './out/BCE_kl{}_elr{}_glr{}_bs{}_pw{}/'.format(kl_weight, e_lr, g_lr, batch_size, pos_weight)
images_dir = './out/BCE_kl{}_elr{}_glr{}_bs{}_pw{}/'.format(kl_weight, e_lr, g_lr, batch_size, pos_weight)

def print_params():
    l = 16
    print (l*'*' + 'hyper-parameters' + l*'*')

    print ('epochs =', epochs)
    print ('batch_size =', batch_size)
    print ('soft_labels =', soft_label)
    print ('adv_weight =', adv_weight)
    print ('d_thresh =', d_thresh)
    print ('z_dim =', z_dim)
    print ('z_dis =', z_dis)
    print ('model_images_save_step =', model_save_step)
    print ('device =', device)
    print ('g_lr =', g_lr)
    print ('d_lr =', d_lr)
    print ('cube_len =', cube_len)
    print ('leak_value =', leak_value)
    print ('bias =', bias)

    print (l*'*' + 'hyper-parameters' + l*'*')


