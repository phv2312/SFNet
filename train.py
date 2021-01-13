import torch
import torch.optim.lr_scheduler as lrs
import torch.nn.functional as F
import numpy as np
import os
import random

from geek_dataset import RandomAugmentPairAnimeDataset as PairAnimeDataset
from custom_loss import loss_function
from model import SFNet
import argparse

parser = argparse.ArgumentParser(description="SFNet")
parser.add_argument('--seed', type=int, default=None, help='random seed')
parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size for training')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs for training')
parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.2, help='decaying factor')
parser.add_argument('--decay_schedule', type=str, default='30', help='learning rate decaying schedule')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data loader')
parser.add_argument('--feature_h', type=int, default=48, help='height of feature volume')
parser.add_argument('--feature_w', type=int, default=32, help='width of feature volume')
parser.add_argument('--train_image_path', type=str, default='./data/training_data/VOC2012_seg_img.npy', help='directory of pre-processed(.npy) images')
parser.add_argument('--train_mask_path', type=str, default='./data/training_data/VOC2012_seg_msk.npy', help='directory of pre-processed(.npy) foreground masks')
parser.add_argument('--valid_csv_path', type=str, default='./data/PF_Pascal/bbox_val_pairs_pf_pascal.csv', help='directory of validation csv file')
parser.add_argument('--valid_image_path', type=str, default='./data/PF_Pascal/', help='directory of validation data')
parser.add_argument('--beta', type=float, default=50, help='inverse temperature of softmax @ kernel soft argmax')
parser.add_argument('--kernel_sigma', type=float, default=8, help='standard deviation of Gaussian kerenl @ kernel soft argmax')
parser.add_argument('--lambda1', type=float, default=3, help='weight parameter of mask consistency loss')
parser.add_argument('--lambda2', type=float, default=16, help='weight parameter of flow consistency loss')
parser.add_argument('--lambda3', type=float, default=0.5, help='weight parameter of smoothness loss')
parser.add_argument('--eval_type', type=str, default='bounding_box', choices=('bounding_box','image_size'), help='evaluation type for PCK threshold (bounding box | image size)')
args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Set seed
if args.seed == None:
    args.seed = np.random.randint(10000)
    print('Seed number: ', args.seed)

global global_seed
global_seed = args.seed
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)
torch.cuda.manual_seed_all(global_seed)
np.random.seed(global_seed)
random.seed(global_seed)
torch.backends.cudnn.deterministic=True

def _init_fn(worker_id):
    seed = global_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return

# Make a log file & directory for saving weights
def log(text, LOGGER_FILE):
    with open(LOGGER_FILE, 'a') as f:
        f.write(text)
        f.close()

LOGGER_FILE = './training_log.txt'
weight_path = "./weights/best_checkpoint.pt"
if os.path.exists(LOGGER_FILE):
    os.remove(LOGGER_FILE)

if not os.path.exists("./weights/"):
    os.mkdir("./weights/")

# Data Loader
root_dir = "./data/sample"
h = 768
w = 512
image_size = (w, h) # (w, h)
mean = [2.0, 7.0, 20.0, 20.0, 10.0, 0.0, 0.0, 0.0]
std = [0.8, 2.0, 10.0, 10.0, 30.0, 20.0, 30.0, 1.0]
mean = np.array(mean)[:, np.newaxis][:, np.newaxis]
std = np.array(std)[:, np.newaxis][:, np.newaxis]

train_dataset = PairAnimeDataset(root_dir, image_size, mean, std, args.feature_h, args.feature_w)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers = args.num_workers,
                                           worker_init_fn = _init_fn)

valid_dataset = train_dataset
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                           batch_size=1,
                                           shuffle=False, num_workers = args.num_workers)

# Instantiate model
net = SFNet(args.feature_h, args.feature_w, beta=args.beta, kernel_sigma = args.kernel_sigma)

# Load pre-trained weight
if os.path.exists(weight_path):
    try:
        best_weights = torch.load(weight_path, map_location='cpu')
        adap3_dict = best_weights['state_dict1']
        adap4_dict = best_weights['state_dict2']
        net.adap_layer_feat3.load_state_dict(adap3_dict, strict=False)
        net.adap_layer_feat4.load_state_dict(adap4_dict, strict=False)
    except Exception as e:
        print ('exception while loading weight from pre-trained, detail:', str(e))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# Instantiate loss
criterion = loss_function(args).to(device)

# Instantiate optimizer
param = list(net.adap_layer_feat3.parameters())+list(net.adap_layer_feat4.parameters())
optimizer = torch.optim.Adam(param, lr=args.lr)
decay_schedule = list(map(lambda x: int(x), args.decay_schedule.split('-')))
scheduler = lrs.MultiStepLR(optimizer, milestones = decay_schedule, gamma = args.gamma)

# PCK metric from 'https://github.com/ignacio-rocco/weakalign/blob/master/util/eval_util.py'
def correct_keypoints(source_points, warped_points, L_pck, alpha=0.1):
    # compute correct keypoints
    p_src = source_points[0,:]
    p_wrp = warped_points[0,:]

    N_pts = torch.sum(torch.ne(p_src[0,:],-1)*torch.ne(p_src[1,:],-1))
    point_distance = torch.pow(torch.sum(torch.pow(p_src[:,:N_pts]-p_wrp[:,:N_pts],2),0),0.5)
    L_pck_mat = L_pck[0].expand_as(point_distance)
    correct_points = torch.le(point_distance, L_pck_mat * alpha)
    pck = torch.mean(correct_points.float())
    return pck

# Training
best_pck = 0
print('Training started')
for ep in range(args.epochs):
    print('Current epoch : %d' % ep)
    log('Current epoch : %d\n' % ep, LOGGER_FILE)
    log('Current learning rate : %e\n' % optimizer.state_dict()['param_groups'][0]['lr'], LOGGER_FILE)

    net.train()
    net.feature_extraction.eval()
    total_loss = 0
    for i, batch in enumerate(train_loader):
        src_image = batch['image1'].to(device)
        tgt_image = batch['image2'].to(device)
        GT_src_mask = batch['mask1'].to(device)
        GT_tgt_mask = batch['mask2'].to(device)

        output = net(src_image, tgt_image, GT_src_mask, GT_tgt_mask)

        optimizer.zero_grad()
        loss, L1, L2, L3 = criterion(output, GT_src_mask, GT_tgt_mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print("Epoch %03d (%04d/%04d) = Loss : %5f (Now : %5f)\t" % (ep, i, len(train_dataset) // args.batch_size, total_loss / (i + 1), loss.cpu().data))
        print("L1 : %5f, L2 : %5f, L3 : %5f\n" % (L1.item(), L2.item(), L3.item()))

    scheduler.step()
    print("Epoch %03d finished... Average loss : %5f\n"%(ep,total_loss/len(train_loader)))

    torch.save({'state_dict1': net.adap_layer_feat3.state_dict(),
                'state_dict2': net.adap_layer_feat4.state_dict()},
               weight_path)
                
print('Done')