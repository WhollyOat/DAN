import argparse
import time
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
#import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from eval_metrics import evaluate
from model_main import embed_net
from utils import *
from dataset import SYSUGroupDataset
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='VVIG', help='dataset name')
parser.add_argument('--lr', default=0.05 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet50')
parser.add_argument('--resume', '-r', default='xxxx.pt', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='./checkpoints/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='./logs/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='./logs/vvig_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=128, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=256, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch_size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test_batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--margin', default=0.7, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=2, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--root', default='/media/data1/xiongjh/CMGV', type=str,
                    help='dataset path')

args = parser.parse_args()
os.environ['CUDA_DEVICE_ORDER'] ='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
cudnn.benchmark = True
dataset = args.dataset

#添加

root = args.root
seq_lenth =11
test_batch = args.test_batch
log_path = "./logs"
test_mode = [1, 2]
height = args.img_h
width = args.img_w
checkpoint_path = args.model_path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

query_dataset = SYSUGroupDataset(root, mode='query', transform=transform_test)
gallery_dataset = SYSUGroupDataset(root, mode='gallery', transform=transform_test)
queryloader = DataLoader(dataset=query_dataset,
                            batch_size=test_batch,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False,
                            num_workers=8)


galleryloader = DataLoader(dataset=gallery_dataset,
                            batch_size=test_batch,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False,
                            num_workers=8)


# ----------------visible to infrared----------------
queryloader_1 = DataLoader(dataset=gallery_dataset,
                            batch_size=test_batch,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False,
                            num_workers=8)

galleryloader_1 = DataLoader(dataset=query_dataset,
                            batch_size=test_batch,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False,
                            num_workers=8)

nquery_1 = 5629
ngall_1 = 3845

n_class = 285
nquery = 3845
ngall = 5629

n_train = 18235

print('==> Building model..')

net = embed_net( n_class,  arch=args.arch)
net.to(device)
cudnn.benchmark = True

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))


def test2():
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall_1, 2048*5))
    q_pids, q_camids = [], []
    g_pids, g_camids = [], []
    with torch.no_grad():
        for batch_idx, (imgs, pids, camids, nums_p, nums_f,len_bbox) in enumerate(galleryloader_1):
            input = imgs
            input = Variable(input.cuda())
            
            batch_num = input.size(0)
            feat = net(input, input, nums_p, nums_f, len_bbox, nums_p, nums_f, len_bbox, test_mode[1], seq_len=seq_lenth)
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()

            ptr = ptr + batch_num
            #
            g_pids.extend(pids)
            g_camids.extend(camids)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery_1, 2048*5))
    with torch.no_grad():
        for batch_idx, (imgs, pids, camids, nums_p, nums_f,len_bbox) in enumerate(queryloader_1):
            input = imgs

            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = net(input, input, nums_p, nums_f, len_bbox, nums_p, nums_f, len_bbox, test_mode[0], seq_len=seq_lenth)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
        

            
            ptr = ptr + batch_num

            q_pids.extend(pids)
            q_camids.extend(camids)

    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    
    # evaluation
    cmc, mAP = evaluate(-distmat, q_pids, g_pids, q_camids, g_camids)
    
    


    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    ranks = [1, 5, 10, 20]
    print("Results ----------")
    print("testmAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r - 1]))
    print("------------------")
    return cmc,mAP

def test():
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 2048*5))
    q_pids, q_camids = [], []
    g_pids, g_camids = [], []
    with torch.no_grad():
        for batch_idx, (imgs, pids, camids, nums_p, nums_f,len_bbox) in enumerate(galleryloader):
            input = imgs
            batch_num = input.size(0)

            input = Variable(input.cuda())
            feat = net(input, input, nums_p, nums_f, len_bbox, nums_p, nums_f, len_bbox, test_mode[0], seq_len=seq_lenth)
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
        

            ptr = ptr + batch_num

            g_pids.extend(pids)
            g_camids.extend(camids)

    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 2048*5))

    with torch.no_grad():
        for batch_idx, (imgs, pids, camids, nums_p, nums_f,len_bbox ) in enumerate(queryloader):
            input = imgs

            batch_num = input.size(0)

            input = Variable(input.cuda())
            feat = net(input, input, nums_p, nums_f, len_bbox, nums_p, nums_f, len_bbox, test_mode[1], seq_len=seq_lenth)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
     

            ptr = ptr + batch_num

            q_pids.extend(pids)
            q_camids.extend(camids)

    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(-distmat, q_pids, g_pids, q_camids, g_camids)
    

    ranks = [1, 5, 10, 20]
    print("Results ----------")
    print("testmAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r - 1]))
    print("------------------")
    return cmc,mAP

# testing
print('==> Start Testing...')

#---------infrared to visible---------
cmc, mAP = test()

#---------visible to infrared---------
cmc_1, mAP_1 = test2()
# log output
print('t2v:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
    cmc[0], cmc[4], cmc[9], cmc[19], mAP))

print('v2t:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
    cmc_1[0], cmc_1[4], cmc_1[9], cmc_1[19], mAP_1))

