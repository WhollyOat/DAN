import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
#import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from eval_metrics import evaluate
from model_main import embed_net, modal_Classifier
from utils import *
from cc import CenterTripletLoss
from dataset import SYSUGroupDataset
from sampler import CrossModalityIdentitySampler
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='VVIG', help='dataset name')
parser.add_argument('--lr', default=0.05 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='checkpoints/', type=str,
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

checkpoint_path = './checkpoints/'
vis_log_path = './logs/vvig_log/'

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

# log file name
suffix = dataset

suffix = suffix + '_{}_{}_lr_{}_seed_{}'.format( args.num_pos, args.batch_size, args.lr, args.seed)
if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

test_log_file = open(log_path + suffix + '.txt', "w")
sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_acc_v2t = 0

best_map_acc = 0  # best test accuracy
best_map_acc_v2t = 0

start_epoch = 0
end = time.time()

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
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
net_modal_classifier1 = modal_Classifier(embed_dim=2048*5, modal_class=3)
net_modal_classifier1.to(device)

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

# define loss function
criterion1 = nn.CrossEntropyLoss()
loader_batch = args.batch_size * args.num_pos
criterion2 = CenterTripletLoss(k_size=args.num_pos, margin=args.margin)
criterion1.to(device)
criterion2.to(device)

# optimizer
if args.optim == 'sgd':
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.classifier.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer_P = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr},
        ],
        weight_decay=5e-4, momentum=0.9, nesterov=True)

modal_classifier_optimizer_1 = torch.optim.SGD(net_modal_classifier1.parameters(), lr=0.05, weight_decay=5e-4, momentum=0.9, nesterov=True)

def adjust_learning_rate(optimizer_P, epoch):
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif 10 <= epoch < 30:
        lr = args.lr
    elif 30 <= epoch < 60:
        lr = args.lr * 0.1
    elif epoch >= 60:
        lr = args.lr * 0.01

    optimizer_P.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer_P.param_groups) - 1):
        optimizer_P.param_groups[i + 1]['lr'] = lr
    return lr

def train(epoch):
    # adjust learning rate
    current_lr = adjust_learning_rate(optimizer_P, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    
    total = 0

    net.train()
    end = time.time()

    for batch_idx, (t_imgs, gid, cam, nums_p, nums_f,len_bbox) in enumerate(trainloader):
        sub = (cam == 4) + (cam == 5) + (cam == 6)

        input1 = t_imgs[sub == 0]
        nums_p1 = nums_p[sub == 0]
        nums_f1 = nums_f[sub == 0]
        len_bbox1 = len_bbox[sub == 0]

        input2 = t_imgs[sub == 1]
        nums_p2 = nums_p[sub == 1]
        nums_f2 = nums_f[sub == 1]
        len_bbox2 = len_bbox[sub == 1]

        labels = gid
        
        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())

        labels = Variable(labels.cuda())
        data_time.update(time.time() - end)

        modal_3_labels = Variable(2 * torch.ones(int(loader_batch/2)).long().cuda())

        feat, out0 = net(input1, input2, nums_p1, nums_f1, len_bbox1, nums_p2, nums_f2, len_bbox2, seq_len=seq_lenth)

        loss_id = criterion1(out0, labels)
        loss_tri, _,_ = criterion2(feat, labels)
        

        if epoch < 20:
            loss = loss_id + loss_tri

            loss_total = loss

            # optimization
            optimizer_P.zero_grad()
            loss_total.backward()
            optimizer_P.step()

        else:

            out2 = net_modal_classifier1(feat)
            loss2 = criterion1(out2[:int(loader_batch/2)], modal_3_labels) + criterion1(out2[int(loader_batch/2):],
                                                                                        modal_3_labels)

            loss = loss_id + loss_tri + loss2
            loss_total = loss

            # optimization
            optimizer_P.zero_grad()
            loss_total.backward()
            optimizer_P.step()

            if batch_idx % 10 == 0:
                # print('modal_loss: ' + str(modal_loss.cpu().detach().numpy()))
                print('loss2: ' + str(loss2.cpu().detach().numpy()))

        # log different loss components
        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        tri_loss.update(loss_tri.item(), 2 * input1.size(0))
        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 10 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'lr:{} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  
                  'Accu: {:.2f}'.format(
                   epoch, batch_idx, len(trainloader), current_lr,
                   1, batch_time=batch_time,
                   train_loss = train_loss, id_loss = id_loss, tri_loss = tri_loss))
    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('tri_loss', tri_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)
    return 1. / (1. + train_loss.avg)

def test2(epoch):
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
    print("testmAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")
    return cmc,mAP

def test(epoch):
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
    print("testmAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")
    return cmc,mAP

# training
print('==> Start Training...')

for epoch in range(start_epoch, 200 - start_epoch):

    print('==> Preparing Data Loader...')

    train_dataset = SYSUGroupDataset(root, mode='train', transform = transform_train)
    loader_batch = args.batch_size * args.num_pos
    print('len(train_dataset.ids)',len(train_dataset.ids))
    id_list =[]

    for id in set(train_dataset.ids):
            id_list_ir = [train_dataset.ids[i] for i in range(len(train_dataset.ids)) if train_dataset.ids[i]== id and train_dataset.cam_ids[i] in [4, 5, 6]]
            id_list_rgb= [train_dataset.ids[i] for i in range(len(train_dataset.ids)) if train_dataset.ids[i]== id and train_dataset.cam_ids[i] in [1, 2, 3]]
            if len(id_list_ir)<=len(id_list_rgb):
                id_list.extend(id_list_ir)
            else:
                id_list.extend(id_list_rgb)
    print('len(id_list)=', len(id_list))
    sampler = CrossModalityIdentitySampler(train_dataset, id_list, args.batch_size, args.num_pos)
    


    trainloader = DataLoader(train_dataset, loader_batch, sampler = sampler, drop_last=True, pin_memory=True,
                            num_workers=8)
    print('len(trainloader)=', len(trainloader))

    id_list.clear()

    # training
    wG = train(epoch)

    if epoch >=0 and epoch % 10 == 0:
        print('Test Epoch: {}'.format(epoch))
        print('Test Epoch: {}'.format(epoch), file=test_log_file)

        # testing
        cmc, mAP = test(epoch)

        state1 = {
                'net': net.state_dict(),
                'net_modal_classifier': net_modal_classifier1.state_dict(),
                'mAP': mAP,
                'epoch': epoch,
            }
        torch.save(state1, checkpoint_path + str(epoch) + 't2v.pt')

        if cmc[0] > best_acc:
            best_acc = cmc[0]
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'net_modal_classifier': net_modal_classifier1.state_dict(),
                'mAP': mAP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + 't2v_rank1_best.pt')

        if mAP > best_map_acc:
            best_map_acc = mAP
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'net_modal_classifier': net_modal_classifier1.state_dict(),
                'mAP': mAP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + 't2v_map_best.pt')

        print(
            'FC(t2v):   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        print('Best t2v epoch [{}]'.format(best_epoch))
        print(
            'FC(t2v):   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP), file=test_log_file)
#-------------------------------------------------------------------------------------------------------------------
        cmc, mAP = test2(epoch)
        if cmc[0] > best_acc_v2t:
            best_acc_v2t = cmc[0]
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'net_modal_classifier': net_modal_classifier1.state_dict(),
                'mAP': mAP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + 'v2t_rank1_best.pt')

        if mAP > best_map_acc_v2t:
            best_map_acc_v2t = mAP
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'net_modal_classifier': net_modal_classifier1.state_dict(),
                'mAP': mAP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + 'v2t_map_best.pt')

        print(
            'FC(v2t):   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        print('Best v2t epoch [{}]'.format(best_epoch))
        print(
            'FC(v2t):   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP), file=test_log_file)


        test_log_file.flush()
