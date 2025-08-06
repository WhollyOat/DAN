import torch
import torch.nn as nn
from torch.nn import init
from resnet import resnet50
import torch.nn.functional as F


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)

class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        x = self.visible.layer1(x)
        x = self.visible.layer2(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        x = self.thermal.layer1(x)
        x = self.thermal.layer2(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base
      

    def forward(self, x):
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


class SE(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SE, self).__init__()
        
        self.channel_attention = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False), 
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
        
    def forward(self, f, x):
        pooled = F.avg_pool2d(x, x.size()[2:])
        y = self.channel_attention(pooled)
        return x * y + f * (1 - y)
    
class SEM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEM, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, f, x):
        y = self.fc(x)
        return x * y + f * (1 - y)

class temporal_feat(nn.Module):
    def __init__(self,  dim=2048):
        super(temporal_feat, self).__init__()
       
        self.se = SEM(dim)

    def forward(self, t_x, x, num_p):

        f1 = self.se(t_x,x[0]).unsqueeze(dim=1)
        f2 = self.se(t_x,x[1]).unsqueeze(dim=1)
        f3 = self.se(t_x,x[2]).unsqueeze(dim=1)
        f4 = self.se(t_x,x[3]).unsqueeze(dim=1)
        f5 = self.se(t_x,x[4]).unsqueeze(dim=1)
        f6 = self.se(t_x,x[5]).unsqueeze(dim=1)
        f7 = self.se(t_x,x[6]).unsqueeze(dim=1)
        f8 = self.se(t_x,x[7]).unsqueeze(dim=1)
        f9 = self.se(t_x,x[8]).unsqueeze(dim=1)
        f10 = self.se(t_x,x[9]).unsqueeze(dim=1)
        f11 = self.se(t_x,x[10]).unsqueeze(dim=1)

        f = torch.cat((f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11),dim=1)
        f = f.mean(dim=1)

        f_out = torch.empty(f.size(0), 2048*5).cuda()

        for i in range(f.size(0)):
            ff = f[i].repeat(num_p[i])
            
            padding = torch.zeros((5-num_p[i])* 2048)
            padding = padding.cuda()
            padding.requires_grad_(False)
            f_out[i] = torch.cat((ff, padding), dim=0)

        return f_out
    
class image_feat(nn.Module):
    def __init__(self,  dim=512):
        super(image_feat, self).__init__()
       
        self.se = SE(dim)

    def forward(self, x):
        feat = []
        feat_mean = torch.mean(x, dim=0, keepdim=True)
        len = x.size(0)
        for i in range(len):
            f_i = self.se(feat_mean,x[i].unsqueeze(0))
            feat.append(f_i)

        f = torch.cat(feat,dim=0)
        f = torch.mean(f, dim=0, keepdim=True)

        return f

class modal_Classifier(nn.Module):
    def __init__(self, embed_dim, modal_class):
        super(modal_Classifier, self).__init__()
        hidden_size = 1024
        self.first_layer = nn.Sequential(
                nn.Conv1d(in_channels=embed_dim, out_channels=hidden_size, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True)
        )
        self.layers = nn.ModuleList()
        for layer_index in range(7):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size // 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(inplace=True)
            )
            hidden_size = hidden_size // 2  # 512-32-8
            self.layers.append(conv_block)
        self.Liner = nn.Linear(hidden_size, modal_class)

    def forward(self, latent):
        latent = latent.unsqueeze(2)
        hidden = self.first_layer(latent)
        for i in range(7):
            hidden = self.layers[i](hidden)
        style_cls_feature = hidden.squeeze(2)
        modal_cls = self.Liner(style_cls_feature)
        if self.training:
            return modal_cls  # [batch,3]
        
class MAM(nn.Module):
    def __init__(self, dim, r=8):
        super(MAM, self).__init__()
        
        self.channel_attention = nn.Sequential(
                nn.Conv2d(dim, dim // r, kernel_size=1, bias=False), 
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // r, dim, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
        self.IN = nn.InstanceNorm2d(dim, track_running_stats=False)

    def forward(self, x):
        pooled = F.avg_pool2d(x, x.size()[2:])
        mask = self.channel_attention(pooled)
        x = x * mask + self.IN(x) * (1 - mask)

        return x
    

class embed_net(nn.Module):
    def __init__(self, class_num, arch='resnet50'):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        pool_dim = 2048*5

        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck1 = nn.BatchNorm1d(pool_dim)
        self.bottleneck1.bias.requires_grad_(False)


        self.classifier = nn.Linear(pool_dim, class_num, bias=False)

        self.classifier1 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier2 = nn.Linear(pool_dim, class_num, bias=False)
        self.bottleneck1.apply(weights_init_kaiming)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.classifier1.apply(weights_init_classifier)
        self.classifier2.apply(weights_init_classifier)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(2048, 2048, 2)
        self.temporal_feat = temporal_feat()
        self.image_feat = image_feat()
        self.MAM1 = MAM(512)
        self.MAM2= MAM(512)


    def forward(self, x1, x2, nums_p1, nums_f1, len_bbox1, nums_p2, nums_f2, len_bbox2, modal=0, seq_len = 11):
        
        if modal == 1:
            # vis
            x = []
            b1 = x1.size(0)
            for i in range(b1):
                tracklet_feat_vis = []
                sample_tracklet_vis = x1[i]
                len_bbox_i_vis = len_bbox1[i]

                for j in range(seq_len):
                    len_bbox = len_bbox_i_vis[j]
                    sample_img_vis = sample_tracklet_vis[j][0:len_bbox]
                    feat_vis = self.visible_module(sample_img_vis)
                    feat_vis = self.MAM1(feat_vis)
                    feat_vis = self.image_feat(feat_vis)
                    tracklet_feat_vis.append(feat_vis)

                x.append(torch.cat(tracklet_feat_vis, dim=0))
            
            num_p = nums_p1
            x = torch.cat(x, dim=0)
        elif modal == 2:
            # ir
            x = []
            b2 = x2.size(0)
            for i in range(b2):
                tracklet_feat_ir = []
                sample_tracklet_ir = x2[i]
                len_bbox_i_ir= len_bbox2[i]

                for j in range(seq_len):
                    len_bbox = len_bbox_i_ir[j]
                    sample_img_ir = sample_tracklet_ir[j][0:len_bbox]
                    feat_ir = self.thermal_module(sample_img_ir)
                    feat_ir = self.MAM2(feat_ir)
                    feat_ir = self.image_feat(feat_ir)
                    tracklet_feat_ir.append(feat_ir)

                x.append(torch.cat(tracklet_feat_ir, dim=0))

            num_p = nums_p2
            x = torch.cat(x, dim=0)
        elif modal == 0:
            # vis
            x11 = []
            b1 = x1.size(0)
            for i in range(b1):
                tracklet_feat_vis = []
                sample_tracklet_vis = x1[i]
                len_bbox_i_vis = len_bbox1[i]

                for j in range(seq_len):
                    len_bbox = len_bbox_i_vis[j]
                    sample_img_vis = sample_tracklet_vis[j][0:len_bbox]
                    feat_vis = self.visible_module(sample_img_vis)
                    feat_vis = self.MAM1(feat_vis)
                    feat_vis = self.image_feat(feat_vis)
                    tracklet_feat_vis.append(feat_vis)

                x11.append(torch.cat(tracklet_feat_vis, dim=0))

            x11 = torch.cat(x11, dim=0)

            # ir
            x22 = []
            b2 = x2.size(0)
            for i in range(b2):
                tracklet_feat_ir = []
                sample_tracklet_ir = x2[i]
                len_bbox_i_ir= len_bbox2[i]

                for j in range(seq_len):
                    len_bbox = len_bbox_i_ir[j]
                    sample_img_ir = sample_tracklet_ir[j][0:len_bbox]
                    feat_ir = self.thermal_module(sample_img_ir)
                    feat_ir = self.MAM2(feat_ir)
                    feat_ir = self.image_feat(feat_ir)
                    tracklet_feat_ir.append(feat_ir)

                x22.append(torch.cat(tracklet_feat_ir, dim=0))
            
            x22 = torch.cat(x22, dim=0)

            num_p = torch.cat((nums_p1, nums_p2), 0)
            x = torch.cat((x11, x22), 0)
       
        x_t = self.base_resnet(x)
        x_l = self.avgpool(x_t).squeeze()
        x_l = x_l.view(x_l.size(0)//seq_len, seq_len, -1).permute(1, 0, 2)

        h0 = torch.zeros(2, x_l.shape[1], x_l.shape[2]).cuda()
        c0 = torch.zeros(2, x_l.shape[1], x_l.shape[2]).cuda()
        if self.training: self.lstm.flatten_parameters()
        output, _ = self.lstm(x_l, (h0, c0))
        tt = output[-1]
       
        x_pool = self.temporal_feat(tt,x_l, num_p)
        feat  = self.bottleneck(x_pool)

        if self.training:
            return x_pool, self.classifier(feat)
        else:
            return self.l2norm(feat)