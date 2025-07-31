import monai
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from scipy import ndimage
from sklearn.metrics import confusion_matrix
from torch import optim
from torch.nn import init
from torch.optim import lr_scheduler
from tqdm import tqdm

from models.AtlasNet import AtlasNet, BasicBlock
from models.MultiAtlasNet import MultiAtlasNet


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------


def kl_divergence(A, B):
    return F.kl_div(F.log_softmax(B, dim=-1), F.softmax(A, dim=-1), reduction='batchmean')




def balanced_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    # 提取 TP, FN, TN, FP
    TN, FP, FN, TP = cm.ravel()

    # 计算 TPR 和 TNR
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)

    # 计算 Balanced Accuracy
    balanced_acc = (TPR + TNR) / 2

    return balanced_acc

def save_results_to_csv(val_acc_list, val_loss_list, val_f1_score_list, val_auc_list, val_spe_list, val_precision_list, val_recall_list,result_dir):
    # 确保所有列表长度相同
    num_folds = len(val_acc_list)
    assert num_folds == len(val_loss_list) == len(val_f1_score_list) == len(val_auc_list) == len(val_spe_list) == len(val_precision_list) == len(val_recall_list), "All lists must have the same length"

    # 创建一个DataFrame来存储结果
    results_df = pd.DataFrame({
        'fold': [f'fold_{i}' for i in range(num_folds)],
        'val_acc': val_acc_list,
        'val_loss': val_loss_list,
        'val_f1_score': val_f1_score_list,
        'val_auc': val_auc_list,
        'val_spe': val_spe_list,
        'val_precision': val_precision_list,
        'val_recall': val_recall_list
    })

    # 将结果存储到CSV文件中
    results_df.to_csv(result_dir+'/results.csv', index=False)

    # 计算summary结果
    summary_df = results_df.describe().transpose()[['mean', 'std']].reset_index()
    summary_df = summary_df.rename(columns={'index': 'metric'})

    # 将summary结果存储到CSV文件中
    summary_df.to_csv(result_dir+'/summary.csv', index=False)
    print(summary_df)

import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epoch_count - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr



def get_scheduler(optimizer, opt):
    """
    scheduler definition
    :param optimizer:  原始优化器
    :param opt: 对应参数
    :return: 对应scheduler
    """
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

        # Return the LambdaLR scheduler
    elif opt.lr_policy == 'lambda_exp':
        def lambda_rule(epoch):
            if epoch < opt.warmup_epochs:
                # 预热阶段：学习率从 0 增加到 1
                lr_l = min(1,(epoch+1) / opt.warmup_epochs)
            else:
                # 衰减阶段：指数衰减，最小学习率为 min_lr
                lr_l = max(0.02, 1.0 * (opt.lr_decay ** (epoch - opt.warmup_epochs)))
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif opt.lr_policy == 'lambda_cosine':
        def lambda_rule(epoch):
            if epoch < opt.warmup_epochs:
                lr_l = epoch / opt.warmup_epochs
            else:
                lr_l = max(1e-5, 0.5 * \
                            (1. + math.cos(
                                math.pi * (epoch - opt.warmup_epochs) / (opt.epoch_count - opt.warmup_epochs))))
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch_count, eta_min=0)
    elif opt.lr_policy == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.lr_decay)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')


            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)



def cudanet(net,  gpu_ids=''):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net)
        net.cuda()
    return net


def load_checkpoint(net,pretrained=None):
    if isinstance(pretrained, str):
        print('pretraining...')
        print('load model from:{}'.format(pretrained))
    #             # Directly load 3D model.
        checkpoint_model = torch.load(pretrained,map_location='cpu')
        state_dict = net.state_dict()
        net.load_state_dict(checkpoint_model)
        print('load ending...')
    return net


def define_Cls(netCls,class_num=4, lambda_init=0.6,init_type='normal', init_gain=0.02,pretrained=None,gpu_ids=[]):
    if netCls == 'multi_atlas':
        net=MultiAtlasNet(BasicBlock,
                 [3, 3],
                 shortcut_type='B',
                 num_classes=class_num,
                 num_heads=8,
                 dim_latent=256,
                 num_atlas=56,
                 lambda_init=lambda_init
                 )
    elif netCls == 'single_atlas':
        from models.AtlasNet import AtlasNet
        net = AtlasNet(BasicBlock,
                                [3, 3],
                                shortcut_type='B',
                                num_classes=class_num,
                                num_heads=8,
                                dim_latent=256,
                                num_atlas=56,
                                lambda_init=lambda_init
                                )
    elif netCls== 'resnet':
        from comp_models.resnet import resnet18
        net=resnet18(sample_input_D=192,sample_input_H=224,sample_input_W=192)
    elif netCls == 'multinet':
        from comp_models.multiresnet import MultiNet
        net = MultiNet()
    elif netCls=='i3d':
        from comp_models.I3d import InceptionI3d
        net=  InceptionI3d(2, in_channels=1)
    elif netCls=='triplenet':
        from comp_models.triplenet import TripleMRNet
        net=TripleMRNet(backbone='resnet18')
    elif netCls=='aagn':
        from comp_models.aagn import AtlasNet
        net = AtlasNet(BasicBlock,
                       [3, 3],
                       shortcut_type='B',
                       num_classes=class_num,
                       num_heads=8,
                       dim_latent=256,
                       num_atlas=56
                       )
    elif netCls=='attn_transformer':
        from comp_models.attn_transformer import AtlasNet
        net = AtlasNet(BasicBlock,
                       [3, 3],
                       shortcut_type='B',
                       num_classes=class_num,
                       num_heads=8,
                       dim_latent=256,
                       num_atlas=56
                       )
    elif netCls=='swintrans':
        from comp_models.swintrans import SwinTransformerForClassification
        net  = SwinTransformerForClassification(
            img_size=(192,224,192),
            num_classes = 2,
            in_channels=1,
            out_channels=192,
            feature_size=12,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            num_heads= (1, 1, 2, 4),
            depths=(1,1,2,2)
        )
    elif netCls=='pipnet':
        from comp_models.PIPNet import get_network,PIPNet
        network_layers = get_network(2)
        feature_net = network_layers[0]
        add_on_layers = network_layers[1]
        pool_layer = network_layers[2]
        classification_layer = network_layers[3]
        num_prototypes = network_layers[4]

        net = PIPNet(
            num_classes=2,
            num_prototypes=num_prototypes,
            feature_net=feature_net,
            add_on_layers=add_on_layers,
            pool_layer=pool_layer,
            classification_layer=classification_layer
        )
    elif netCls=='madformer':
        from comp_models.Madformer import MADFormer_fun
        net=MADFormer_fun()
        #112 #128 #112
    elif netCls=='M3T':
        from comp_models.M3T import M3T
        net=M3T()
    elif netCls=='major_voting':
        from comp_models.mynn import MajorityVoting3D
        import torchvision
        # Set the embedding dimension according to the model
        embedding_dim = 768
        # Load the pretrained weights from torchvision
        weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        # Pass the weights to the model
        model = torchvision.models.convnext_tiny(weights=weights)
        backbone = model.features
        backbone[0][0] = nn.Conv2d(1, 96, kernel_size=4, stride=4, padding=2, bias=False)
        net=MajorityVoting3D(
        backbone, embedding_dim,
        num_slices=192, num_classes=2)
    elif netCls=='axial':
        from  comp_models.mynn import  Axial3D
        import torchvision
        # Set the embedding dimension according to the model
        embedding_dim = 768
        # Load the pretrained weights from torchvision
        weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        # Pass the weights to the model
        model = torchvision.models.convnext_tiny(weights=weights)
        backbone = model.features
        backbone[0][0] = nn.Conv2d(1, 96, kernel_size=4, stride=4, padding=2, bias=False)
        net=Axial3D(
            backbone, embedding_dim,
            num_slices=192, num_classes=2
        )
    elif netCls=='transformer':
        from comp_models.mynn import TransformerConv3D
        import torchvision
        # Set the embedding dimension according to the model
        embedding_dim = 768
        # Load the pretrained weights from torchvision
        weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        # Pass the weights to the model
        model = torchvision.models.convnext_tiny(weights=weights)
        backbone = model.features
        backbone[0][0] = nn.Conv2d(1, 96, kernel_size=4, stride=4, padding=2, bias=False)
        net= TransformerConv3D(
            backbone=backbone,
            embedding_dim=embedding_dim,
            num_slices=192,
            num_heads=8,
            num_classes=2
        )
    elif netCls=='senet':
        net=monai.networks.nets.SENet(spatial_dims=3,
                                      layers=(2, 2, 2, 2),
                                      in_channels=1,
                                      block='se_bottleneck',
                                      reduction=16,
                                      groups=1,
                                      inplanes=64,
                                      num_classes=2)
    elif netCls=='awarenet':
        from comp_models.awarenet import AwareNet
        net=AwareNet()
    elif netCls=='dcfnet':
        from comp_models.DCFNet import DCFMNet
        net=DCFMNet()
    else:
        return None
    init_weights(net, init_type, gain=init_gain)
    net = load_checkpoint(net, pretrained)
    return cudanet(net, gpu_ids)


