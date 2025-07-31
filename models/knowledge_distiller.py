import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.dis_recon_loss import cross_expert_loss
from utils.tools import kl_divergence


def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
    """
    Label smoothing implementation.
    This function is taken from https://github.com/MIT-HAN-LAB/ProxylessNAS/blob/master/proxyless_nas/utils.py
    """

    logsoftmax = nn.LogSoftmax()
    n_classes = pred.size(1)
    # convert to one-hot
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros_like(pred)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))

class AtlasDistiller(nn.Module):
    def __init__(self, opt, t_net, s_net,label_smooth=False):
        super(AtlasDistiller, self).__init__()

        self.t_net = t_net
        self.s_net = s_net
        self.t_net.requires_grad = False
        self.t_net.eval()

        self.T_label = 2
        self.T_ra=2
        self.lda_soft= opt.lda_soft
        self.lda_feat =opt.lda_feat
        self.lda_attn=opt.lda_attn
        self.lda_sh=opt.lda_sh
        self.lda_dis=opt.lda_dis
        self.lda_rec= opt.lda_rec
        self.C1=1024
        self.C2=512
        self.W1 = nn.Parameter(torch.ones(self.C1//2, self.C2//2))
        self.W2 = nn.Parameter(torch.ones(self.C1 // 2, self.C2 // 2))
        self.softmax = nn.Softmax(dim=1)

        if label_smooth:
            self.hard_loss = cross_entropy_with_label_smoothing
        else:
            self.hard_loss = nn.CrossEntropyLoss()
            self.hard_loss = self.hard_loss

    def rearrange_cross(self,A, B):
        batch, C = A.shape  # 假设 C1 == C2，简化为 C
        B_expanded = B.unsqueeze(0).expand(batch, *B.shape)  # 扩展 B 的维度
        temp_B = B_expanded.clone()

        # 创建 mask
        mask = torch.eye(batch, device=A.device).bool()
        mask = mask.unsqueeze(2).expand(batch, batch, C)

        # 使用 scatter 替换对角线元素
        temp_B = torch.where(mask, A.unsqueeze(0).expand(batch, batch, C), temp_B)

        return temp_B



    def gram_calculate(self,A,with_l2_norm=False):
        A= F.normalize(A, p=2, dim=-1)
        if len(A.shape) == 3:
            det_matrices = torch.matmul(A, A.transpose(1, 2))
            if with_l2_norm:
                det_matrices = det_matrices / torch.sqrt(torch.diagonal(det_matrices, offset=0, dim1=1, dim2=2)[:,:, None])

        else:
            det_matrices = torch.matmul(A, A.t())
            if with_l2_norm:
                det_matrices = det_matrices / torch.sqrt(torch.diag(det_matrices)[ :, None])
        return det_matrices
        # det
        # return None,torch.det(det_matrices),None
        # svd
        # torch.linalg.svd(det_matrices,full_matrices=True))

    def CKA(self,gram_X, gram_Y):
        if len(gram_X.shape)==2:
            # compute cka
            cross_trace =torch.trace(torch.matmul(gram_X, gram_Y.t()))
            norm_X= torch.trace(torch.matmul(gram_X, gram_X.t()))
            norm_Y=torch.trace(torch.matmul(gram_Y, gram_Y.t()))
        else:
            # [batch, m, m] @ [batch, m, m].transpose -> [batch]
            cross_trace =  torch.diagonal(torch.einsum('bij,bjk->bik', gram_X, gram_Y.transpose(-1, -2)), dim1=1, dim2=2).sum(dim=1)

            # 归一化
            norm_X = torch.diagonal(torch.einsum('bij,bjk->bik', gram_X, gram_X.transpose(-1, -2)), dim1=1, dim2=2).sum(dim=1)
            norm_Y = torch.diagonal(torch.einsum('bij,bjk->bik', gram_Y, gram_Y.transpose(-1, -2)), dim1=1, dim2=2).sum(dim=1)

            # CKA计算
        cka =1-( cross_trace / torch.sqrt(norm_X * norm_Y)).mean()

        return cka
    def relation_loss(self, f_t, f_s):
        with torch.no_grad():
             S_t2t= self.gram_calculate(f_t.detach()) #t2t


        # student_volumetric = self.volumentric_calculate(f_s)
        f_t_transformed1=torch.matmul(f_t[:,:self.C1//2], self.W1)
        f_t_transformed2= torch.matmul(f_t[:, self.C1 // 2:], self.W2)
        S_t2s = self.gram_calculate(self.rearrange_cross(torch.cat((f_t_transformed1,f_t_transformed2),dim=-1), f_s))  # t2s

        f_s_transformed1 = torch.matmul(f_s[:, :self.C2 // 2], self.W1.t())
        f_s_transformed2 = torch.matmul(f_s[:, self.C2 // 2:], self.W2.t())
        S_s2t=self.gram_calculate(self.rearrange_cross(torch.cat((f_s_transformed1,f_s_transformed2),dim=-1), f_t))  #s2t

        S_s2s = self.gram_calculate(f_s) #s2s

        # 计算损失
        # [vanilla]
        cross_s2t_loss = self.CKA(S_t2s,S_t2t.unsqueeze(0).expand(S_t2s.shape[0],S_t2t.shape[0],S_t2t.shape[1]))
        cross_t2s_loss =self.CKA(S_s2t,S_t2t.unsqueeze(0).expand(S_s2t.shape[0],S_t2t.shape[0],S_t2t.shape[1]))
        cross_s2s_loss =self.CKA(S_s2s,S_t2t)
        #     F.mse_loss(S_t2t.unsqueeze(0).expand(S_t2s.shape[0],S_t2t.shape[0],S_t2t.shape[1]), S_t2s))
        # cross_t2s_loss = F.mse_loss(S_s2t, S_t2t.unsqueeze(0).expand(S_s2t.shape[0],S_t2t.shape[0],S_t2t.shape[1]))
        # cross_s2s_loss = F.mse_loss(S_s2s,S_t2t)
        # cross_s2t_loss = F.mse_loss(S_A.unsqueeze(0).expand(S_A_faked.shape[0], S_A.shape[0], S_A.shape[1]), S_A_faked)
        # cross_t2s_loss = F.mse_loss(S_B,
        #                             S_B_faked.unsqueeze(0).expand(S_B.shape[0], S_B_faked.shape[0], S_B_faked.shape[1]))
        # cross_s2s_loss=F.mse_loss(S_B,S_A)
        # [det]
        # cross_s2t_loss = F.mse_loss(S_t2t.expand(S_t2s.shape[0]), S_t2s)
        # cross_t2s_loss = F.mse_loss(S_s2t, S_t2t.expand(S_s2t.shape[0]))
        # cross_s2s_loss = F.mse_loss(S_s2s,S_t2t)
        # [svd]
        # cross_s2t_loss = F.mse_loss(S_A.unsqueeze(0).expand(S_A_faked.shape[0],S_A.shape[0]), S_A_faked)
        # cross_t2s_loss = F.mse_loss( S_B, S_B_faked.unsqueeze(0).expand(S_B.shape[0],S_B_faked.shape[0]))

        total_loss = cross_s2t_loss + cross_t2s_loss+ cross_s2s_loss


        return total_loss, cross_t2s_loss, cross_s2t_loss,cross_s2s_loss


    def forward(self, mri_images, pet_images, atlases, label, type='multi'):
        # 提取学生网络特征
        logit_s, fea_s, expert_outputs, shared_outputs, expert_inputs, recon_outputs, ra_s = self.s_net(mri_images,
                                                                                                        atlases)

        # 硬标签损失
        hard_loss = self.hard_loss(logit_s, label)

        # 专家损失
        loss_share, loss_distenglement, loss_recon = cross_expert_loss(
            shared_outputs, expert_outputs, recon_outputs, expert_inputs
        )

        if type == 'multi':
            # 使用 no_grad 和 detach 避免重复计算
            with torch.no_grad():
                logit_t, fea_t, _, _, _, _, ra_t = self.t_net(mri_images, pet_images, atlases)

            soft_loss = kl_divergence(logit_s / self.T_label, logit_t / self.T_label)
            #(self.T_label**2)*
            # 注意力蒸馏损失
            attn_distill_loss = F.kl_div(torch.log(ra_s), ra_t.detach(), reduction='batchmean')
            # 特征关系损失
            fea_loss, cross_t2s_loss, cross_s2t_loss,cross_s2s_loss = self.relation_loss(fea_t, fea_s)



            # 综合损失
            loss = (
                    hard_loss +
                    self.lda_sh * loss_share +
                    self.lda_dis * loss_distenglement +
                    self.lda_rec * loss_recon +
                    self.lda_soft * soft_loss +
                    self.lda_feat * fea_loss +
                    self.lda_attn * attn_distill_loss
            )

            return (
                logit_s, loss, hard_loss,
                self.lda_sh * loss_share,
                self.lda_dis * loss_distenglement,
                self.lda_rec * loss_recon,
                self.lda_soft * soft_loss,
                self.lda_feat * fea_loss,
                self.lda_feat * cross_t2s_loss,
                self.lda_feat * cross_s2t_loss,
                self.lda_feat * cross_s2s_loss,
                self.lda_attn * attn_distill_loss
            )

        else:
            loss = (
                    hard_loss +
                    self.lda_sh * loss_share +
                    self.lda_dis * loss_distenglement +
                    self.lda_rec * loss_recon
            )
            return (
                logit_s, loss, hard_loss,
                self.lda_sh * loss_share,
                self.lda_dis * loss_distenglement,
                self.lda_rec * loss_recon
            )
