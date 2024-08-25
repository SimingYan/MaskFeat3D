import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

NUM = 1.2#2.0
W = 1.0#10.0


def cal_loss_raw(pred, gold):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    eps = 0.2
    n_class = pred.size(1)

    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    #one_hot = F.one_hot(gold, pred.shape[1]).float()

    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)

    loss_raw = -(one_hot * log_prb).sum(dim=1)


    loss = loss_raw.mean()

    return loss,loss_raw

def mat_loss(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss



def cls_loss(pred, pred_aug, gold, pc_tran, aug_tran, pc_feat, aug_feat, ispn = True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    mse_fn = torch.nn.MSELoss(reduce=True, size_average=True)

    cls_pc, cls_pc_raw = cal_loss_raw(pred, gold)
    cls_aug, cls_aug_raw = cal_loss_raw(pred_aug, gold)
    if ispn:
        cls_pc = cls_pc + 0.001*mat_loss(pc_tran)
        cls_aug = cls_aug + 0.001*mat_loss(aug_tran)

    feat_diff = 10.0*mse_fn(pc_feat,aug_feat)
    parameters = torch.max(torch.tensor(NUM).cuda(), torch.exp(1.0-cls_pc_raw)**2).cuda()
    cls_diff = (torch.abs(cls_pc_raw - cls_aug_raw) * (parameters*2)).mean()
    cls_loss = cls_pc + cls_aug  + feat_diff# + cls_diff

    return cls_loss

def aug_loss(pred, pred_aug, gold, pc_tran, aug_tran, ispn = True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    mse_fn = torch.nn.MSELoss(reduce=True, size_average=True)

    cls_pc, cls_pc_raw = cal_loss_raw(pred, gold)
    cls_aug, cls_aug_raw = cal_loss_raw(pred_aug, gold)
    if ispn:
        cls_pc = cls_pc + 0.001*mat_loss(pc_tran)
        cls_aug = cls_aug + 0.001*mat_loss(aug_tran)
    pc_con = F.softmax(pred, dim=-1)#.max(dim=1)[0]
    one_hot = F.one_hot(gold, pred.shape[1]).float()
    pc_con = (pc_con*one_hot).max(dim=1)[0]

     
    parameters = torch.max(torch.tensor(NUM).cuda(), torch.exp(pc_con) * NUM).cuda()
    
    # both losses are usable
    aug_diff = W * torch.abs(1.0 - torch.exp(cls_aug_raw - cls_pc_raw * parameters)).mean()
    #aug_diff =  W*torch.abs(cls_aug_raw - cls_pc_raw*parameters).mean()
    aug_loss = cls_aug + aug_diff

    return aug_loss


def cls_loss_only(pred, gold, pc_tran, ispn = True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    cls_pc, cls_pc_raw = cal_loss_raw(pred, gold)
    if ispn:
        cls_pc = cls_pc + 0.001*mat_loss(pc_tran)

    cls_loss = cls_pc

    return cls_loss

def getBack(var_grad_fn, filename):
    f = open(filename, 'a')

    f.write(str(var_grad_fn))
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                f.write(str(n[0]))
                f.write('Tensor with grad found:' + str(tensor))
                f.write(' - gradient:' + str(tensor.grad))
            except AttributeError as e:
                getBack(n[0], filename)
    
    f.close()


def cosine_similarity_loss(x, y): 
    cos = torch.nn.CosineSimilarity(dim=2)
    output = cos(x, y)

    loss = (1 - output).mean()

    return loss

def abs_cosine_similarity_loss(x, y, dim=2):
    cos = torch.nn.CosineSimilarity(dim=dim)
    output = cos(x, y)
    loss = (1 - torch.abs(output)).mean()

    return loss

class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, label_smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        print('Label Smoothing:', label_smoothing)
        self.smoothing = label_smoothing
        self.confidence = 1. - self.smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=1)
        nll_loss = -logprobs.gather(dim=1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class SoftTargetCrossEntropy(nn.Module):

    def __init__(self, **kwargs):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
