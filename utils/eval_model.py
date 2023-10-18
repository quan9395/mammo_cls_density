import torch
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.autograd import Variable
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def eval(model, device, have_prj, loader, metric_loss, miner, criterion, split):
    model.eval()
    print('Evaluating model on ' + split + ' data')
    birads_loss_sum = 0
    density_loss_sum = 0
    ce_loss_sum = 0
    metric_loss_sum = 0
    correct_birads = 0
    correct_density = 0
    f1_pred_birads = []
    f1_res_birads = []
    f1_pred_density = []
    f1_res_density = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            images, labels = data
            label_birads = labels[0]
            # label_density = labels[1]
            label_birads = label_birads - 1
            # label_density = label_density - 1
            b = [item.item() for item in label_birads]
            # c = [item.item() for item in label_density]
            f1_res_birads.extend(b)
            # f1_res_density.extend(c)
            # print(a)
            images = images.to(device)
            # label_birads = label_birads.to(device)
            label_density = label_density.to(device)
            if have_prj:  #not used
                p, logits = model(images)
                pminer = miner(p, labels)
                p_mloss = metric_loss(p, labels, pminer)
                ce_loss = criterion(logits, labels)
            else:
                pred_density = model(images)
                # p_mloss = torch.tensor([0.0])
                # birads_loss = FocalLoss(gamma=4)(pred_birad, label_birads.long())
                density_loss = FocalLoss(gamma=4)(pred_density, label_density.long())
                ce_loss = density_loss
            # birads_loss_sum += birads_loss.item()
            density_loss_sum += density_loss.item()
            ce_loss_sum += ce_loss.item()
            metric_loss_sum += 0

            # pred1 = pred_birad.max(1, keepdim=True)[1]
            pred2 = pred_density.max(1, keepdim=True)[1]
            # correct_birads += pred1.eq(label_birads.view_as(pred1)).sum().item()
            correct_density += pred2.eq(label_density.view_as(pred2)).sum().item()
            # b = [item.item() for item in pred1]
            c = [item.item() for item in pred2]
            # f1_pred_birads.extend(b)
            f1_pred_density.extend(c)

    # f1_birads = f1_score(f1_res_birads, f1_pred_birads, average='macro')
    f1_density = f1_score(f1_res_density, f1_pred_density, average='macro')
    loss_avg = ce_loss_sum / (i+1)
    # birads_loss_avg = birads_loss_sum / (i+1)
    density_loss_avg = density_loss_sum / (i+1)

    metric_loss_avg = metric_loss_sum / (i+1)

    # accuracy_birads = correct_birads / len(loader.dataset)
    accuracy_density = correct_density / len(loader.dataset)

    return loss_avg, density_loss_avg, accuracy_density, f1_density
