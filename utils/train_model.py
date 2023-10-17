import os
import torch
from tqdm import tqdm
from utils.eval_model import eval
from torch.autograd import Variable
from utils.mixup_utils import mixup_data, mixup_criterion
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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


def train(model,
          device,
          have_prj,
          trainloader,
        #   valloader,
          testloader,
          metric_loss,
          miner,
          criterion,
          optimizer,
          scheduler,
          save_path,
          start_epoch,
          end_epoch,
          best_val_acc):

    best_acc = best_val_acc


    for epoch in range(start_epoch + 1, end_epoch + 1):
        f = open(os.path.join(save_path, 'log.txt'), 'a')
        model.train()
        print('Training %d epoch' % epoch)

        lr = next(iter(optimizer.param_groups))['lr']
        turn = True
        a = [5,6,7]
        for _, data in enumerate(tqdm(trainloader)):
            images, labels = data

            # label_birads = labels[0]
            label_density = labels[1]
            # label_birads = label_birads - 1
            label_density = label_density - 1
            # b = [item.item() for item in label_birads]
            # a.extend(b)
            # print(a)
            
            images= images.to(device)
            # label_birads = label_birads.to(device)
            label_density = label_density.to(device)
            optimizer.zero_grad()
            if have_prj:  # not used
                if turn:
                    p, logits = model(images)
                    pminer = miner(p, labels)
                    p_mloss = metric_loss(p, labels, pminer)
                    ce_loss = criterion(logits, labels)
                    total_loss = ce_loss + p_mloss
                else:
                    images, labels_a, labels_b, lam = mixup_data(images, labels, 1.0, True)
                    images, labels_a, labels_b = map(Variable, (images, labels_a, labels_b))
                    _, logits = model(images)
                    ce_loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
                    total_loss = ce_loss
                turn = not turn
            else:
                pred_density = model(images)
                # print(pred_birad)
                # print(pred_birad.shape)
                # print(label_birads)
  
                # print(pred_density.shape)
                # print(label_density)
                # birads_loss = FocalLoss(gamma=4)(pred_birad, label_birads.long())
                density_loss = FocalLoss(gamma=4)(pred_density, label_density.long())
                total_loss = density_loss

            total_loss.backward()
            optimizer.step()
            
        scheduler.step()
        
        f.write('\nEPOCH' + str(epoch) + '\n')
        # eval valset
        # val_loss_avg, val_metric_loss_avg, val_accuracy = eval(model, device, have_prj, valloader, metric_loss, miner, criterion, split='val')
        # print('Validation set: Avg Val CE Loss: {:.4f}; Avg Val Metric Loss: {:.4f}; Val accuracy: {:.2f}%'.format(val_loss_avg, val_metric_loss_avg, 100. * val_accuracy))
        # f.write('Validation set: Avg Val CE Loss: {:.4f}; Avg Val Metric Loss: {:.4f}; Val accuracy: {:.2f}% \n'.format(val_loss_avg, val_metric_loss_avg, 100. * val_accuracy))
        # eval testset
        test_loss_avg, density_loss_avg, accuracy_density, f1_density = eval(model, device, have_prj, testloader, metric_loss, miner, criterion, split='test')
        print('Test set: Avg Focal Loss: {:.4f};, Avg density Focal Loss: {:.4f}; density accuracy: {:.2f}%; f1_density: {:.2f}% '.format(test_loss_avg,density_loss_avg, 100. * accuracy_density,100. * f1_density))
        f.write('Test set: Avg Focal Loss: {:.4f};, Avg density Focal Loss: {:.4f}; density accuracy: {:.2f}%; f1_density: {:.2f}% '.format(test_loss_avg,density_loss_avg, 100. * accuracy_density,100. * f1_density))
        
        test_accuracy = accuracy_density
        # save checkpoint
        print('Saving checkpoint')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'learning_rate': lr,
            # 'val_acc': val_accuracy,
            'test_acc_birads': accuracy_birads
        }, os.path.join(save_path, 'current_model' + '.pth'))

        if test_accuracy > best_acc:
            print('Saving best model')
            f.write('\nSaving best model!\n')
            best_acc = test_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'learning_rate': lr,
                # 'val_acc': val_accuracy,
                'test_acc': test_accuracy
            }, os.path.join(save_path, 'best_model' + '.pth'))
        f.close()
