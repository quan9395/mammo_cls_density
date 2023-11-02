import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from imblearn.metrics import geometric_mean_score
from utils.read_dataset import read_dataset
from collections import OrderedDict
from tqdm import tqdm
from models.resnet import resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, resnext101_64x4d, wide_resnet50_2, wide_resnet101_2

from config import input_size, batch_size, root, dataset_path
from torchsummary import summary


# device = torch.device("cuda")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# change
# pth_path = "/media/Z/toannt28/checkpoint/mt_prj_resnet152_bsz8.pth"
pth_path = r"./best_model.pth"
trainloader, testloader = read_dataset(input_size, batch_size, root, dataset_path)

# change
model = resnet50(pth_url=pth_path, pretrained=False)
checkpoint = torch.load(pth_path, map_location='cpu')
# new_state_dict = OrderedDict()
# for k, v in checkpoint['model_state_dict'].items():
#     name = k[7:]
#     new_state_dict[name] = v
new_state_dict = {}
for key, value in checkpoint['model_state_dict'].items():
    new_key = key.replace("module.", "")
    new_state_dict[new_key] = value
model.load_state_dict(new_state_dict)
if __name__ == '__main__':
    print("Model loaded!")
    model = model.to(device)

    all_preds_density = []
    all_preds_birads = []
    all_labels_birads = []
    all_labels_density = []
    # model.eval()
    with torch.no_grad():
        for _, data in enumerate(tqdm(testloader)):
            images, labels = data
            # label_birads = labels[0]
            label_density = labels[1]
            # label_birads = label_birads - 1
            label_density = label_density - 1
            images= images.to(device)
            # label_birads = label_birads.to(device)
            label_density = label_density.to(device)
            density = model(images)
            # birads = torch.nn.functional.softmax(birads, dim=1)
            # density = torch.nn.functional.softmax(density, dim=1)
            # print(birads, density)
            # pred_birads = birads.max(1, keepdim=True)[1]
            pred_density = density.max(1, keepdim=True)[1]
            # all_preds_birads.append(pred_birads)
            all_preds_density.append(pred_density)
            # all_labels_birads.append(label_birads)
            all_labels_density.append(label_density)
            # print(pred_density, label_density)
            

    # all_preds_birads = torch.cat(all_preds_birads, axis=0)
    all_preds_density = torch.cat(all_preds_density, axis=0)
    # all_labels_birads = torch.cat(all_labels_birads, axis=0)
    all_labels_density = torch.cat(all_labels_density, axis=0)
    all_preds_density, all_labels_density = all_preds_density.cpu(), all_labels_density.cpu()

    # print("birads1: ", (all_labels_birads == 0).sum().item())
    # print("birads2: ", (all_labels_birads == 1).sum().item())
    # print("birads3: ", (all_labels_birads == 2).sum().item())

    print("densityA: ", (all_labels_density == 0).sum().item())
    print("densityB: ", (all_labels_density == 1).sum().item())
    print("densityC: ", (all_labels_density == 2).sum().item())
    print("densityD: ", (all_labels_density == 3).sum().item())


    # constant for classes
    # classes1 = ('BIRADS-1','BIRADS-2','BIRADS-3')
    classes2 = ('Density A','Density B','Density C','Density D')
    # # Build confusion matrix
    # cf_matrix = confusion_matrix(all_labels_birads, all_preds_birads)
    # df_cm = pd.DataFrame(cf_matrix , index = [i for i in classes1],
    #                     columns = [i for i in classes1])
    # with pd.option_context('display.max_columns', None):
    #   print(df_cm)
    # # plt.figure(figsize = (12,7))
    # cmn = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
    # sn.heatmap(cmn, annot=True, fmt='.2f', xticklabels=[i for i in classes1], yticklabels=[i for i in classes1])
    # plt.title('Confusion Matrix')
    # plt.ylabel('Actal Values')
    # plt.xlabel('Predicted Values')
    # plt.savefig(r"./birads_cfm.png")
    
    # plt.clf()

    cf_matrix2 = confusion_matrix(all_labels_density, all_preds_density)
    df_cm2 = pd.DataFrame(cf_matrix2 , index = [i for i in classes2],
                        columns = [i for i in classes2])
    with pd.option_context('display.max_columns', None):
      print(df_cm2)

    cmn2 = cf_matrix2.astype('float') / cf_matrix2.sum(axis=1)[:, np.newaxis]
    sn.heatmap(cmn2, annot=True, fmt='.2f', xticklabels=[i for i in classes2], yticklabels=[i for i in classes2])
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.savefig(r"./density_cfm.png")

    print("Acc birads: ", accuracy_score(all_labels_birads, all_preds_birads), "f1 birads: ", f1_score(all_labels_birads, all_preds_birads, average='macro'))
    print("Acc density: ", accuracy_score(all_labels_density, all_preds_density), "f1 density: ", f1_score(all_labels_density, all_preds_density, average='macro'))
