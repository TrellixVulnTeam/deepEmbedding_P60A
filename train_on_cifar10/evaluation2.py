import os
import torch
import torch.backends.cudnn as cudnn
import torchvision_qinxiao
import torchvision.transforms as transforms
from models.resnet import ResNet18
import numpy as np
from utils_qinxiao import KNNEvaluation,progress_bar
from evaluations.NMI import NMI
from evaluations.recall_at_k import Recall_at_ks
from scipy.spatial import distance_matrix
from evaluate import *
def file_name(file_dir,suffix=None):   
    L=[]   
    for dirpath, dirnames, filenames in os.walk(file_dir):  
        for file in filenames :  
            if suffix !=None and os.path.splitext(file)[1] != suffix:
                continue
            L.append(os.path.join(dirpath, file))  
    return L 
def evaluate(model_name,embeddings,targets):
    NMI_score=NMI(embeddings,targets,n_cluster=5)
    
    Recall_k_names, Recall_k_accs=evaluate_emb(embeddings,targets)
    # return Recall_k_score
    MAP_names,MAP_accs=MAP_eval(embeddings,targets)
    
    fo = open('Fisher_same_distribution_evaluation.txt', "w")
    fo.write("============================================="+model_name+"=============================================================\n")
    fo.write("=============================================NMI=============================================================\n")
    fo.write("NMI score:"+str(NMI_score)+"\n")
    fo.write("=============================================Recall@k=========================================================\n")
    for i in range(len(Recall_k_names)):
        fo.write(Recall_k_names[i]+":  "+str(Recall_k_accs[i])+"\n")
    fo.write("=============================================MAP@k============================================================\n")
    for i in range(len(MAP_names)):
        fo.write(MAP_names[i]+":  "+str(MAP_accs[i])+"\n")
    fo.close()


if __name__ == '__main__':
    # Data
    print('==> Preparing data..')
    bs=256
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision_qinxiao.datasets.CIFAR10(root='../Embedding_data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2)

    testset = torchvision_qinxiao.datasets.CIFAR10(root='../Embedding_data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # Model
    print('==> Building model..')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = ResNet18()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # fileList=file_name('checkpoint','.t7')
    fileList=['checkpoint/Fisher2_ckpt.t7']
    for ckpt_file in fileList:
        print('==> Resuming from checkpoint '+ckpt_file)
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(ckpt_file)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        net.eval()
        with torch.no_grad():
            all_embeddings=np.zeros((len(test_loader.dataset),128))
            all_targets=np.zeros(len(test_loader.dataset))
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                _,embeddings = net(inputs)
                all_embeddings[batch_idx*bs:batch_idx*bs+embeddings.shape[0],:]=embeddings.cpu().numpy()
                all_targets[batch_idx*bs:batch_idx*bs+targets.shape[0]]=targets.cpu().numpy()
                progress_bar(batch_idx, len(test_loader))
        evaluate(ckpt_file,all_embeddings,all_targets)
        