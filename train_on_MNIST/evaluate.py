import numpy as np
from sklearn.cluster import KMeans
import math
from scipy.spatial import distance_matrix

def evaluate_emb(emb,labels):
    """Evaluate embeddings based on Recall@k."""
    d_mat = distance_matrix(emb,emb)
    # d_mat = d_mat.asnumpy()
    # labels = labels.asnumpy()

    names = []
    accs = []
    for k in [1, 2,4, 8, 16]:
        names.append('Recall@%d' % k)
        correct, cnt = 0.0, 0.0
        for i in range(emb.shape[0]):
            d_mat[i, i] = 1e10
            nns = np.argpartition(d_mat[i], k)[:k]
            if np.array([labels[i] == labels[nn] for nn in nns]).any():
                correct += 1
            cnt += 1
        accs.append(correct/cnt)
    return names, accs

# def NMI(A,B):
#     # len(A) should be equal to len(B)
#     total = len(A)
#     A_ids = set(A)
#     B_ids = set(B)
#     #Mutual information
#     MI = 0
#     eps = 1.4e-45
#     for idA in A_ids:
#         for idB in B_ids:
#             idAOccur = np.where(A==idA)
#             idBOccur = np.where(B==idB)
#             idABOccur = np.intersect1d(idAOccur,idBOccur)
#             px = 1.0*len(idAOccur[0])/total
#             py = 1.0*len(idBOccur[0])/total
#             pxy = 1.0*len(idABOccur)/total
#             MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
#     # Normalized Mutual information
#     Hx = 0
#     for idA in A_ids:
#         idAOccurCount = 1.0*len(np.where(A==idA)[0])
#         Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
#     Hy = 0
#     for idB in B_ids:
#         idBOccurCount = 1.0*len(np.where(B==idB)[0])
#         Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
#     MIhat = 2.0*MI/(Hx+Hy)
#     return MIhat

# def NMI_eval(emb,labels):
#     estimator = KMeans(n_clusters=np.amax(labels)+1)
#     estimator.fit(emb)
#     label_pred = estimator.labels_ 
#     score=NMI(labels,label_pred)
#     return score

def get_AP(a):
    b=np.argwhere(a==a[0])
    b=b[:,0]+1
    return np.sum((np.arange(len(b))+1)*1.0/b)/len(b)
def MAP_eval(features,ys):
    names = []
    accs = []
    ks=[64,256,1024,2048,4096]
    sum_AP=np.array([0.0,0.0,0.0,0.0,0.0])
    for i in range(features.shape[0]):
        sim=np.sqrt(np.sum(np.square(features-features[i,:]),axis=1))
        sim_idx=np.argsort(sim,axis=0)
        sorted_ys=ys[sim_idx]
        idx=0
        for k in ks:
            sum_AP[idx]+= get_AP(sorted_ys[:np.minimum(features.shape[0],k)])
            idx+=1
            
    idx=0
    for k in ks:
        names.append('MAP@%d' % k)
        accs.append(sum_AP[idx]/features.shape[0])
        idx+=1
    return names,accs
