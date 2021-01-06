import argparse
import baselineUtils
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from transformer.batch import subsequent_mask
from torch.optim import Adam,SGD,RMSprop,Adagrad
from transformer.noam_opt import NoamOpt
import numpy as np
import scipy.io
import json
import pickle
import matplotlib.pyplot as plt
import random
import scipy
from numpy.linalg import svd
from sklearn.metrics.pairwise import rbf_kernel

from torch.utils.tensorboard import SummaryWriter
import quantized_TF
def rank(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    s = svd(A, compute_uv=False)
    tol = max(atol, rtol * s[0])
    rank = int((s >= tol).sum())
    return rank

def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return scipy.linalg.orth(ns)


def elem_sympoly(lambda_=None,k=None):
    # given a vector of lambdas and a maximum size k, determine the value of
    # the elementary symmetric polynomials:
    #   E(l+1,n+1) = sum_{J \subseteq 1..n,|J| = l} prod_{i \in J} lambda(i)
    
    N= len (lambda_)
    # elem_sympoly.m:6
    E= np.zeros((k + 1,N + 1))
    # elem_sympoly.m:7
    E[0,:]=1
     # elem_sympoly.m:8
    for l_ in range(k):
        for n_ in range(N):
            l = l_+1
            n = n_+1
            E[l,n]=E[l,n - 1] + lambda_[n - 1]*E[l-1, n-1]
    return E


def sample_k(lambda_=None,k=None):

    # pick k lambdas according to p(S) \propto prod(lambda \in S)
    
    # compute elementary symmetric polynomials
    E=elem_sympoly(lambda_,k)
    
    i=len(lambda_)
    remaining=k
    S = []
    while remaining > 0:

        # compute marginal of i given that we choose remaining values from 1:i
        if i == remaining:
            marg=1
        else:
            marg=np.dot(lambda_[i-1],E[remaining-1,i-1]) / E[remaining,i]
        # sample marginal
        if np.random.uniform() < marg:
            S.append(i-1) # indexing problems
            remaining=remaining - 1
        i = i - 1
    return S




def decompose_kernel(M):
    """
    Decomposes the kernel so that dpp function can sample. 
    based on: https://github.com/javiergonzalezh/dpp/blob/master/dpp/samplers/decompose_kernel.m
    
    
    this function returns:
        * M - the original kernel
        * V - eigenvectors
        * D - diagonals of eigenvalues
    """
    L = {}    
    D, V  = np.linalg.eig(M)
    L['M'] = M.copy()
    L['V'] = np.real(V.copy())
    L['D'] = np.real(D.copy())
    return L


def sample_dpp(L=None,k=None):
    """
    Sample a set from a dpp. L is the (decomposed) kernel, and k is (optionally) 
    the size of the set to return 
    """    
    if k == L['V'].shape[1]: 
        # error handling
        return list(range(k))
    if k is None:
        # choose eigenvectors randomly
        D = np.divide(L['D'], (1+L['D']))
        # check this - might just do a random.sample along axis.
        v = np.random.randint(0, L['V'].shape[1], random.choice(range(L['V'].shape[1])))
        v = np.argwhere(np.random.uniform(size=(len(D), 1) <= D))
    else:
        v = sample_k(L['D'], k)
    
    k = len(v)    
    V = L['V'][:, v]    

    # iterate
    y_index = list(range(L['V'].shape[1]))
    Y=[]
    
    for _ in range(k):
        # compute probabilities for each item
        P=np.sum(np.power(V, 2), 1)
        # sample_dpp.m:21
        # sample_dpp.m:22
        #find(rand <= cumsum(P),1)   

                
            
        P_index = [(indx, prob) for indx, prob in list(zip(range(len(P)), P)) if indx not in Y]
        P_list = [x for x, _ in P_index]
        P_norm = np.array([p for _, p in P_index])
        P_norm = P_norm/np.sum(P_norm)        
        choose_item = np.random.choice(range(len(P_list)) , 1, p=P_norm)[0]
        
        # add the index into our sampler
        Y.append(y_index[choose_item])
        if len(Y) == k:
            return Y
        
        # delete item from y_index...
        y_index.pop(choose_item)

        # update...choose a vector to elinate, lets pick randomly
        j = random.choice(range(V.shape[1]))
        Vj = V[:, j]
        V = np.delete(V, j, axis=1)
        
        # make sure we do a projection onto Vj, 
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4524741/


        """
        where Proj⊥Bi Bj is the the projection of Bj on the subspace perpendicular to Bi. 
        For Bi ≠ 0 and Bj = 0 the projection is ∥Proj⊥Bi Bj∥2 = 0.
        """
        # is orthogonal basis
        Vj_basis = nullspace(Vj)
        # project onto basis
        V = np.apply_along_axis(lambda x: np.dot(x, Vj_basis), 0, V)
        V = scipy.linalg.orth(V)



def diversity_metric(samples):
    #print(len(samples))
    final_norm = 0
    c = 0
    #print(samples)
    for i in samples:
      #print('know')
      temp_min = np.inf
      for j in samples:
        if np.sum(i-j) == 0 :
          pass
        else:
          norm = np.linalg.norm(i-j)
          if norm < temp_min:
            temp_min = norm
      if temp_min == np.inf:
        c = c+1
      else:
        #print(temp_min)
        final_norm += temp_min
    #print('final_norm_step', final_norm/(len(samples)))
    return final_norm/(len(samples)) #you have to divide 

def our_diversity_metric(preds_dict):
  final_div = 0
  for key,values in preds_dict.items() :
      final_div += diversity_metric(values)
      #print(final_div)
  #print(final_div)
  return final_div/len(preds_dict), final_div



def get_min_distance_ADE(list_arr1, list_arr2) :
    #list_arr2 is the predicted
    #list_arr1 is the GT
    list_min = []
    for arr in list_arr1 :
      #print(arr.shape[0])
      min_dist = np.inf
      for arr2 in list_arr2 :
       # print(np.sum((arr - arr2)**2))
        #print(arr-arr2)
        dist = np.sum((arr - arr2)**2)
        if dist < min_dist:
          min_dist = dist
      list_min.append(min_dist)
    #print(list_min)
    #print(np.sum(list_min))
    #print(len(list_min))
    #print(np.sum(list_min) / (len(list_min) * arr.shape[0]))
    return np.sum(list_min) / (len(list_min) * arr.shape[0])
'''
add comment
'''
def get_min_distance_FDE(list_arr1, list_arr2) :
    list_min = []
    for arr in list_arr1 :
      #print(arr.shape[0])
      min_dist = np.inf
      for arr2 in list_arr2 :
        dist = np.sum((arr[-1,:] - arr2[-1,:])**2) 
        if dist < min_dist:
          min_dist = dist
      list_min.append(min_dist)
    return np.sum(list_min) / len(list_min) 

'''
add comment
'''

def get_metrics_ADEandFDE(gts_dict, prs_dict):
  dict_of_metrics = {}
  FDE = 0
  ADE = 0
  for key,values in gts_dict.items() :
    FDE += get_min_distance_ADE(values, prs_dict[key])
    ADE += get_min_distance_FDE(values, prs_dict[key])
  #now take average
  scaling = len(gts_dict)
  return FDE/scaling, ADE/scaling, FDE, ADE


def main():
    parser=argparse.ArgumentParser(description='Train the individual Transformer model')
    parser.add_argument('--dataset_folder',type=str,default='datasets')
    parser.add_argument('--dataset_name',type=str,default='zara1')
    parser.add_argument('--obs',type=int,default=8)
    parser.add_argument('--preds',type=int,default=12)
    parser.add_argument('--emb_size',type=int,default=512)
    parser.add_argument('--heads',type=int, default=8)
    parser.add_argument('--layers',type=int,default=6)
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--batch_size',type=int,default=256)
    parser.add_argument('--delim',type=str,default='\t')
    parser.add_argument('--name', type=str, default="zara1")
    parser.add_argument('--epoch',type=str,default="00018")
    parser.add_argument('--num_samples', type=int, default="21")




    args=parser.parse_args()
    model_name=args.name

    try:
        os.mkdir('models')
    except:
        pass
    try:
        os.mkdir('output')
    except:
        pass
    try:
        os.mkdir('output/QuantizedTF')
    except:
        pass
    try:
        os.mkdir(f'models/QuantizedTF')
    except:
        pass

    try:
        os.mkdir(f'output/QuantizedTF/{args.name}')
    except:
        pass

    try:
        os.mkdir(f'models/QuantizedTF/{args.name}')
    except:
        pass

    #log=SummaryWriter('logs/%s'%model_name)

    # log.add_scalar('eval/mad', 0, 0)
    # log.add_scalar('eval/fad', 0, 0)
    device=torch.device("cuda")

    if args.cpu or not torch.cuda.is_available():
        device=torch.device("cpu")

    args.verbose=True


    ## creation of the dataloaders for train and validation

    test_dataset,_ =  baselineUtils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=False,eval=True,verbose=args.verbose)

    mat = scipy.io.loadmat(os.path.join(args.dataset_folder, args.dataset_name, "clusters.mat"))

    clusters=mat['centroids']

    model=quantized_TF.QuantizedTF(clusters.shape[0], clusters.shape[0]+1, clusters.shape[0], N=args.layers,
                   d_model=args.emb_size, d_ff=1024, h=args.heads).to(device)

    model.load_state_dict(torch.load(f'models/QuantizedTF/{args.name}/{args.epoch}.pth'))
    model.to(device)


    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    #optim = SGD(list(a.parameters())+list(model.parameters())+list(generator.parameters()),lr=0.01)
    #sched=torch.optim.lr_scheduler.StepLR(optim,0.0005)




    # DETERMINISTIC MODE
    with torch.no_grad():
        model.eval()
        gt=[]
        pr=[]
        inp_=[]
        peds=[]
        frames=[]
        dt=[]
        for id_b,batch in enumerate(test_dl):
            print(f"batch {id_b:03d}/{len(test_dl)}")
            #print('batch', batch)
            peds.append(batch['peds'])
            frames.append(batch['frames'])
            dt.append(batch['dataset'])
            scale = np.random.uniform(0.5, 2)
            # rot_mat = np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])
            n_in_batch = batch['src'].shape[0]
            speeds_inp = batch['src'][:, 1:, 2:4]
            gt_b = batch['trg'][:, :, 0:2]
            inp = torch.tensor(
                scipy.spatial.distance.cdist(speeds_inp.reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch,
                                                                                                         -1)).to(
                device)
            src_att = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
            start_of_seq = torch.tensor([clusters.shape[0]]).repeat(n_in_batch).unsqueeze(1).to(device)
            dec_inp = start_of_seq

            for i in range(args.preds):
                trg_att = subsequent_mask(dec_inp.shape[1]).repeat(n_in_batch, 1, 1).to(device)
                out = model(inp, dec_inp, src_att, trg_att)
                #print(out.shape)
                #print('out', out)
                #print('out-1', out[:,-1,:])
                #print('out-1', out[:,-1,:].shape)
                dec_inp=torch.cat((dec_inp,out[:,-1:].argmax(dim=2)),1)
                #print('dec_inp', dec_inp)
                #print("out -1argmaxxxxxxxxxxxxx", out[:,-1:].argmax(dim=2))
                #print(out[:,-1:].argmax(dim=2).shape)

            
            preds_tr_b=clusters[dec_inp[:,1:].cpu().numpy()].cumsum(1)+batch['src'][:,-1:,0:2].cpu().numpy()
           # print('dec_inp[:,1:].cpu().numpy()', dec_inp[:,1:].cpu().numpy())
            #print('dec_inp[:,1:].cpu().numpy()', dec_inp[:,1:].cpu().numpy().shape)
            #print(' cluster dec_inp[:,1:].cpu().numpy()', clusters[dec_inp[:,1:].cpu().numpy()])
            #print('dec_inp[:,1:].cpu().numpy()cumsum', dec_inp[:,1:].cpu().numpy().cumsum(1))
            gt.append(gt_b)
            pr.append(preds_tr_b)
            #print("pr", pr)

        peds=np.concatenate(peds,0)
        frames=np.concatenate(frames,0)
        dt=np.concatenate(dt,0)
        gt=np.concatenate(gt,0)
        dt_names=test_dataset.data['dataset_name']
        pr=np.concatenate(pr,0)
        mad,fad,errs=baselineUtils.distance_metrics(gt,pr)

        #log.add_scalar('eval/DET_mad', mad, epoch)
        #log.add_scalar('eval/DET_fad', fad, epoch)

        scipy.io.savemat(f"output/QuantizedTF/{args.name}/MM_deterministic.mat",{'input':inp,'gt':gt,'pr':pr,'peds':peds,'frames':frames,'dt':dt,'dt_names':dt_names})

        print("Determinitic:")
        print("mad: %6.3f"%mad)
        print("fad: %6.3f" % fad)

        print('MULITMODALITY NOW')
        # MULTI MODALITY
        num_samples= 21 #args.num_samples
        
        #print("Entered multi modality")
        model.eval()
        gt=[]
        pr_all={}
        inp_=[]
        peds=[]
        frames=[]
        dt=[]
        cluster_num = {} ##added by nina
        #fill an empty dictionary with the future predictions sample per sample
        for sam in range(num_samples):
            pr_all[sam]=[]
            cluster_num[sam] = []

        #now loop in the batches
        for id_b,batch in enumerate(test_dl):
          #  print('batch', batch)
            #print(f"batch {id_b:03d}/{len(test_dl)}")
            peds.append(batch['peds'])
            #print("peds", peds)
            #print("peds shape", len(peds))
            frames.append(batch['frames'])
            dt.append(batch['dataset'])
            scale = np.random.uniform(0.5, 2)
            # rot_mat = np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])
            n_in_batch = batch['src'].shape[0]
            #print("batch_src", batch['src'].shape)
            speeds_inp = batch['src'][:, 1:, 2:4]
            gt_b = batch['trg'][:, :, 0:2]
            gt.append(gt_b)
            inp__=batch['src'][:,:,0:2]
            inp_.append(inp__)
            inp = torch.tensor(
                scipy.spatial.distance.cdist(speeds_inp.reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch,
                                                                                                         -1)).to(
                device)
            src_att = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
            start_of_seq = torch.tensor([clusters.shape[0]]).repeat(n_in_batch).unsqueeze(1).to(device)
            #print('start of seq' , start_of_seq.shape)

            #a for in the samples
            #for sam in range(num_samples): 

            #while should_stop == False :
            dec_inp = start_of_seq #random inizialization so dec_inp[:,1] is random!!!! dim 1024, 1
   
           
            trg_att = subsequent_mask(dec_inp.shape[1]).repeat(n_in_batch, 1, 1).to(device)
                      #predict the outcome, that is iof dimension 1024, i, 1000
            out = model.predict(inp, dec_inp, src_att, trg_att)
                
                      #now We take only the last predicted probabilities
            h=out[:,-1] #just take the class here
      
            values, index= h.sort(1)
            best_out = index[:,-100:]#.to(device)
            #print('best_out', best_out)
            coord = clusters[best_out.cpu().numpy()]
            coord = np.prod(coord, axis = 2)
                      #print('cooord', coord.shape)
            M = rbf_kernel(coord.T) # (1024 , 100)
                      #print('M', M.shape)
            L = decompose_kernel(M)
                      #print('L', L)
            indx = sample_dpp(L=L, k=7) #indexes from DPP
            print(indx)
            sam = 0  
                 
            for i in range(len(indx)):
                dec_inp = start_of_seq
                #print('dec_inp' , dec_inp.shape)
                #print('best_out[:,i]', best_out[:,indx[i]].shape)
                
                dec_inp=torch.cat((dec_inp, torch.reshape(best_out[:,indx[i]],(-1,1))),1)
                #start time 2 
                trg_att = subsequent_mask(dec_inp.shape[1]).repeat(n_in_batch, 1, 1).to(device)
                out = model.predict(inp, dec_inp, src_att, trg_att)
                h=out[:,-1] #just take the class here
                values, index= h.sort(1)
                best_out2 = index[:,-50:]#.to(device)
                #print('best_out', best_out)
                coord = clusters[best_out2.cpu().numpy()]
                coord = np.prod(coord, axis = 2)
                      #print('cooord', coord.shape)
                M = rbf_kernel(coord.T) # (1024 , 100)
                      #print('M', M.shape)
                L = decompose_kernel(M)
                      #print('L', L)
                indx2 = sample_dpp(L=L, k=3) #indexes from DPP

                for j in range(len(indx2)):
                  dec_inp2=torch.cat((dec_inp, torch.reshape(best_out2[:,indx2[j]],(-1,1))),1)
                

                  for k in range(args.preds-2):
                        trg_att = subsequent_mask(dec_inp2.shape[1]).repeat(n_in_batch, 1, 1).to(device)
                              #predict the outcome, that is iof dimension 1024, i, 1000
                        out = model.predict(inp, dec_inp2, src_att, trg_att)
                        h=out[:,-1] #just take the class here
                        dec_inp2=torch.cat((dec_inp2,torch.multinomial(h,1)),1)
                  preds_tr_b=clusters[dec_inp2[:,1:].cpu().numpy()].cumsum(1)+batch['src'][:,-1:,0:2].cpu().numpy()
                  pr_all[sam].append(preds_tr_b)
                  sam += 1
                  #print('sammmm' , sam)
                        
                      

        peds=np.concatenate(peds,0)
       
        frames=np.concatenate(frames,0)
        dt=np.concatenate(dt,0)
        #print('gt', gt)
        gt=np.concatenate(gt,0)
       #print('looooooooooooooooook hereeeeeeee')
       # print('gt', gt)
       # print('gt_shape' , gt.shape)
        dt_names=test_dataset.data['dataset_name']
        #pr=np.concatenate(pr,0)
        inp=np.concatenate(inp_,0)
        samp = {}

        #print(pr_all.keys())
      
        for k in pr_all.keys():
            #print('len(pr_all)' , len(pr_all))
           # print('pr', pr_all[k])
            #print('pr_shape0', pr_all[k][0].shape)
            #print('pr_shape0', pr_all[k][1].shape)
            #print('pr_shape0', pr_all[k][2].shape)
            #print('pr_shape', len(pr_all[k]))
            #print(np.concatenate((pr_all[k][0],pr_all[k][1], pr_all[k][2])).shape)
            
            samp[k] = {}
            #samp[k]['pr'] = np.concatenate(pr_all[k], 0)
            samp[k]['pr'] = np.concatenate((pr_all[k][0],pr_all[k][1], pr_all[k][2]))
            #print(samp[k]['pr'])
            
            


            #samp[k]['mad'], samp[k]['fad'], samp[k]['err'] = baselineUtils.distance_metrics(gt, samp[k]['pr'])
        #print(samp[0].keys)
        #print(len(samp[0]['pr']))
        my_dict = {}

        for i in range(len(samp[0]['pr'])):
            my_dict[str(i)] = []

        for item in range(len(samp[0]['pr'])):
          for i in range(len(pr_all.keys())):
            my_dict[str(item)].append(samp[i]['pr'][item])

        dict_of_noisy_GT = {}
        for i in range(n_in_batch):
          dict_of_noisy_GT[str(i)]= [gt_b[i].numpy()]
        How_many_to_noise = len(my_dict['0'])
            #print('1 GT', dict_of_noisy_GT['33'])

        for key, value in dict_of_noisy_GT.items():
          for i in range(How_many_to_noise - 1):
            noise = np.random.normal(0, 1, dict_of_noisy_GT[key][0].shape)
            new_signal = dict_of_noisy_GT[key][0] + noise
            dict_of_noisy_GT[key].append(new_signal)

        ADE, FDE, ADE_unsc, FDE_unsc = get_metrics_ADEandFDE(dict_of_noisy_GT, my_dict)

        print('##########################################')   
        print ('plots')
        plt.plot(*dict_of_noisy_GT['33'][0].T, linestyle='-', c = 'g')
        for i in range(len(my_dict['33'])) :
          plt.plot(*my_dict['33'][i].T, linestyle='-', c = 'r')

        plt.show()
        print('temporary ADE', ADE)
        print('temporary FDE', FDE)
        #print(len(my_dict))
        #print(my_dict['33'])
        print('##########################################') 
        print('The final ADE in average is: ', ADE_unsc/len(my_dict) )
        print('The final FDE in average is: ', FDE_unsc / len(my_dict)  )
        print('the average diversity is', our_diversity_metric(my_dict)[0])
        print('##########################################') 

     

        #log.add_scalar('eval/MM_mad', mad_samp, epoch)
        #log.add_scalar('eval/MM_fad', fad_samp, epoch)
        preds_all_fin=np.stack(list([samp[i]['pr'] for i in range(num_samples)]),-1)
        scipy.io.savemat(f"output/QuantizedTF/{args.name}/MM_{num_samples}.mat",{'input':inp,'gt':gt,'pr':preds_all_fin,'peds':peds,'frames':frames,'dt':dt,'dt_names':dt_names})







if __name__=='__main__':
    main()

