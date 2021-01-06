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

from torch.utils.tensorboard import SummaryWriter
import quantized_TF

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
    parser.add_argument('--name', type=str, default="zara2")
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
                print(out.shape)
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
        num_samples=args.num_samples
        print("num_samples", num_samples)
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
            for sam in range(num_samples): 
                dec_inp = start_of_seq #random inizialization so dec_inp[:,1] is random!!!! dim 1024, 1
                #print('start_of_seq', start_of_seq.shape)

                #for i in 12(future points)
                for i in range(args.preds):
                    #the mask for decoder
                    #print('i', i)
                    #print('first dec inpppppp', dec_inp.shape[1])
                    trg_att = subsequent_mask(dec_inp.shape[1]).repeat(n_in_batch, 1, 1).to(device)
                    #predict the outcome, that is iof dimension 1024, i, 1000
                    out = model.predict(inp, dec_inp, src_att, trg_att)
                    #print('out.shapeeeeeeeeeee', out)
                    #now We take only the last predicted probabilities
                    h=out[:,-1] #just take the class here
                    #h is a the probs for the future_time i of the batch
                    print('h.shape', h.shape)
                    print('out.shape', out.shape)
                    #print(torch.multinomial(h,1).shape)
                    dec_inp=torch.cat((dec_inp,torch.multinomial(h,1)),1)
                    #print('last dec inppppppppp', dec_inp )
                    #dec_inp=torch.cat((dec_inp,out[:,-1:].argmax(dim=2)),1)
                    print('torch.multinomial(h,1))', torch.multinomial(h,1).shape)
                    #print('last dec inppppppppp', dec_inp.shape )
                 #   print('dec_inp', dec_inp.shape)
                #print('dec_inp', dec_inp.shape)
                #print('dec_inp[:,1:]',dec_inp[:,1:].shape)
                #print('batch[\'src\']', batch['src'][:,-1:,0:2])
                #print('hereeeeeeeeeeeeeeeeee' ,clusters[dec_inp[:,1:].cpu().numpy()])
                preds_tr_b=clusters[dec_inp[:,1:].cpu().numpy()].cumsum(1)+batch['src'][:,-1:,0:2].cpu().numpy()
              #  print('preds_tr_b', preds_tr_b.shape)
                #print("clusters" , clusters.shape)
                
                pr_all[sam].append(preds_tr_b)
                #print("pr_all", pr_all)
                #print("pr_all_shape", pr_all[0][0].shape)
                #print("cluster_num", cluster_num[0].shape)
                #print(pr_all[0])

                
                #print("pr_all_shape1", len(pr_all[0]))
                #print('clusters.shape', clusters.shape)

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
        print('####################################################')
        print('keys')
        #print(pr_all.keys())
        print('####################################################')
        for k in pr_all.keys():
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
            
            

            #dict = {'horse' : [traj1, traj2, traj3]}
           # fun (9f)
            #GT = {'horse': [traj1]}
            
            samp[k]['mad'], samp[k]['fad'], samp[k]['err'] = baselineUtils.distance_metrics(gt, samp[k]['pr'])
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

        ev = [samp[i]['err'] for i in range(num_samples)]
        e20 = np.stack(ev, -1)
        mad_samp=e20.mean(1).min(-1).mean()
        fad_samp=e20[:,-1].min(-1).mean()
        #mad,fad,errs=baselineUtils.distance_metrics(gt,pr)

        #log.add_scalar('eval/MM_mad', mad_samp, epoch)
        #log.add_scalar('eval/MM_fad', fad_samp, epoch)
        preds_all_fin=np.stack(list([samp[i]['pr'] for i in range(num_samples)]),-1)
        scipy.io.savemat(f"output/QuantizedTF/{args.name}/MM_{num_samples}.mat",{'input':inp,'gt':gt,'pr':preds_all_fin,'peds':peds,'frames':frames,'dt':dt,'dt_names':dt_names})

        print("Determinitic:")
        print("mad: %6.3f"%mad)
        print("fad: %6.3f" % fad)

        print("Multimodality:")
        print("mad: %6.3f"%mad_samp)
        print("fad: %6.3f" % fad_samp)

#here thee is our new and powerful code
#all rights reserved


'''
HERE DPP sampling (yea sure..)
'''

#1) take 1 probability vector (e.g. vec = [.2,.3,.5])
#2) define a trainable sampling function DSF (e.g. depends on some ***parameter***!)
#3) sample with the trainable DSF from vec
#4) the samples are the z1, ...zN in paper (e.g are the #num of cluster chosen)
#5) you transform z1,...zN in coordinates x1 =(x_1,y_1), ...xN=(x_N,y_N) as in paper notation
#6) Define S, r as in Paper, then L(***params***)
#7) compute loss
#8) with Gdsecent update ***parameters*** of DSF until convergence
#9) after convergence just sample stuff and enjoy free nap.
#10) hot chocolate!




if __name__=='__main__':
    main()
