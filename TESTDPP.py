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
from quantized_TFsamples import QuantizedTF
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import quantized_TF


np.random.seed(seed=7)
torch.seed()
#we write a path creator

'''here the fun to run in every batch, it will return a dict with all paths for each element of the batch
'''
def convertfunc(list_of_tensor, clusters, batch_src, key):
  final_list = []
  #print('ONE LOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP')
  for tens in list_of_tensor:
    #print('THE SHAPE OF THE TENSOR')
    #print(tens.shape)
    #print('WHAT I PUT INSIDEEEEEEEEEEEEEEEEEEEEEE')
    #print(tens.cpu().numpy())
    #print(clusters[tens.cpu().numpy()].cumsum(1)+batch_src[key,-1:,0:2].cpu().numpy())
    #print('BYEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE')
    final_list.append(clusters[tens.cpu().numpy()].cumsum(1)+batch_src[key,-1:,0:2].cpu().numpy())
  return(final_list)

def Path_creator(n_in_batch, inp, dec_inp, src_att, model, device, temperature = 0.1, dim_temp = 12):

    '''
    first we write the rule for selecting #samples at each future step
    #temp ranges between 0 and 1
    '''
    assert temperature <= 0.5
    assert temperature > 0

    num_samp_times = torch.zeros(dim_temp, dtype=torch.uint8)
    for i in range(dim_temp): 
        num = temperature*(dim_temp*dim_temp)//((i+2)**2)
        if num == 0:
            num_samp_times[i] = 1.
        else:
            num_samp_times[i] = temperature*(dim_temp*dim_temp)//((i+2)**2)
    '''
    So our sampling rule is the tensor num_samp_times of dim dim_temp, call the singletons with .item()
    '''

    our_rule = num_samp_times #[7, 3, 1, 1, 1, 1, 1, 1, 1, 1]
    #our_rule = torch.tensor([1,1,1,1,1,1,1,1,1,1,1,1])
    print('How many samples for each input', torch.prod(our_rule, 0).item())
    '''
    create the storing place(global) and the final dict structure
    '''

    storing_place = [torch.empty((n_in_batch, 1)).to(device) for i in range(dim_temp)]
    final_dict = {}
    #here for each i there will be stores the paths as lists
    for i in range(n_in_batch):
      final_dict[str(i)]= []

    '''
    now we prepare everything, and save first step in the store_place
    '''

    '''call the recursive patheator'''
    #but where do we start calling it?
    step = 0
    trg_att = subsequent_mask(dec_inp.shape[1]).repeat(n_in_batch, 1, 1).to(device)
    out, out_clusters = model.predict(inp, dec_inp, src_att, trg_att)  
    input_columns = out_clusters[:, :our_rule[0]].to(device)  #our_rule[0]
    storing_place[step] = torch.cat((storing_place[0],input_columns), dim = 1)
    recursive_patheator(dec_inp, step + 1, input_columns, n_in_batch, our_rule, inp, src_att, model,  final_dict, storing_place, device)
    #print('###############################################')
    #print('Computing the final dict')#, final_dict)
    #print('###############################################')
    return final_dict

    
    '''modify function that transformrs in paths'''
    '''compute metrics'''
    '''cry'''
    
    
'''
Here the recursive function that creates the paths
'''
def recursive_patheator(dec_inp, step, input_columns, num_in_batch, our_rule, inp, src_att, model, final_dict, storing_place, device):
  
  if step == 12:
    dec_inp = torch.cat((dec_inp, input_columns), dim = 1)
    #print('dec_inp shape', dec_inp.shape)
    assert dec_inp.shape[1] == our_rule.shape[0] + 1
    for i in range(num_in_batch):
        #global final_dict
        final_dict[str(i)].append(dec_inp[i, 1:])
    return 
  else:
    #print('dec_inp shape', dec_inp.shape)
    for i in range(input_columns.shape[1]):
        #print(dec_inp.shape, input_columns[:, i].shape)
        dec_inp_next = torch.cat((dec_inp, input_columns[:, i].reshape(-1, 1)), dim = 1).to(device)
        #run the model
        trg_att = subsequent_mask(dec_inp_next.shape[1]).repeat(num_in_batch, 1, 1).to(device)
        out, out_clusters = model.predict(inp, dec_inp_next, src_att, trg_att )
        #print('step', step)
        #print('our_rule(step)', our_rule[step])
        new_columns = out_clusters[:, :our_rule[step]].to(device)
        #print('the new columns shape', new_columns.shape)
        #global storing_place
        #print('fucking shapes', storing_place[step].shape , new_columns.shape)
        storing_place[step] = torch.cat((storing_place[step], new_columns), dim = 1)
        recursive_patheator(dec_inp_next, step + 1, new_columns, num_in_batch, our_rule, inp, src_att, model,final_dict, storing_place, device)
'''
Here are functions for getting the mad and fad metrics from gt noisy and pr dictionaries
Based on the paper, for eah element in noist gt we calculate the distance
between that and all prs samples and get the distance that is min
For mad we have also arr.shape[0]=12 in denominator but for fad we don't have it
'''


'''
add comment
'''
def diversity_metric(samples):
    final_norm = 0
    #print(samples)
    for i in samples:
      temp_min = np.inf
      for j in samples:
        if np.sum(i-j) == 0 :
          pass
        else:
          #print(i)
         # print(j)
          norm = np.linalg.norm(i-j)
          if norm < temp_min:
            temp_min = norm
           # print(norm)
      final_norm += temp_min
      print(temp_min)
    print('final_norm', final_norm/len(samples))
    return final_norm/len(samples) #you have to divide 

def our_diversity_metric(preds_dict):
  final_div = 0
  for key,values in preds_dict.items() :
      final_div += diversity_metric(values)
  return final_div/len(preds_dict), final_div

'''
add comment
'''

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
    parser.add_argument('--epoch',type=str,default="00015")
    parser.add_argument('--num_samples', type=int, default="20")

    args=parser.parse_args()
    model_name=args.name


    '''
    i don't think we need this
    try:
        os.mkdir('models')
    except:
        pass
    try:
        os.mkdir('output')
    except:
        pass
    try:
        os.mkdir('output/QuantizedTFsamples')
    except:
        pass
    try:
        os.mkdir(f'models/QuantizedTFsamples')
    except:
        pass

    try:
        os.mkdir(f'output/QuantizedTFsamples/{args.name}')
    except:
        pass

    try:
        os.mkdir(f'models/QuantizedTFsamples/{args.name}')
    except:
        pass
    '''
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
    num_samples = 20 #for now
    #print(args.layers)
    model=QuantizedTF(clusters.shape[0], clusters.shape[0]+1, clusters, clusters.shape[0],  N=args.layers,
                   d_model=args.emb_size, d_ff=1024, h=args.heads).to(device)

    model.load_state_dict(torch.load(f'models/QuantizedTFsamples/{args.name}/{args.epoch}.pth'))
    model.to(device)

    '''
    Now we have a tf that gives in output a batch of dim [1024, 10, 2]
    We will create many paths on it, all paths influenced by the selection of the previous step
    MASK will be a mess but we have hope

    '''

    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        model.eval()
        gt=[]
        pr=[]
        inp_=[]
        peds=[]
        frames=[]
        dt=[]
        ADE = 0
        FDE = 0
        final_diversity = 0
        #I need for later the num of test elements
        num_of_elements = 0
        for id_b, batch in enumerate(test_dl):
            print(f"batch {id_b:03d}/{len(test_dl)-1}")
            peds.append(batch['peds'])
            frames.append(batch['frames'])
            dt.append(batch['dataset'])
            scale = np.random.uniform(0.5, 2)
            # rot_mat = np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])
            n_in_batch = batch['src'].shape[0]
            speeds_inp = batch['src'][:, 1:, 2:4]
            gt_b = batch['trg'][:, :, 0:2]
            #print('THIS IS THE GROUND TRUTH')
            #print('shape', gt_b.shape)
            inp = torch.tensor(
                scipy.spatial.distance.cdist(speeds_inp.reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch,
                                                                                                         -1)).to(
                device)
            src_att = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
            start_of_seq = torch.tensor([clusters.shape[0]]).repeat(n_in_batch).unsqueeze(1).to(device)
            #print(start_of_seq.shape)
            dec_inp = start_of_seq
            dim_temp = 12
            storing_place = [torch.empty((n_in_batch, 1)).to(device) for i in range(dim_temp)]
            final_dict = {}
            #here for each i there will be stores the paths as lists of tensors(later)
            for i in range(n_in_batch):
               final_dict[str(i)]= []
            #here recursively we samples paths following a rule
            dict_paths = Path_creator(n_in_batch, inp, dec_inp, src_att, model, device, temperature = 0.2, dim_temp = 12)
            #print('path_rule', path_rule)
            #the path rule is a dict with list of tensors at each element
            #print('PATH RULE 33')
            #print('##################################################################################')
            #print(dict_paths['33'])
            #print('its length', len(dict_paths['33']))
            #print('##################################################################################')
            #Here is for transforming the dictionary

            my_dictionary = {k: convertfunc(v, clusters, batch['src'], int(k)) for k, v in dict_paths.items()}
          
            num_of_elements += len(my_dictionary)
            #print(num_of_elements)
            del dict_paths
            #print('###############################################################')
            #print('The final dict for 1 batch')
            #print(len(my_dictionary['33']))
            #print('#############')
            #print(my_dictionary['33'][0])
            #print('#############')
            #print(my_dictionary['33'])
            #print('###############################################################')
            
            '''
            to implement the distances we have to augment the groun truth
            for each element(e.g. '33') normally we would have 1 GT

            but for the metrics we need #samples GT for every element

            exemplum gratia GT_33 : [[0.2,0.1], [0.4 , 0.8]] (imagine only 2 steps)
            ===========> we want, if we sampled 3 paths for every sample
            GT_33_a ~ GT_33
            GT_33_b ~ noise + GT_33
            GT_33_c ~ //

            '''
            dict_of_noisy_GT = {}
            for i in range(n_in_batch):
               dict_of_noisy_GT[str(i)]= [gt_b[i].numpy()]
            How_many_to_noise = len(my_dictionary['0'])
            #print('1 GT', dict_of_noisy_GT['33'])

            for key, value in dict_of_noisy_GT.items():
              for i in range(How_many_to_noise - 1):
                noise = np.random.normal(0, 1, dict_of_noisy_GT[key][0].shape)
                new_signal = dict_of_noisy_GT[key][0] + noise
                dict_of_noisy_GT[key].append(new_signal)
            #print('after we add noise')
            #print('##########################################')
            #print('1 GT', dict_of_noisy_GT['33'])
            #print('##########################################')

            #now we have noise and we have to compute the metrics for every batch
            #add them and study them

            #we have a dict with preds (every key has n samples)
            #we have a dict with Gts (every k has 1GT and n-1 NoisyGT)

            
            temp_ADE, temp_FDE, ADE_unsc, FDE_unsc = get_metrics_ADEandFDE(dict_of_noisy_GT, my_dictionary)
            diversity_metric_, div_m= our_diversity_metric(my_dictionary)
            print('#################################################################')
            print('ADE for this batch{id_b:03d}/{len(test_dl)}', temp_ADE )
            print('FAD for this batch{id_b:03d}/{len(test_dl)}', temp_FDE )
            print('Diversity metric for batch{id_b:03d}/{len(test_dl)}', diversity_metric_)
            print('#################################################################')
            #del my_dictionary
            #del dict_of_noisy_GT
            #for the final average now we save the ade and fad and at end of test display the final value
            ADE += ADE_unsc
            FDE += FDE_unsc
            final_diversity += div_m 
            #print(ADE)
            #print(FDE)
    print('################################################################')
    print('The final ADE is in average', ADE/num_of_elements)
    print('The final FDE is in average', FDE/ num_of_elements)
    print('The final Diversity is in average', final_diversity/ num_of_elements)
    print('################################################################')
    print('Gently offered by Leonardo Placidi and Negin Amininodoushan =)')
    #you have a my_dictionary
    #you dict_of_noisy_GT

    print('Nowwwwww plotting')
    print('gtttttttttdict_of_noisy_GT[33][0]')
    
    plt.plot(*dict_of_noisy_GT['33'][0].T, linestyle='-', c = 'g')
    for i in range(len(my_dictionary['33'])) :
      plt.plot(*my_dictionary['33'][i].T, linestyle='-', c = 'r')

    plt.show()

    plt.plot(*dict_of_noisy_GT['100'][0].T, linestyle='-', c = 'g')
    for i in range(len(my_dictionary['100'])) :
      plt.plot(*my_dictionary['100'][i].T, linestyle='-', c = 'r')

    plt.show()

    plt.plot(*dict_of_noisy_GT['200'][0].T, linestyle='-', c = 'g')
    for i in range(len(my_dictionary['200'])) :
      plt.plot(*my_dictionary['200'][i].T, linestyle='-', c = 'r')

    plt.show()



            
if __name__=='__main__':
    main()
