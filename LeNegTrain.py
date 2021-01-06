# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 16:57:20 2020

@author: leona
"""
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
from LossLeNeg import LeNeg_loss
from quantized_TFsamples import QuantizedTF   #here!!!!!!!!!!!
from torch.utils.tensorboard import SummaryWriter

'''
Here we train this new TF, called quantized_TFsamples

as stated in the import we will use the name QuantizedTF but is the new TF in use, see above
We point out here that we did some modification and the training is fully working for any batch size.
Also the Quantized TF has some more argument that prof.Galasso's one.
'''

def main():
    
    parser=argparse.ArgumentParser(description='Train the individual Transformer model')
    
    parser.add_argument('--dataset_folder',type=str,default='datasets')
    
    parser.add_argument('--dataset_name',type=str,default='zara1')
    
    parser.add_argument('--obs',type=int,default=8)
    parser.add_argument('--preds',type=int,default=12)
    parser.add_argument('--emb_size',type=int,default=512)
    
    parser.add_argument('--heads',type=int, default=8)
    parser.add_argument('--layers',type=int,default=6)
    parser.add_argument('--dropout',type=float,default=0.1)
    
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--output_folder',type=str,default='Output')
    parser.add_argument('--val_size',type=int, default=0)
    
    parser.add_argument('--gpu_device',type=str, default="0")
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--max_epoch',type=int, default=20)
    
    parser.add_argument('--batch_size',type=int,default=100)
    parser.add_argument('--validation_epoch_start', type=int, default=30)
    parser.add_argument('--resume_train',action='store_true')
    
    parser.add_argument('--delim',type=str,default='\t')
    parser.add_argument('--name', type=str, default="zara1")
    parser.add_argument('--factor', type=float, default=1.)
    
    parser.add_argument('--evaluate',type=bool,default=True)
    parser.add_argument('--save_step', type=int, default=1)

    

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

    log=SummaryWriter('logs/%s'%model_name)

    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    device=torch.device("cuda")

    if args.cpu or not torch.cuda.is_available():
        device=torch.device("cpu")

    args.verbose=True

        
    ## creation of the dataloaders for train and validation
    if args.val_size==0:
        train_dataset,_ = baselineUtils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=True,verbose=args.verbose)
        val_dataset, _ = baselineUtils.create_dataset(args.dataset_folder, args.dataset_name, 0, args.obs,
                                                                    args.preds, delim=args.delim, train=False,
                                                                    verbose=args.verbose)
    else:
        train_dataset, val_dataset = baselineUtils.create_dataset(args.dataset_folder, args.dataset_name, args.val_size, args.obs,
                                                              args.preds, delim=args.delim, train=True,
                                                              verbose=args.verbose)
    
    test_dataset,_ =  baselineUtils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=False,eval=True,verbose=args.verbose)

    mat = scipy.io.loadmat(os.path.join(args.dataset_folder, args.dataset_name, "clusters.mat"))
    clusters=mat['centroids']
    num_samples = 20
    model=QuantizedTF(clusters.shape[0], clusters.shape[0]+1, clusters, clusters.shape[0],  N=args.layers,
                   d_model=args.emb_size, d_ff=1024, h=args.heads, dropout=args.dropout, num_samples = num_samples).to(device)
    '''
    Here we put the parameters from the trained standard TF, so we can train again
    '''
    model.load_state_dict(torch.load(f'models/QuantizedTF/zara2/00019.pth'))
    model.to(device)
    tr_dl=torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    #optim = SGD(list(a.parameters())+list(model.parameters())+list(generator.parameters()),lr=0.01)
    #sched=torch.optim.lr_scheduler.StepLR(optim,0.0005)
    optim = NoamOpt(args.emb_size, args.factor, len(tr_dl)*5,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    #optim=Adagrad(list(a.parameters())+list(model.parameters())+list(generator.parameters()),lr=0.01,lr_decay=0.001)
    epoch=0


    loss_epochs_train = {}
    loss_epochs_Val = {}
    while epoch<args.max_epoch:
        epoch_loss=0
        model.train()

        for id_b,batch in enumerate(tr_dl):

            optim.optimizer.zero_grad()
            scale=np.random.uniform(0.5,4)
            #rot_mat = np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])
            n_in_batch=batch['src'].shape[0]
            speeds_inp=batch['src'][:,1:,2:4]*scale
            inp=torch.tensor(scipy.spatial.distance.cdist(speeds_inp.reshape(-1,2),clusters).argmin(axis=1).reshape(n_in_batch,-1)).to(device)
            speeds_trg = batch['trg'][:,:,2:4]*scale
            target = torch.tensor(
                scipy.spatial.distance.cdist(speeds_trg.reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch, -1)).to(
                device)
            src_att = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
            trg_att=subsequent_mask(target.shape[1]).repeat(n_in_batch,1,1).to(device)
            start_of_seq=torch.tensor([clusters.shape[0]]).repeat(n_in_batch).unsqueeze(1).to(device)
            dec_inp=torch.cat((start_of_seq,target[:,:-1]),1)

            print('Hi I am training')

            out = model(inp, dec_inp, src_att, trg_att) #those are the selected points (2D)

            vector_of_zs = model.zs #those are probs of the selected points!
            
            R = torch.quantile(vector_of_zs, 0.9)
            '''here we call our artigianal loss'''
            loss_class = LeNeg_loss(R)
            print('Hi I am entering the loss for the train')
            loss = loss_class(model.zs, out, num_samples)
            #compute loss using zs and xs
            loss.backward()
            optim.step()
            print("epoch %03i/%03i  frame %04i / %04i loss: %7.4f" % (epoch, args.max_epoch, id_b, len(tr_dl), loss.item()))
            epoch_loss += loss.item()
        #sched.step()
        loss_epochs_train[str(epoch)] =  epoch_loss / len(tr_dl)
        log.add_scalar('Loss/train', epoch_loss / len(tr_dl), epoch)
        
        epoch = epoch+1

        if epoch % args.save_step == 0:
                torch.save(model.state_dict(), f'models/QuantizedTFsamples/{args.name}/{epoch:05d}.pth')



        #Here the Validation check.


        with torch.no_grad():
            model.eval()

            gt=[]
            pr=[]
            val_loss=0
            step=0
            j = 0
            for batch in val_dl:
                # rot_mat = np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])
                n_in_batch = batch['src'].shape[0]
                speeds_inp = batch['src'][:, 1:, 2:4]
                inp = torch.tensor(
                    scipy.spatial.distance.cdist(speeds_inp.contiguous().reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch,
                                                                                                             -1)).to(
                    device)
                speeds_trg = batch['trg'][:, :, 2:4]
                target = torch.tensor(
                    scipy.spatial.distance.cdist(speeds_trg.contiguous().reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch,
                                                                                                             -1)).to(
                    device)
                src_att = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
                trg_att = subsequent_mask(target.shape[1]).repeat(n_in_batch, 1, 1).to(device)
                start_of_seq = torch.tensor([clusters.shape[0]]).repeat(n_in_batch).unsqueeze(1).to(device)
                dec_inp = torch.cat((start_of_seq, target[:, :-1]), 1)
                out = model(inp, dec_inp, src_att, trg_att) #those are the selected points (2D)

                vector_of_zs2 = model.zs #those are probs of the selected points!
                j += 1
                R = torch.quantile(vector_of_zs2, 0.9)
                loss_class = LeNeg_loss(R)
                print('Hi I am entering the Val_loss')
                loss = loss_class(model.zs, out, num_samples)
                print("val epoch %03i/%03i  frame %04i / %04i loss: %7.4f" % (
                epoch, args.max_epoch, step, len(val_dl), loss.item()))
                val_loss+=loss.item()
                step+=1

            loss_epochs_Val[str(epoch)] = val_loss  / j
            log.add_scalar('validation/loss', val_loss / len(val_dl), epoch)

    with open('Loss_train_LeNegbig_zara2.json', 'w') as fp:
        json.dump(loss_epochs_train, fp)
    with open('Loss__Val_LeNegbig_zara2.json', 'w') as fp:
        json.dump(loss_epochs_Val, fp)
    print('epoch vs average loss in train:')
    print(loss_epochs_train)
    print('epoch vs average loss in Val:')
    print(loss_epochs_Val)

if __name__=='__main__':
    main()   
