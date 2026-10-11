"""Paper-architecture RAVEN for daily ETF factors (requires PyTorch).

This is an independent implementation of He et al. (2026), not the authors'
released source. It follows instance normalization, channel-independent patch
projection, reverse chronological CIT, three 3-layer Transformer experts,
pooled CAW, a global self-attention branch, and MSE + entropy + diversity.

Paper specifies patch length 16, d=128, thresholds .3/.6/.9, 60 AdamW epochs.
Our maximum context is 64 daily observations; this and ETF inputs are task
adaptations, and trainable raw gate confidence is underspecified in the text.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    torch=None
    nn=None
    F=None

from index_timing_framework import StudyConfig, load_market
from multifrequency_models import (HORIZONS, EXECUTIONS, TEST_START, factor_matrix,
    screen_factors, target, windows_for)
from raven_lite_timing import backtest, measure


if nn is not None:
    class RAVEN(nn.Module):
        def __init__(self,channels:int,lookback=64,patch=16,dim=128,
                     heads=4,layers=3,thresholds=(.3,.6,.9)):
            super().__init__()
            assert lookback%patch==0 and channels>0
            self.lookback=lookback;self.patch=patch
            self.thresholds=thresholds
            self.patch_embed=nn.Linear(patch,dim)
            n=lookback//patch
            position=torch.arange(n).float().unsqueeze(1)
            divisor=torch.exp(torch.arange(0,dim,2).float()*(-math.log(10000.)/dim))
            pe=torch.zeros(n,dim);pe[:,0::2]=torch.sin(position*divisor)
            pe[:,1::2]=torch.cos(position*divisor)
            self.register_buffer('position',pe,persistent=False)
            self.importance=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,1))
            self.experts=nn.ModuleList([
                nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim,nhead=heads,
                    dim_feedforward=2*dim,batch_first=True,activation='gelu',
                    norm_first=True,dropout=.1),num_layers=layers,
                    enable_nested_tensor=False) for _ in thresholds])
            self.global_attention=nn.MultiheadAttention(dim,heads,batch_first=True)
            self.gate=nn.Linear(dim,1)
            self.redundancy_unconstrained=nn.Parameter(torch.tensor(0.))
            self.head=nn.Sequential(nn.LayerNorm(2*dim),nn.Linear(2*dim,dim),
                                    nn.GELU(),nn.Linear(dim,1))

        def forward(self,x):
            # x: [batch, 64 daily observations, channels], current day last.
            if x.ndim!=3 or x.shape[1]!=self.lookback:
                raise ValueError('Input must have shape [B, lookback, channels]')
            x=(x-x.mean(dim=1,keepdim=True))/(x.std(dim=1,keepdim=True,
                 unbiased=False)+1e-6)
            b,l,d=x.shape;n=l//self.patch
            patches=x.transpose(1,2).reshape(b,d,n,self.patch).flip(2)
            embedding=self.patch_embed(patches).mean(dim=1)+self.position[None,:,:]
            probs=self.importance(embedding).squeeze(-1).softmax(dim=-1)
            cumulative=probs.cumsum(dim=1)
            prefix=[];zs=[];masses=[];indices=torch.arange(n,device=x.device)[None,:]
            for tau,expert in zip(self.thresholds,self.experts):
                # Include the first patch that crosses tau, ensuring a
                # nonempty contiguous prefix in every sample.
                count=(cumulative<tau).sum(dim=1).add(1).clamp(max=n)
                mask=indices>=count[:,None]
                hidden=expert(embedding,src_key_padding_mask=mask)
                z=(hidden*(~mask).unsqueeze(-1)).sum(dim=1)/count[:,None]
                zs.append(z);prefix.append(count)
                masses.append((probs*(~mask)).sum(dim=1))
            z=torch.stack(zs,dim=1)
            norm=F.normalize(z,dim=-1)
            cosine=norm@norm.transpose(1,2)
            eye=torch.eye(len(self.experts),device=x.device)[None]
            redundancy=(cosine.relu()*(1-eye)).sum(dim=-1)
            # Paper does not define alpha_k's parameterization. We implement
            # its learned raw routing confidence via pooled expert states.
            # The selected probability mass gives the scorer a continuous
            # gradient through the gate despite discrete CIT prefix lengths.
            mass=torch.stack(masses,dim=1)
            alpha=(self.gate(z).squeeze(-1)+mass.clamp_min(1e-8).log()).softmax(dim=-1)
            penalty=F.softplus(self.redundancy_unconstrained)
            weights=alpha*torch.exp(-penalty*redundancy)
            weights=weights/(weights.sum(dim=-1,keepdim=True)+1e-8)
            local=(weights[:,:,None]*z).sum(dim=1)
            global_hidden,_=self.global_attention(embedding,embedding,embedding,
                                                   need_weights=False)
            global_z=global_hidden.mean(dim=1)
            prediction=self.head(torch.cat([local,global_z],dim=-1)).squeeze(-1)
            entropy=(probs*probs.clamp_min(1e-9).log()).sum(dim=1).mean()
            diversity=((cosine-eye)**2).sum(dim=(1,2)).mean()
            return prediction,entropy,diversity,torch.stack(prefix,dim=1)
else:
    RAVEN=None


def fit_predict(train_x,train_y,pred_x,epochs=60,seed=7):
    if torch is None:raise RuntimeError('PyTorch is required for the full RAVEN')
    torch.manual_seed(seed);torch.set_num_threads(1)
    scale=max(float(np.std(train_y)),1e-6)
    mean=float(np.mean(train_y))
    yt=(train_y-mean)/scale
    model=RAVEN(train_x.shape[-1])
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-2)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    tx=torch.tensor(train_x,dtype=torch.float32)
    ty=torch.tensor(yt,dtype=torch.float32)
    generator=torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(tx),generator=generator).split(32):
            opt.zero_grad(set_to_none=True)
            yhat,entropy,diversity,_=model(tx[idx])
            loss=F.mse_loss(yhat,ty[idx])+.1*entropy+.01*diversity
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
        schedule.step()
    model.eval()
    with torch.no_grad():
        pred,_,_,prefix=model(torch.tensor(pred_x,dtype=torch.float32))
    return pred.numpy()*scale+mean,prefix.numpy(),sum(p.numel() for p in model.parameters())


def experiment(path,symbol,h,execution,epochs=60,retrain_every=20):
    if torch is None:
        return {'symbol':symbol,'horizon':h,'execution':execution,
                'status':'dependency_unavailable','detail':'PyTorch not installed; no model was fit'}
    cfg=StudyConfig();panel=load_market(path,cfg)
    matrix,valid_days=factor_matrix(path,panel,cfg)
    y,maturity=target(panel,h,execution)
    split=int(np.searchsorted(panel.dates,TEST_START))
    selected,diag=screen_factors(matrix,y,maturity,split)
    if not selected:return {'symbol':symbol,'horizon':h,'execution':execution,
                            'status':'no_stable_features','screen':diag}
    days=np.array([i for i in range(len(y)) if panel.dates[i] in valid_days],int)
    windows,pos=windows_for(matrix,selected,days,length=64)
    test_idx=np.where((pos>=split)&np.isfinite(y[pos]))[0]
    predictions=[];observed=[];test_pos=[];counts=[];prefixes=[];params=None
    fitted_end=-1
    for ix in test_idx:
        t=int(pos[ix])
        if fitted_end<0 or t-fitted_end>=retrain_every:
            train=np.where((pos<t)&(maturity[pos]<=t)&np.isfinite(y[pos]))[0]
            if len(train)<40:continue
            # Every training label has exited by the refit signal close.
            fitted_end=t
            prediction_block,prefix_block,params=fit_predict(windows[train],y[pos[train]],
                windows[test_idx[test_idx>=ix]],epochs=epochs,seed=7+t)
            cache={int(pos[z]):(float(v),p.tolist()) for z,v,p in
                   zip(test_idx[test_idx>=ix],prediction_block,prefix_block)}
            fit_count=len(train)
        if t not in cache:continue
        v,p=cache[t]
        predictions.append(v);observed.append(y[t]);test_pos.append(t)
        counts.append(fit_count);prefixes.append(p)
    if not predictions:return {'symbol':symbol,'horizon':h,'execution':execution,
        'status':'too_few_training_labels','selected':selected}
    pred=np.asarray(predictions);truth=np.asarray(observed);test_pos=np.asarray(test_pos)
    return {'symbol':symbol,'horizon':h,'execution':execution,'status':'ok',
        'selected':selected,'screen':diag,'n_test':len(pred),
        'n_train_min':min(counts),'n_train_max':max(counts),'parameters':params,
        'prefix_means':np.mean(prefixes,axis=0).tolist(),
        'metric':measure(truth,pred),'daily_long_cash':backtest(panel,test_pos,pred,execution),
        'predictions':[{'signal_day':panel.dates[t],'label_maturity':panel.dates[maturity[t]],
                        'forecast':float(v),'realized':float(q),'train_n':int(n)}
                       for t,v,q,n in zip(test_pos,pred,truth,counts)]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--spy',required=True);p.add_argument('--qqq',required=True)
    p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--retrain-every',type=int,default=20)
    a=p.parse_args();out=[]
    for sym,path in [('SPY',a.spy),('QQQ',a.qqq)]:
        for h in HORIZONS:
            for ex in EXECUTIONS:
                r=experiment(path,sym,h,ex,a.epochs,a.retrain_every)
                out.append(r)
                print(sym,h,ex,r['status'],r.get('n_test'),flush=True)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(out,indent=2,allow_nan=False))

if __name__=='__main__':main()
