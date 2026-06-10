#!/usr/bin/env python3
"""Structure-only VICReg for TCR:pMHC binding from precomputed Boltz interface shards.

This is the structural analogue of the plain TCR-vs-pMHC VICReg baseline.
It uses only the compact structural shards built by build_struct_shards.py.

Views:
  z_T_struct    = encoder(type 0 tcr_pep + type 2 tcr_hla)
  z_pMHC_struct = encoder(type 1 pep_tcr + type 3 hla_tcr)

Training uses positive examples only. Validation/test/IMMREP may contain positives
and negatives. Inference score is -MSE(z_T_struct, z_pMHC_struct).
"""
from __future__ import annotations

import argparse, copy, json, math, random
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, auc
from torch.utils.data import Dataset, DataLoader


def set_seed(seed:int)->None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False


def first_existing_col(df:pd.DataFrame, cols:List[str])->Optional[str]:
    return next((c for c in cols if c in df.columns), None)


def normalise_manifest(df:pd.DataFrame, source:str)->pd.DataFrame:
    if "pair_id" not in df.columns: raise ValueError(f"{source} CSV must contain pair_id")
    out=df.copy(); out["pair_id"]=out["pair_id"].astype(str)
    lab=first_existing_col(out,["binding_flag","label","binder","target"])
    out["binding_flag"] = 1 if lab is None else out[lab].astype(int)
    pep=first_existing_col(out,["Peptide","peptide","pep_seq","peptide_seq"])
    out["peptide_for_eval"] = out[pep].astype(str) if pep else out["pair_id"].astype(str)
    length_specs={
        "tcra_len":["tcra_len","tcr_alpha_len","cdr3a_len","alpha_len"],
        "tcrb_len":["tcrb_len","tcr_beta_len","cdr3b_len","beta_len"],
        "pep_len":["pep_len","peptide_len"],
        "hla_len":["hla_len","mhc_len","mhca_len"],
    }
    seq_specs={
        "tcra_len":["tcra","tcr_alpha","TRA","cdr3a","alpha","alpha_seq","tcr_a"],
        "tcrb_len":["tcrb","tcr_beta","TRB","cdr3b","beta","beta_seq","tcr_b"],
        "pep_len":["Peptide","peptide","pep_seq","peptide_seq"],
        "hla_len":["hla","HLA","hla_seq","mhc","mhc_seq"],
    }
    for target, cands in length_specs.items():
        c=first_existing_col(out,cands)
        if c: out[target]=pd.to_numeric(out[c],errors="coerce").fillna(0).astype(int)
        else:
            sc=first_existing_col(out,seq_specs[target])
            out[target]=out[sc].fillna("").astype(str).str.len().astype(int) if sc else 0
    return out


def complete_meta(csv_path:str, positives_only:bool, source:str)->pd.DataFrame:
    raw=pd.read_csv(csv_path); meta=normalise_manifest(raw,source); before=len(meta)
    if positives_only: meta=meta[meta.binding_flag.astype(int)==1].copy()
    after_label=len(meta)
    complete=(meta.tcra_len>0)&(meta.tcrb_len>0)&(meta.pep_len>0)&(meta.hla_len>0)
    meta=meta[complete].copy()
    print(f"{source}: rows={before} | after_label_filter={after_label} | complete_alpha_beta_pmhc={len(meta)}", flush=True)
    return meta.reset_index(drop=True)


class StructShardStore:
    def __init__(self, root:Path, source:str, cache_size:int=8):
        self.root=Path(root); self.source=source
        index_path=self.root/"struct_shard_index.json"
        if not index_path.exists(): raise FileNotFoundError(f"{source}: missing {index_path}")
        payload=json.loads(index_path.read_text())
        self.index=payload["index"]; self.cap_per_block=int(payload.get("cap_per_block",128)); self.n_max=int(payload.get("n_max",512)); self.dz=int(payload.get("dz",128))
        self.cache_size=max(1,int(cache_size)); self.cache:OrderedDict[str,Any]=OrderedDict()
        print(f"{source}: struct index={self.root} | examples={len(self.index)} | cap={self.cap_per_block} | n_max={self.n_max} | dz={self.dz}", flush=True)
    def pair_ids(self)->set[str]: return set(self.index.keys())
    def _load(self, shard_name:str):
        if shard_name in self.cache:
            self.cache.move_to_end(shard_name); return self.cache[shard_name]
        obj=torch.load(self.root/shard_name,map_location="cpu")
        self.cache[shard_name]=obj; self.cache.move_to_end(shard_name)
        while len(self.cache)>self.cache_size: self.cache.popitem(last=False)
        return obj
    def get(self,pid:str)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        rec=self.index[str(pid)]; shard=self._load(rec["shard"]); row=int(rec["row"])
        return shard["struct_tokens"][row].float(), shard["struct_type_ids"][row].long(), shard["struct_mask"][row].bool()


def subsample_per_type(tokens:torch.Tensor, type_ids:torch.Tensor, mask:torch.Tensor, cap:int)->Tuple[torch.Tensor,torch.Tensor]:
    tokens=tokens[mask.bool()]; type_ids=type_ids[mask.bool()]
    outs=[]; tys=[]
    for tid in torch.unique(type_ids):
        sel=(type_ids==tid).nonzero(as_tuple=True)[0]
        if cap>0 and sel.numel()>cap:
            idx=torch.linspace(0,sel.numel()-1,steps=cap).round().long(); sel=sel[idx]
        outs.append(tokens[sel]); tys.append(type_ids[sel])
    if not outs: return tokens.new_zeros((0,tokens.shape[-1])), type_ids.new_zeros((0,))
    return torch.cat(outs,0), torch.cat(tys,0)


class StructOnlyDataset(Dataset):
    def __init__(self, csv_path:str, struct_root:str, positives_only:bool, cfg:"RunConfig", source:str):
        self.cfg=cfg; self.source=source; self.meta=complete_meta(csv_path, positives_only, source)
        self.store=StructShardStore(Path(struct_root), source, cfg.struct_shard_cache_size)
        self.meta=self.meta[self.meta.pair_id.astype(str).isin(self.store.pair_ids())].reset_index(drop=True)
        print(f"{source}: kept_complete_with_struct={len(self.meta)}", flush=True)
        if len(self.meta)==0: raise RuntimeError(f"{source}: no rows after struct matching")
    def __len__(self): return len(self.meta)
    def __getitem__(self, idx:int)->Dict[str,Any]:
        row=self.meta.iloc[idx]; pid=str(row.pair_id)
        tokens, type_ids, mask=self.store.get(pid)
        tokens, type_ids=subsample_per_type(tokens,type_ids,mask,self.cfg.tokens_per_interface_at_load)
        tcr_mask=(type_ids==0)|(type_ids==2); pmhc_mask=(type_ids==1)|(type_ids==3)
        return {
            "tcr_tokens":tokens[tcr_mask].float(), "tcr_type_ids":type_ids[tcr_mask].long(),
            "pmhc_tokens":tokens[pmhc_mask].float(), "pmhc_type_ids":type_ids[pmhc_mask].long(),
            "binding_flag":int(row.binding_flag), "pair_id":pid, "peptide":str(row.peptide_for_eval),
        }


def _pad(items:List[torch.Tensor], pad_value:float=0.0)->Tuple[torch.Tensor,torch.Tensor]:
    B=len(items); max_n=max(x.shape[0] for x in items); d=items[0].shape[-1]
    out=torch.full((B,max_n,d), pad_value, dtype=torch.float32); mask=torch.zeros((B,max_n), dtype=torch.bool)
    for i,x in enumerate(items): out[i,:x.shape[0]]=x; mask[i,:x.shape[0]]=True
    return out,mask

def _pad_ids(items:List[torch.Tensor], max_n:int)->torch.Tensor:
    out=torch.zeros((len(items),max_n), dtype=torch.long)
    for i,x in enumerate(items): out[i,:x.shape[0]]=x
    return out


def collate(rows:List[Dict[str,Any]])->Dict[str,Any]:
    t_tok,t_mask=_pad([r["tcr_tokens"] for r in rows]); p_tok,p_mask=_pad([r["pmhc_tokens"] for r in rows])
    return {
        "tcr_tokens":t_tok, "tcr_mask":t_mask, "tcr_type_ids":_pad_ids([r["tcr_type_ids"] for r in rows], t_tok.shape[1]),
        "pmhc_tokens":p_tok, "pmhc_mask":p_mask, "pmhc_type_ids":_pad_ids([r["pmhc_type_ids"] for r in rows], p_tok.shape[1]),
        "binding_flag":torch.tensor([r["binding_flag"] for r in rows], dtype=torch.long),
        "pair_id":[r["pair_id"] for r in rows], "peptide":[r["peptide"] for r in rows],
    }


class AttentionPooler(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, out_dim:int, n_heads:int, dropout:float, n_token_types:int, n_queries:int=4):
        super().__init__(); assert hidden_dim % n_heads == 0
        self.input_proj=nn.Linear(input_dim, hidden_dim); self.type_embedding=nn.Embedding(n_token_types, hidden_dim)
        self.query=nn.Parameter(torch.zeros(1, n_queries, hidden_dim)); nn.init.normal_(self.query,std=0.02)
        self.attn=nn.MultiheadAttention(hidden_dim,n_heads,dropout=dropout,batch_first=True)
        self.mlp=nn.Sequential(nn.LayerNorm(hidden_dim*n_queries), nn.Linear(hidden_dim*n_queries, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim,out_dim))
    def forward(self,tokens,mask,type_ids):
        B=tokens.shape[0]; x=self.input_proj(tokens)+self.type_embedding(type_ids.clamp_min(0)); q=self.query.expand(B,-1,-1)
        key_padding=~mask
        all_pad=key_padding.all(1)
        if all_pad.any(): key_padding=key_padding.clone(); key_padding[all_pad,0]=False
        pooled,_=self.attn(q,x,x,key_padding_mask=key_padding,need_weights=False)
        return self.mlp(pooled.reshape(B,-1))


class StructOnlyVICReg(nn.Module):
    def __init__(self,cfg:"RunConfig"):
        super().__init__()
        self.tcr=AttentionPooler(128,cfg.hidden_dim,cfg.latent_dim,cfg.heads,cfg.dropout,cfg.max_pair_types,cfg.pool_queries)
        self.pmhc=AttentionPooler(128,cfg.hidden_dim,cfg.latent_dim,cfg.heads,cfg.dropout,cfg.max_pair_types,cfg.pool_queries)
    def forward(self,batch):
        zT=self.tcr(batch["tcr_tokens"],batch["tcr_mask"],batch["tcr_type_ids"])
        zP=self.pmhc(batch["pmhc_tokens"],batch["pmhc_mask"],batch["pmhc_type_ids"])
        return zT,zP


def vicreg_variance(u,gamma,eps):
    u=u-u.mean(0,keepdim=True); std=torch.sqrt(u.var(0,unbiased=False)+eps); return F.relu(gamma-std).mean()
def vicreg_covariance(u):
    B,d=u.shape
    if B<=1: return torch.tensor(0.,device=u.device,dtype=u.dtype)
    u=u-u.mean(0,keepdim=True); cov=(u.T@u)/(B-1); off=cov-torch.diag_embed(torch.diag(cov)); return (off**2).sum()/d
def vicreg_loss(zT,zP,cfg,return_parts=False):
    inv=F.mse_loss(zT,zP); var=vicreg_variance(zT,cfg.gamma_var,cfg.eps_var)+vicreg_variance(zP,cfg.gamma_var,cfg.eps_var); cov=vicreg_covariance(zT)+vicreg_covariance(zP)
    loss=cfg.alpha*inv+cfg.beta*var+cfg.delta*cov
    if not return_parts: return loss
    return loss,{"loss":float(loss.detach().cpu()),"L_inv_mse":float(inv.detach().cpu()),"L_var":float(var.detach().cpu()),"L_cov":float(cov.detach().cpu()),"weighted_inv":float((cfg.alpha*inv).detach().cpu()),"weighted_var":float((cfg.beta*var).detach().cpu()),"weighted_cov":float((cfg.delta*cov).detach().cpu()),"zT_std":float(zT.std(unbiased=False).detach().cpu()),"zP_std":float(zP.std(unbiased=False).detach().cpu())}

def score_mse(zT,zP):
    d=(zT-zP).pow(2).mean(-1); return d,-d

def safe_auroc(y,s): return float("nan") if len(np.unique(y))<2 else float(roc_auc_score(y,s))
def safe_auprc(y,s): return float("nan") if len(np.unique(y))<2 else float(average_precision_score(y,s))
def partial_auc_raw(y,s,max_fpr=0.1):
    if len(np.unique(y))<2: return float("nan")
    fpr,tpr,_=roc_curve(y,s); stop=np.searchsorted(fpr,max_fpr,side="right")
    f=np.concatenate([fpr[:stop],[max_fpr]]); t=np.concatenate([tpr[:stop],[np.interp(max_fpr,fpr,tpr)]])
    return float(auc(f,t))




def partial_auc_mcclish(y,s,max_fpr=0.1):
    """McClish-standardised partial AUROC for IMMREP Macro AUC0.1.

    sklearn roc_auc_score(..., max_fpr=max_fpr) returns the standardised partial AUC
    where random performance is ~0.5 and perfect performance is 1.0.
    """
    if len(np.unique(y))<2: return float("nan")
    return float(roc_auc_score(y,s,max_fpr=max_fpr))


def per_peptide(y,s,peps,max_fpr=0.1):
    rows=[]; df=pd.DataFrame({"label":y,"score":s,"peptide":peps})
    for pep,g in df.groupby("peptide",sort=True):
        yy=g.label.to_numpy(int); ss=g.score.to_numpy(float); valid=len(np.unique(yy))==2
        pr=partial_auc_raw(yy,ss,max_fpr) if valid else float("nan")
        pm=partial_auc_mcclish(yy,ss,max_fpr) if valid else float("nan")
        rows.append({"peptide":pep,"n":len(g),"n_pos":int(yy.sum()),"n_neg":int((yy==0).sum()),"auroc":safe_auroc(yy,ss) if valid else float("nan"),f"auc{max_fpr:g}_raw":pr,f"auc{max_fpr:g}_raw_div_maxfpr":pr/max_fpr if valid else float("nan"),f"auc{max_fpr:g}_norm":pm if valid else float("nan"),f"auc{max_fpr:g}_mcclish":pm if valid else float("nan"),"valid":valid})
    tab=pd.DataFrame(rows); vt=tab[tab.valid].copy() if len(tab) else tab
    if len(vt)==0:
        summ={"macro_per_peptide_auroc":float("nan"),"weighted_per_peptide_auroc":float("nan"),f"macro_per_peptide_auc{max_fpr:g}_raw":float("nan"),f"weighted_per_peptide_auc{max_fpr:g}_raw":float("nan"),f"macro_per_peptide_auc{max_fpr:g}_raw_div_maxfpr":float("nan"),f"weighted_per_peptide_auc{max_fpr:g}_raw_div_maxfpr":float("nan"),f"macro_per_peptide_auc{max_fpr:g}_norm":float("nan"),f"weighted_per_peptide_auc{max_fpr:g}_norm":float("nan"),f"macro_per_peptide_auc{max_fpr:g}_mcclish":float("nan"),f"weighted_per_peptide_auc{max_fpr:g}_mcclish":float("nan"),"n_valid_peptides":0,"n_peptides_total":len(tab)}
    else:
        summ={"macro_per_peptide_auroc":float(vt.auroc.mean()),"weighted_per_peptide_auroc":float(np.average(vt.auroc,weights=vt.n)),f"macro_per_peptide_auc{max_fpr:g}_raw":float(vt[f"auc{max_fpr:g}_raw"].mean()),f"weighted_per_peptide_auc{max_fpr:g}_raw":float(np.average(vt[f"auc{max_fpr:g}_raw"],weights=vt.n)),f"macro_per_peptide_auc{max_fpr:g}_raw_div_maxfpr":float(vt[f"auc{max_fpr:g}_raw_div_maxfpr"].mean()),f"weighted_per_peptide_auc{max_fpr:g}_raw_div_maxfpr":float(np.average(vt[f"auc{max_fpr:g}_raw_div_maxfpr"],weights=vt.n)),f"macro_per_peptide_auc{max_fpr:g}_norm":float(vt[f"auc{max_fpr:g}_mcclish"].mean()),f"weighted_per_peptide_auc{max_fpr:g}_norm":float(np.average(vt[f"auc{max_fpr:g}_mcclish"],weights=vt.n)),f"macro_per_peptide_auc{max_fpr:g}_mcclish":float(vt[f"auc{max_fpr:g}_mcclish"].mean()),f"weighted_per_peptide_auc{max_fpr:g}_mcclish":float(np.average(vt[f"auc{max_fpr:g}_mcclish"],weights=vt.n)),"n_valid_peptides":int(len(vt)),"n_peptides_total":int(len(tab))}
    return tab.sort_values(["valid","n"],ascending=[False,False]), summ

def move(batch,device): return {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}

@torch.no_grad()
def evaluate(model,loader,device,cfg,split):
    model.eval(); rows=[]; run={}; steps=0
    for batch in loader:
        batch=move(batch,device); zT,zP=model(batch); _,parts=vicreg_loss(zT,zP,cfg,True); dist,score=score_mse(zT,zP); labels=batch["binding_flag"].cpu().numpy().astype(int)
        for i,pid in enumerate(batch["pair_id"]): rows.append({"split":split,"pair_id":pid,"peptide":batch["peptide"][i],"label":int(labels[i]),"mse_distance":float(dist[i].cpu()),"model_score":float(score[i].cpu())})
        for k,v in parts.items(): run[k]=run.get(k,0.)+float(v)
        steps+=1
    pred=pd.DataFrame(rows); y=pred.label.to_numpy(int); s=pred.model_score.to_numpy(float); peps=pred.peptide.to_numpy(str)
    tab,pep=per_peptide(y,s,peps,cfg.partial_auc_max_fpr)
    pos=pred[pred.label==1].mse_distance; neg=pred[pred.label==0].mse_distance
    metrics={"split":split,"n":len(pred),"n_pos":int(y.sum()),"n_neg":int((y==0).sum()),"auroc":safe_auroc(y,s),"auprc":safe_auprc(y,s),f"auc{cfg.partial_auc_max_fpr:g}_raw":partial_auc_raw(y,s,cfg.partial_auc_max_fpr),f"auc{cfg.partial_auc_max_fpr:g}_norm":partial_auc_mcclish(y,s,cfg.partial_auc_max_fpr),f"auc{cfg.partial_auc_max_fpr:g}_mcclish":partial_auc_mcclish(y,s,cfg.partial_auc_max_fpr),**pep,"pos_mse_mean":float(pos.mean()) if len(pos) else float("nan"),"neg_mse_mean":float(neg.mean()) if len(neg) else float("nan"),"mse_gap_neg_minus_pos":float(neg.mean()-pos.mean()) if len(pos) and len(neg) else float("nan"),**{f"mean_{k}":v/max(steps,1) for k,v in run.items()}}
    return {"predictions":pred,"per_peptide":tab,"metrics":metrics}


def get_selection(metrics,cfg):
    for k in [cfg.selection_metric,"weighted_per_peptide_auroc","auroc"]:
        v=metrics.get(k)
        if isinstance(v,(float,int)) and not math.isnan(float(v)): return float(v)
    return float("nan")

@dataclass
class RunConfig:
    train_csv:str="/home/natasha/multimodal_model/data/train/train_multiview.csv"
    val_csv:str="/home/natasha/multimodal_model/data/val/val_multiview.csv"
    test_csv:str="/home/natasha/multimodal_model/data/test/test_multiview.csv"
    immrep_csv:str="/home/natasha/multimodal_model/data/immrep_test/immrep_test_multiview.csv"
    struct_train_root:str="/home/natasha/multimodal_model/outputs_data/train_struct_shards"
    struct_val_root:str="/home/natasha/multimodal_model/outputs_data/val_struct_shards"
    struct_test_root:str="/home/natasha/multimodal_model/outputs_data/test_struct_shards"
    struct_immrep_root:str="/home/natasha/multimodal_model/outputs_data/immrep_test_struct_shards"
    out_dir:str="/home/natasha/multimodal_model/models/checkpoints/hpo_training/boltz_struct_only_vicreg"
    fig_dir:str="/home/natasha/multimodal_model/models/figures/hpo_training/boltz_struct_only_vicreg"
    run_tag:str="boltz_struct_only_vicreg"
    seed:int=31; batch_size:int=8; num_workers:int=0; epochs:int=30; min_epochs:int=5; patience:int=10
    tokens_per_interface_at_load:int=64; struct_shard_cache_size:int=8
    latent_dim:int=128; hidden_dim:int=128; heads:int=8; pool_queries:int=4; max_pair_types:int=16; dropout:float=0.1
    lr:float=3e-4; weight_decay:float=1e-2; grad_clip:float=1.0
    alpha:float=25.; beta:float=25.; delta:float=1.; gamma_var:float=1.; eps_var:float=1e-4; partial_auc_max_fpr:float=0.1
    selection_metric:str="weighted_per_peptide_auroc"

def parse_args():
    p=argparse.ArgumentParser(); defaults=asdict(RunConfig())
    for k,v in defaults.items():
        arg="--"+k.replace("_","-")
        if isinstance(v,bool): p.add_argument(arg,action=argparse.BooleanOptionalAction,default=v)
        elif isinstance(v,int): p.add_argument(arg,type=int,default=v)
        elif isinstance(v,float): p.add_argument(arg,type=float,default=v)
        else: p.add_argument(arg,default=v)
    return RunConfig(**vars(p.parse_args()))


def main():
    cfg=parse_args(); set_seed(cfg.seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out=Path(cfg.out_dir); fig=Path(cfg.fig_dir); out.mkdir(parents=True,exist_ok=True); fig.mkdir(parents=True,exist_ok=True)
    print("="*80); print("Boltz structure-only VICReg"); print(f"Device: {device}"); print(json.dumps(asdict(cfg),indent=2)); print("="*80, flush=True)
    train_ds=StructOnlyDataset(cfg.train_csv,cfg.struct_train_root,True,cfg,"train")
    val_ds=StructOnlyDataset(cfg.val_csv,cfg.struct_val_root,False,cfg,"val")
    test_ds=StructOnlyDataset(cfg.test_csv,cfg.struct_test_root,False,cfg,"test")
    imm_ds=StructOnlyDataset(cfg.immrep_csv,cfg.struct_immrep_root,False,cfg,"immrep_test") if cfg.immrep_csv else None
    train_loader=DataLoader(train_ds,cfg.batch_size,shuffle=True,num_workers=cfg.num_workers,collate_fn=collate,pin_memory=torch.cuda.is_available())
    val_loader=DataLoader(val_ds,cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,collate_fn=collate,pin_memory=torch.cuda.is_available())
    test_loader=DataLoader(test_ds,cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,collate_fn=collate,pin_memory=torch.cuda.is_available())
    imm_loader=None if imm_ds is None else DataLoader(imm_ds,cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,collate_fn=collate,pin_memory=torch.cuda.is_available())
    sample=next(iter(train_loader)); print(f"[batch] tcr_tokens={tuple(sample['tcr_tokens'].shape)} pmhc_tokens={tuple(sample['pmhc_tokens'].shape)}",flush=True)
    model=StructOnlyVICReg(cfg).to(device); opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    best={"epoch":-1,"selection_value":-math.inf,"state":None,"metrics":None}; bad=0; hist=[]
    for epoch in range(1,cfg.epochs+1):
        model.train(); run={}; steps=0
        for batch in train_loader:
            batch=move(batch,device); opt.zero_grad(set_to_none=True); zT,zP=model(batch); loss,parts=vicreg_loss(zT,zP,cfg,True); loss.backward()
            if cfg.grad_clip>0: torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
            opt.step(); steps+=1
            for k,v in parts.items(): run[k]=run.get(k,0.)+float(v)
        train={f"train_{k}":v/max(steps,1) for k,v in run.items()}
        val=evaluate(model,val_loader,device,cfg,"val"); m=val["metrics"]; sel=get_selection(m,cfg)
        hist.append({"epoch":epoch,**train,**{f"val_{k}":v for k,v in m.items() if isinstance(v,(int,float,str))}}); pd.DataFrame(hist).to_csv(out/f"{cfg.run_tag}__history.csv",index=False)
        print(f"Epoch {epoch:03d} | train_loss={train.get('train_loss',float('nan')):.4f} | inv={train.get('train_L_inv_mse',float('nan')):.4f} | var={train.get('train_L_var',float('nan')):.4f} | cov={train.get('train_L_cov',float('nan')):.4f} | val_auroc={m['auroc']:.4f} | val_auprc={m['auprc']:.4f} | val_pep_weighted={m['weighted_per_peptide_auroc']:.4f} | pos_mse={m['pos_mse_mean']:.4f} | neg_mse={m['neg_mse_mean']:.4f} | gap={m['mse_gap_neg_minus_pos']:.4f} | sel[{cfg.selection_metric}]={sel:.4f}", flush=True)
        if not math.isnan(sel) and sel>best["selection_value"]:
            best={"epoch":epoch,"selection_value":sel,"state":copy.deepcopy(model.state_dict()),"metrics":m}; bad=0
            torch.save({"config":asdict(cfg),"model_state_dict":best["state"],"epoch":epoch,"val_metrics":m,"selection_metric":cfg.selection_metric,"selection_value":sel}, out/f"{cfg.run_tag}__best.pt")
        else: bad+=1
        if epoch>=cfg.min_epochs and bad>=cfg.patience: print(f"Early stopping at epoch {epoch}",flush=True); break
    if best["state"] is not None: model.load_state_dict(best["state"])
    val=evaluate(model,val_loader,device,cfg,"val"); test=evaluate(model,test_loader,device,cfg,"test"); imm=None if imm_loader is None else evaluate(model,imm_loader,device,cfg,"immrep_test")
    for name,res in [("val",val),("test",test),("immrep_test",imm)]:
        if res is None: continue
        res["predictions"].to_csv(out/f"{cfg.run_tag}__{name}_predictions.csv",index=False); res["per_peptide"].to_csv(out/f"{cfg.run_tag}__{name}_per_peptide_metrics.csv",index=False)
    summary={"config":asdict(cfg),"best_epoch":best["epoch"],"selection_metric":cfg.selection_metric,"selection_value":best["selection_value"],"final_val_metrics":val["metrics"],"final_test_metrics":test["metrics"],"final_immrep_test_metrics":None if imm is None else imm["metrics"]}
    (out/f"{cfg.run_tag}__summary.json").write_text(json.dumps(summary,indent=2))
    torch.save({"config":asdict(cfg),"model_state_dict":model.state_dict(),"summary":summary}, out/f"{cfg.run_tag}__final.pt")
    print("Final validation metrics:", json.dumps(val["metrics"],indent=2), flush=True); print("Final test metrics:",json.dumps(test["metrics"],indent=2),flush=True)
    if imm: print("Final IMMREP test metrics:",json.dumps(imm["metrics"],indent=2),flush=True)
    print(f"Outputs written to: {out}", flush=True)

if __name__=="__main__": main()
