#!/usr/bin/env python3
"""Plain sequence VICReg with Boltz neighbourhood guidance.

Base objective: positive-only VICReg between TCR sequence embedding and pMHC
sequence embedding, scored at inference by -MSE(zT, zPH).

Auxiliary objective: for positive training batches, preserve local Euclidean
neighbourhood geometry implied by fixed Boltz interface summaries. If two complexes
are close in Boltz interface space, their learned sequence-pair representation
should also be close.

Important: the Boltz summary vectors are NOT L2-normalised by default. Their
magnitude is preserved because magnitude may carry structural/confidence signal.
The optional normalisation in the neighbourhood loss rescales scalar pairwise
distance matrices only; it does not normalise the vectors themselves.

This uses the precomputed structural shards built by build_struct_shards.py. No raw
Boltz .npz files are opened during training.
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
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
from torch.utils.data import Dataset, DataLoader


def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def first_existing_col(df,cols): return next((c for c in cols if c in df.columns), None)

def to_str_list(x):
    if torch.is_tensor(x): x=x.detach().cpu().numpy().tolist()
    elif isinstance(x,np.ndarray): x=x.tolist()
    elif not isinstance(x,(list,tuple)): x=[x]
    return [v.decode("utf-8") if isinstance(v,bytes) else str(v) for v in x]


def normalise_manifest(df:pd.DataFrame, source:str)->pd.DataFrame:
    if "pair_id" not in df.columns: raise ValueError(f"{source}: CSV must contain pair_id")
    out=df.copy(); out["pair_id"]=out["pair_id"].astype(str)
    lab=first_existing_col(out,["binding_flag","label","binder","target"]); out["binding_flag"]=1 if lab is None else out[lab].astype(int)
    pep=first_existing_col(out,["Peptide","peptide","pep_seq","peptide_seq"]); out["peptide_for_eval"]=out[pep].astype(str) if pep else out["pair_id"].astype(str)
    length_specs={"tcra_len":["tcra_len","tcr_alpha_len","alpha_len"],"tcrb_len":["tcrb_len","tcr_beta_len","beta_len"],"pep_len":["pep_len","peptide_len"],"hla_len":["hla_len","mhc_len"]}
    seq_specs={"tcra_len":["tcra","tcr_alpha","TRA","cdr3a","alpha","alpha_seq","tcr_a"],"tcrb_len":["tcrb","tcr_beta","TRB","cdr3b","beta","beta_seq","tcr_b"],"pep_len":["Peptide","peptide","pep_seq","peptide_seq"],"hla_len":["hla","HLA","hla_seq","mhc","mhc_seq"]}
    for target,cands in length_specs.items():
        c=first_existing_col(out,cands)
        if c: out[target]=pd.to_numeric(out[c],errors="coerce").fillna(0).astype(int)
        else:
            sc=first_existing_col(out,seq_specs[target]); out[target]=out[sc].fillna("").astype(str).str.len().astype(int) if sc else 0
    return out

def complete_meta(csv_path:str, positives_only:bool, source:str)->pd.DataFrame:
    meta=normalise_manifest(pd.read_csv(csv_path),source); before=len(meta)
    if positives_only: meta=meta[meta.binding_flag.astype(int)==1].copy()
    after=len(meta); complete=(meta.tcra_len>0)&(meta.tcrb_len>0)&(meta.pep_len>0)&(meta.hla_len>0); meta=meta[complete].copy()
    print(f"{source}: rows={before} | after_label_filter={after} | complete_alpha_beta_pmhc={len(meta)}", flush=True)
    return meta.reset_index(drop=True)

class StructShardStore:
    def __init__(self,root:Path,source:str,cache_size:int=8):
        self.root=Path(root); p=self.root/"struct_shard_index.json"
        if not p.exists(): raise FileNotFoundError(f"{source}: missing {p}")
        payload=json.loads(p.read_text()); self.index=payload["index"]; self.cache_size=max(1,cache_size); self.cache=OrderedDict(); self.source=source
        self.cap_per_block=int(payload.get("cap_per_block",128)); self.dz=int(payload.get("dz",128))
        print(f"{source}: struct index={self.root} | examples={len(self.index)} | cap={self.cap_per_block}", flush=True)
    def pair_ids(self): return set(self.index.keys())
    def _load(self,name):
        if name in self.cache: self.cache.move_to_end(name); return self.cache[name]
        obj=torch.load(self.root/name,map_location="cpu"); self.cache[name]=obj; self.cache.move_to_end(name)
        while len(self.cache)>self.cache_size: self.cache.popitem(last=False)
        return obj
    def get(self,pid):
        rec=self.index[str(pid)]; sh=self._load(rec["shard"]); row=int(rec["row"])
        return sh["struct_tokens"][row].float(), sh["struct_type_ids"][row].long(), sh["struct_mask"][row].bool()

def subsample_per_type(tokens,type_ids,mask,cap):
    tokens=tokens[mask.bool()]; type_ids=type_ids[mask.bool()]; outs=[]; tys=[]
    for tid in torch.unique(type_ids):
        sel=(type_ids==tid).nonzero(as_tuple=True)[0]
        if cap>0 and sel.numel()>cap:
            idx=torch.linspace(0,sel.numel()-1,steps=cap).round().long(); sel=sel[idx]
        outs.append(tokens[sel]); tys.append(type_ids[sel])
    return (torch.cat(outs,0),torch.cat(tys,0)) if outs else (tokens,type_ids)

def boltz_per_type_mean(tokens, type_ids, n_types=4, l2_normalise: bool = False):
    """Magnitude-preserving fixed Boltz summary vector.

    Each example is summarised by concatenating the mean Boltz z-token vector for
    each directional interface type. By default this vector is NOT L2-normalised:
    preserving magnitude is deliberate because Boltz embedding magnitudes may carry
    interaction/confidence information. Set l2_normalise=True only for an explicit
    ablation.
    """
    parts=[]
    for tid in range(n_types):
        x=tokens[type_ids==tid]
        parts.append(x.mean(0) if x.numel() else torch.zeros(tokens.shape[-1],dtype=tokens.dtype))
    b=torch.cat(parts,0)
    return F.normalize(b,dim=0) if l2_normalise else b

class ESMStructRowDataset(Dataset):
    def __init__(self, embed_dir:Path, csv_path:str, struct_root:str, positives_only:bool, cfg:"RunConfig", source:str):
        self.embed_dir=Path(embed_dir); self.shards=sorted(self.embed_dir.glob("shard_*.pt")); self.cfg=cfg; self.source=source
        if not self.shards: raise FileNotFoundError(f"{source}: no shard_*.pt in {self.embed_dir}")
        self.meta=complete_meta(csv_path,positives_only,source); self.meta_by_pid={str(r.pair_id):r for _,r in self.meta.iterrows()}
        self.struct_store=StructShardStore(Path(struct_root),source,cfg.struct_shard_cache_size)
        allowed=set(self.meta_by_pid.keys()) & self.struct_store.pair_ids()
        self.index=[]; seen=0
        for sp in self.shards:
            shard=torch.load(sp,map_location="cpu")
            for bidx,batch in enumerate(shard):
                pids=to_str_list(batch["pair_id"])
                for ridx,pid in enumerate(pids):
                    seen+=1
                    if pid in allowed: self.index.append((sp,bidx,ridx,pid))
        print(f"{source}: esm_rows_seen={seen} | kept_complete_with_struct={len(self.index)}", flush=True)
        if not self.index: raise RuntimeError(f"{source}: no examples after ESM/CSV/struct matching")
        self._cache_path=None; self._cache_data=None
    def __len__(self): return len(self.index)
    def _load(self,sp):
        if self._cache_path!=sp: self._cache_data=torch.load(sp,map_location="cpu"); self._cache_path=sp
        return self._cache_data
    def __getitem__(self,idx):
        sp,bidx,ridx,pid=self.index[idx]; batch=self._load(sp)[bidx]; row=self.meta_by_pid[pid]
        st,ty,mask=self.struct_store.get(pid); st,ty=subsample_per_type(st,ty,mask,self.cfg.tokens_per_interface_at_load); b=boltz_per_type_mean(st, ty, 4, l2_normalise=self.cfg.l2_normalise_boltz_summary)
        return {"emb_T":batch["emb_T"][ridx].float(),"emb_P":batch["emb_P"][ridx].float(),"emb_H":batch["emb_H"][ridx].float(),"mask_T":batch["mask_T"][ridx].bool(),"mask_P":batch["mask_P"][ridx].bool(),"mask_H":batch["mask_H"][ridx].bool(),"boltz_summary":b.float(),"binding_flag":int(row.binding_flag),"pair_id":pid,"peptide":str(row.peptide_for_eval)}

def resolve_shard_dir(root:str,split:str)->Path:
    r=Path(root); cand=[r/split,r]
    for c in cand:
        if c.exists() and list(c.glob("shard_*.pt")): return c
    raise FileNotFoundError(f"Cannot resolve {split} shards under {root}")

def collate(rows):
    return {"emb_T":torch.stack([r["emb_T"] for r in rows]),"emb_P":torch.stack([r["emb_P"] for r in rows]),"emb_H":torch.stack([r["emb_H"] for r in rows]),"mask_T":torch.stack([r["mask_T"] for r in rows]),"mask_P":torch.stack([r["mask_P"] for r in rows]),"mask_H":torch.stack([r["mask_H"] for r in rows]),"boltz_summary":torch.stack([r["boltz_summary"] for r in rows]),"binding_flag":torch.tensor([r["binding_flag"] for r in rows],dtype=torch.long),"pair_id":[r["pair_id"] for r in rows],"peptide":[r["peptide"] for r in rows]}

class ESMProjectionHead(nn.Module):
    def __init__(self,D,rL,rD,d,L_max,dropout=0.1):
        super().__init__(); self.D=D; self.rL=rL; self.rD=rD; self.d=d; self.L_max=L_max
        self.B_c=nn.Parameter(torch.empty(D,rD)); self.A_c=nn.Parameter(torch.empty(L_max,rL)); self.H_c=nn.Parameter(torch.empty(rL*rD,d))
        nn.init.xavier_uniform_(self.B_c); nn.init.xavier_uniform_(self.A_c); nn.init.xavier_uniform_(self.H_c)
        self.expander=nn.Sequential(nn.Linear(d,d),nn.ReLU(),nn.Dropout(dropout),nn.Linear(d,d))
    def forward(self,emb,mask):
        B,L,D=emb.shape
        if D!=self.D or L>self.L_max: raise ValueError(f"shape mismatch {emb.shape}, expected D={self.D}, Lmax={self.L_max}")
        L_true=mask.sum(1); outs=[]
        for b in range(B):
            Lb=int(L_true[b].item())
            if Lb==0: outs.append(torch.zeros(self.d,device=emb.device,dtype=emb.dtype)); continue
            X=emb[b,:Lb,:]*mask[b,:Lb].unsqueeze(-1).float(); Y=X@self.B_c; U=self.A_c[:Lb,:].T@Y; outs.append(U.reshape(-1)@self.H_c)
        return self.expander(torch.stack(outs,0))

class PMHCProjectionHead(nn.Module):
    def __init__(self,D,rL,rD,d,L_P,L_H,R_PH=0.7,dropout=0.1):
        super().__init__(); dP=int(round(R_PH*d)); dH=d-dP; self.pep=ESMProjectionHead(D,rL,rD,dP,L_P,dropout); self.hla=ESMProjectionHead(D,rL,rD,dH,L_H,dropout)
    def forward(self,embP,maskP,embH,maskH): return torch.cat([self.pep(embP,maskP),self.hla(embH,maskH)],-1)

class RelationHead(nn.Module):
    def __init__(self,d,hidden=256,out_dim=128,dropout=0.1):
        super().__init__(); self.net=nn.Sequential(nn.LayerNorm(4*d),nn.Linear(4*d,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,out_dim))
    def forward(self,zT,zP): return self.net(torch.cat([zT,zP,zT-zP,zT*zP],-1))

def vicreg_variance(u,gamma,eps):
    u=u-u.mean(0,keepdim=True); std=torch.sqrt(u.var(0,unbiased=False)+eps); return F.relu(gamma-std).mean()
def vicreg_covariance(u):
    B,d=u.shape
    if B<=1: return torch.tensor(0.,device=u.device,dtype=u.dtype)
    u=u-u.mean(0,keepdim=True); cov=(u.T@u)/(B-1); off=cov-torch.diag_embed(torch.diag(cov)); return (off**2).sum()/d
def vicreg_loss(zT,zP,cfg):
    inv=F.mse_loss(zT,zP); var=vicreg_variance(zT,cfg.gamma_var,cfg.eps_var)+vicreg_variance(zP,cfg.gamma_var,cfg.eps_var); cov=vicreg_covariance(zT)+vicreg_covariance(zP)
    loss=cfg.alpha*inv+cfg.beta*var+cfg.delta*cov
    parts={"L_vicreg":float(loss.detach().cpu()),"L_inv_mse":float(inv.detach().cpu()),"L_var":float(var.detach().cpu()),"L_cov":float(cov.detach().cpu()),"weighted_inv":float((cfg.alpha*inv).detach().cpu()),"weighted_var":float((cfg.beta*var).detach().cpu()),"weighted_cov":float((cfg.delta*cov).detach().cpu()),"zT_std":float(zT.std(unbiased=False).detach().cpu()),"zPH_std":float(zP.std(unbiased=False).detach().cpu())}
    return loss,parts

def pairwise_sqdist(x): return torch.cdist(x,x,p=2).pow(2)
def offdiag_mask(B,device): return ~torch.eye(B,dtype=torch.bool,device=device)

def _scale_distances(d: torch.Tensor, mode: str, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return scaled distances plus the detached scale used.

    This rescales scalar pairwise distance matrices, not embeddings. The geometry is
    still Euclidean and magnitude-bearing at vector level; scaling only makes the
    auxiliary loss numerically comparable across learned-sequence and Boltz spaces.
    """
    mode = str(mode).lower()
    if mode in {"none", "raw", "off"}:
        return d, torch.tensor(1.0, device=d.device, dtype=d.dtype)
    positive = d.detach()[d.detach() > eps]
    if mode == "mean":
        scale = d.detach().mean().clamp_min(eps)
    elif mode == "median":
        scale = (torch.median(positive) if positive.numel() else d.detach().mean()).clamp_min(eps)
    else:
        raise ValueError(f"Unknown struct_distance_normalisation={mode!r}; use none, mean or median")
    return d / scale, scale

def _tau_value(dR_raw: torch.Tensor, dB_raw: torch.Tensor, dR_scale: torch.Tensor, dB_scale: torch.Tensor, cfg) -> torch.Tensor:
    """Choose tau for matching learned and Boltz distance units.

    fixed:      tau = cfg.struct_tau.
    batch_ratio: tau = median(dR_raw) / median(dB_raw), useful when using raw distances.
    scaled:     tau = 1, useful when both distance matrices have already been scaled.
    """
    eps = cfg.struct_distance_eps
    mode = str(cfg.struct_tau_mode).lower()
    if mode == "fixed":
        return torch.tensor(float(cfg.struct_tau), device=dR_raw.device, dtype=dR_raw.dtype)
    if mode == "scaled":
        return torch.tensor(1.0, device=dR_raw.device, dtype=dR_raw.dtype)
    if mode == "batch_ratio":
        rpos = dR_raw.detach()[dR_raw.detach() > eps]
        bpos = dB_raw.detach()[dB_raw.detach() > eps]
        rmed = (torch.median(rpos) if rpos.numel() else dR_raw.detach().mean()).clamp_min(eps)
        bmed = (torch.median(bpos) if bpos.numel() else dB_raw.detach().mean()).clamp_min(eps)
        return (rmed / bmed).detach()
    raise ValueError(f"Unknown struct_tau_mode={mode!r}; use fixed, batch_ratio or scaled")

def boltz_neighbour_loss(r,b,cfg):
    """Boltz-neighbourhood geometry loss with magnitude-preserving vectors.

    r: learned sequence relation representation, raw Euclidean coordinates.
    b: fixed Boltz structural summary, raw Euclidean coordinates; detached.

    The loss compares pairwise Euclidean distance matrices. Optional normalisation
    rescales the scalar distances only. It does not L2-normalise r or b, so vector
    magnitudes remain available to the distance geometry.
    """
    B=r.shape[0]
    empty={
        "L_struct":0.,"weighted_struct":0.,
        "raw_dR_mean":float("nan"),"raw_dB_mean":float("nan"),
        "raw_dR_median":float("nan"),"raw_dB_median":float("nan"),
        "scaled_dR_mean":float("nan"),"scaled_dB_mean":float("nan"),
        "corr_raw_dR_dB":float("nan"),"corr_scaled_dR_dB":float("nan"),
        "sigma_b2":float("nan"),"tau":float("nan"),
        "dR_scale":float("nan"),"dB_scale":float("nan"),
        "boltz_summary_norm_mean":float("nan"),"relation_norm_mean":float("nan"),
    }
    if B<3:
        z=torch.tensor(0.,device=r.device,dtype=r.dtype); return z,empty

    dR=pairwise_sqdist(r)
    dB=pairwise_sqdist(b.detach())
    m=offdiag_mask(B,r.device)
    dR_v=dR[m]
    dB_v=dB[m]
    eps=cfg.struct_distance_eps

    dR_use,dR_scale=_scale_distances(dR_v, cfg.struct_distance_normalisation, eps)
    dB_use,dB_scale=_scale_distances(dB_v, cfg.struct_distance_normalisation, eps)

    # Neighbourhood weights are computed from Boltz distances. By default this uses
    # the same scaled distance matrix used in the loss; setting struct_weight_space=raw
    # keeps weights in raw Boltz-distance units.
    weight_space=str(cfg.struct_weight_space).lower()
    dB_for_w = dB_v.detach() if weight_space == "raw" else dB_use.detach()
    nonzero=dB_for_w[dB_for_w>eps]
    sigma2=(torch.median(nonzero) if nonzero.numel() else dB_for_w.mean()).clamp_min(eps) if cfg.sigma_b<=0 else torch.tensor(cfg.sigma_b**2,device=r.device,dtype=r.dtype)
    w=torch.exp(-dB_for_w/sigma2)

    tau=_tau_value(dR_v, dB_v, dR_scale, dB_scale, cfg)
    loss=(w*(dR_use-tau*dB_use.detach()).pow(2)).sum()/w.sum().clamp_min(eps)

    with torch.no_grad():
        def _corr(a,bb):
            return torch.corrcoef(torch.stack([a,bb]))[0,1] if a.numel()>2 and a.std()>0 and bb.std()>0 else torch.tensor(float("nan"),device=r.device)
        raw_corr=_corr(dR_v.detach(), dB_v.detach())
        scaled_corr=_corr(dR_use.detach(), dB_use.detach())
        raw_dR_pos=dR_v.detach()[dR_v.detach()>eps]
        raw_dB_pos=dB_v.detach()[dB_v.detach()>eps]
    return loss,{
        "L_struct":float(loss.detach().cpu()),
        "weighted_struct":float((cfg.lambda_struct*loss).detach().cpu()),
        "raw_dR_mean":float(dR_v.detach().mean().cpu()),
        "raw_dB_mean":float(dB_v.detach().mean().cpu()),
        "raw_dR_median":float((torch.median(raw_dR_pos) if raw_dR_pos.numel() else dR_v.detach().mean()).cpu()),
        "raw_dB_median":float((torch.median(raw_dB_pos) if raw_dB_pos.numel() else dB_v.detach().mean()).cpu()),
        "scaled_dR_mean":float(dR_use.detach().mean().cpu()),
        "scaled_dB_mean":float(dB_use.detach().mean().cpu()),
        "corr_raw_dR_dB":float(raw_corr.detach().cpu()),
        "corr_scaled_dR_dB":float(scaled_corr.detach().cpu()),
        "sigma_b2":float(sigma2.detach().cpu()),
        "tau":float(tau.detach().cpu()),
        "dR_scale":float(dR_scale.detach().cpu()),
        "dB_scale":float(dB_scale.detach().cpu()),
        "boltz_summary_norm_mean":float(b.detach().norm(dim=-1).mean().cpu()),
        "relation_norm_mean":float(r.detach().norm(dim=-1).mean().cpu()),
    }

def score_mse(zT,zP):
    d=(zT-zP).pow(2).mean(-1); return d,-d

def safe_auroc(y,s): return float("nan") if len(np.unique(y))<2 else float(roc_auc_score(y,s))
def safe_auprc(y,s): return float("nan") if len(np.unique(y))<2 else float(average_precision_score(y,s))
def partial_auc_raw(y,s,max_fpr=0.1):
    if len(np.unique(y))<2: return float("nan")
    fpr,tpr,_=roc_curve(y,s); stop=np.searchsorted(fpr,max_fpr,side="right"); f=np.concatenate([fpr[:stop],[max_fpr]]); t=np.concatenate([tpr[:stop],[np.interp(max_fpr,fpr,tpr)]]); return float(auc(f,t))

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
        yy=g.label.to_numpy(int); ss=g.score.to_numpy(float); valid=len(np.unique(yy))==2; pr=partial_auc_raw(yy,ss,max_fpr) if valid else float("nan"); pm=partial_auc_mcclish(yy,ss,max_fpr) if valid else float("nan")
        rows.append({"peptide":pep,"n":len(g),"n_pos":int(yy.sum()),"n_neg":int((yy==0).sum()),"auroc":safe_auroc(yy,ss) if valid else float("nan"),f"auc{max_fpr:g}_raw":pr,f"auc{max_fpr:g}_raw_div_maxfpr":pr/max_fpr if valid else float("nan"),f"auc{max_fpr:g}_norm":pm if valid else float("nan"),f"auc{max_fpr:g}_mcclish":pm if valid else float("nan"),"valid":valid})
    tab=pd.DataFrame(rows); vt=tab[tab.valid].copy() if len(tab) else tab
    if len(vt)==0: summ={"macro_per_peptide_auroc":float("nan"),"weighted_per_peptide_auroc":float("nan"),f"macro_per_peptide_auc{max_fpr:g}_raw":float("nan"),f"weighted_per_peptide_auc{max_fpr:g}_raw":float("nan"),f"macro_per_peptide_auc{max_fpr:g}_raw_div_maxfpr":float("nan"),f"weighted_per_peptide_auc{max_fpr:g}_raw_div_maxfpr":float("nan"),f"macro_per_peptide_auc{max_fpr:g}_norm":float("nan"),f"weighted_per_peptide_auc{max_fpr:g}_norm":float("nan"),f"macro_per_peptide_auc{max_fpr:g}_mcclish":float("nan"),f"weighted_per_peptide_auc{max_fpr:g}_mcclish":float("nan"),"n_valid_peptides":0,"n_peptides_total":len(tab)}
    else: summ={"macro_per_peptide_auroc":float(vt.auroc.mean()),"weighted_per_peptide_auroc":float(np.average(vt.auroc,weights=vt.n)),f"macro_per_peptide_auc{max_fpr:g}_raw":float(vt[f"auc{max_fpr:g}_raw"].mean()),f"weighted_per_peptide_auc{max_fpr:g}_raw":float(np.average(vt[f"auc{max_fpr:g}_raw"],weights=vt.n)),f"macro_per_peptide_auc{max_fpr:g}_raw_div_maxfpr":float(vt[f"auc{max_fpr:g}_raw_div_maxfpr"].mean()),f"weighted_per_peptide_auc{max_fpr:g}_raw_div_maxfpr":float(np.average(vt[f"auc{max_fpr:g}_raw_div_maxfpr"],weights=vt.n)),f"macro_per_peptide_auc{max_fpr:g}_norm":float(vt[f"auc{max_fpr:g}_mcclish"].mean()),f"weighted_per_peptide_auc{max_fpr:g}_norm":float(np.average(vt[f"auc{max_fpr:g}_mcclish"],weights=vt.n)),f"macro_per_peptide_auc{max_fpr:g}_mcclish":float(vt[f"auc{max_fpr:g}_mcclish"].mean()),f"weighted_per_peptide_auc{max_fpr:g}_mcclish":float(np.average(vt[f"auc{max_fpr:g}_mcclish"],weights=vt.n)),"n_valid_peptides":int(len(vt)),"n_peptides_total":int(len(tab))}
    return tab.sort_values(["valid","n"],ascending=[False,False]), summ

def best_f1_threshold(scores,labels):
    best=None
    for thr in np.unique(scores):
        pred=(scores>=thr).astype(int); row={"threshold":float(thr),"f1":float(f1_score(labels,pred,zero_division=0)),"accuracy":float(accuracy_score(labels,pred)),"precision":float(precision_score(labels,pred,zero_division=0)),"recall":float(recall_score(labels,pred,zero_division=0))}
        if best is None or row["f1"]>best["f1"]: best=row
    return best or {"threshold":float("nan"),"f1":float("nan"),"accuracy":float("nan"),"precision":float("nan"),"recall":float("nan")}

def move(batch,device): return {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}

@torch.no_grad()
def evaluate(loader,tcr,pmhc,rel,device,cfg,split):
    tcr.eval(); pmhc.eval(); rel.eval(); rows=[]; run={}; steps=0
    for batch in loader:
        batch=move(batch,device); zT=tcr(batch["emb_T"],batch["mask_T"]); zP=pmhc(batch["emb_P"],batch["mask_P"],batch["emb_H"],batch["mask_H"]); vparts_loss,vparts=vicreg_loss(zT,zP,cfg); r=rel(zT,zP); sloss,sparts=boltz_neighbour_loss(r,batch["boltz_summary"],cfg); dist,score=score_mse(zT,zP); labels=batch["binding_flag"].cpu().numpy().astype(int)
        for i,pid in enumerate(batch["pair_id"]): rows.append({"split":split,"pair_id":pid,"peptide":batch["peptide"][i],"label":int(labels[i]),"mse_distance":float(dist[i].cpu()),"model_score":float(score[i].cpu())})
        for d in [vparts,sparts]:
            for k,v in d.items(): run[k]=run.get(k,0.)+float(v)
        steps+=1
    pred=pd.DataFrame(rows); y=pred.label.to_numpy(int); s=pred.model_score.to_numpy(float); peps=pred.peptide.to_numpy(str); tab,pep=per_peptide(y,s,peps,cfg.partial_auc_max_fpr); best=best_f1_threshold(s,y)
    pos=pred[pred.label==1].mse_distance; neg=pred[pred.label==0].mse_distance
    pr_global=partial_auc_raw(y,s,cfg.partial_auc_max_fpr); pm_global=partial_auc_mcclish(y,s,cfg.partial_auc_max_fpr)
    metrics={"split":split,"n":len(pred),"n_pos":int(y.sum()),"n_neg":int((y==0).sum()),"auroc":safe_auroc(y,s),"auprc":safe_auprc(y,s),f"auc{cfg.partial_auc_max_fpr:g}_raw":pr_global,f"auc{cfg.partial_auc_max_fpr:g}_raw_div_maxfpr":pr_global/cfg.partial_auc_max_fpr if not math.isnan(pr_global) else float("nan"),f"auc{cfg.partial_auc_max_fpr:g}_norm":pm_global,f"auc{cfg.partial_auc_max_fpr:g}_mcclish":pm_global,**pep,"pos_mse_mean":float(pos.mean()) if len(pos) else float("nan"),"neg_mse_mean":float(neg.mean()) if len(neg) else float("nan"),"mse_gap_neg_minus_pos":float(neg.mean()-pos.mean()) if len(pos) and len(neg) else float("nan"),**{f"mean_{k}":v/max(steps,1) for k,v in run.items()},**{f"best_f1_{k}":v for k,v in best.items()}}
    return {"predictions":pred,"per_peptide":tab,"metrics":metrics}

def infer_shapes(ds):
    s=ds[0]; return s["emb_T"].shape[-1],s["emb_T"].shape[0],s["emb_P"].shape[0],s["emb_H"].shape[0]

def selection(metrics,cfg):
    for k in [cfg.selection_metric,"weighted_per_peptide_auroc","auroc"]:
        v=metrics.get(k)
        if isinstance(v,(int,float)) and not math.isnan(float(v)): return float(v)
    return float("nan")

@dataclass
class RunConfig:
    embed_root:str="/home/natasha/multimodal_model/models/embeddings/no_boltz_multiview_ids"
    train_csv:str="/home/natasha/multimodal_model/data/train/train_multiview.csv"
    val_csv:str="/home/natasha/multimodal_model/data/val/val_multiview.csv"
    test_csv:str="/home/natasha/multimodal_model/data/test/test_multiview.csv"
    immrep_csv:str="/home/natasha/multimodal_model/data/immrep_test/immrep_test_multiview.csv"
    immrep_shard_dir:str="/home/natasha/multimodal_model/models/embeddings/immrep_test_set/test"
    struct_train_root:str="/home/natasha/multimodal_model/outputs_data/train_struct_shards"
    struct_val_root:str="/home/natasha/multimodal_model/outputs_data/val_struct_shards"
    struct_test_root:str="/home/natasha/multimodal_model/outputs_data/test_struct_shards"
    struct_immrep_root:str="/home/natasha/multimodal_model/outputs_data/immrep_test_struct_shards"
    out_dir:str="/home/natasha/multimodal_model/models/checkpoints/hpo_training/plain_vicreg_boltz_neighbourhood"
    fig_dir:str="/home/natasha/multimodal_model/models/figures/hpo_training/plain_vicreg_boltz_neighbourhood"
    run_tag:str="plain_vicreg_boltz_neighbourhood"
    seed:int=31; batch_size:int=16; num_workers:int=0; epochs:int=30; min_epochs:int=10; patience:int=10
    rL:int=8; rD:int=16; d:int=128; R_PH:float=0.7; dropout:float=0.1
    lr:float=3e-4; weight_decay:float=1e-2; grad_clip:float=1.0
    alpha:float=25.; beta:float=25.; delta:float=1.; gamma_var:float=1.; eps_var:float=1e-4
    lambda_struct:float=0.05
    # Boltz vectors are magnitude-preserving by default. Only enable this for an explicit ablation.
    l2_normalise_boltz_summary:bool=False
    # Pairwise-distance matrix scaling: none, mean, or median. This does not normalise vectors.
    struct_distance_normalisation:str="median"
    struct_distance_eps:float=1e-6
    # Tau controls unit matching between learned-representation and Boltz distances.
    # fixed: use struct_tau; scaled: tau=1; batch_ratio: median(dR_raw)/median(dB_raw).
    struct_tau_mode:str="scaled"
    struct_tau:float=1.0
    # sigma_b <= 0 uses median Boltz distance in the selected weight space.
    sigma_b:float=0.0
    # Weight neighbourhoods using raw or scaled Boltz distances.
    struct_weight_space:str="scaled"
    relation_hidden:int=256; relation_dim:int=128; tokens_per_interface_at_load:int=64; struct_shard_cache_size:int=8
    partial_auc_max_fpr:float=0.1; selection_metric:str="weighted_per_peptide_auroc"

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
    cfg=parse_args(); set_seed(cfg.seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); out=Path(cfg.out_dir); out.mkdir(parents=True,exist_ok=True); Path(cfg.fig_dir).mkdir(parents=True,exist_ok=True)
    print("="*80); print("Plain VICReg + Boltz neighbourhood guidance"); print(f"Device: {device}"); print(json.dumps(asdict(cfg),indent=2)); print("="*80,flush=True)
    train_ds=ESMStructRowDataset(resolve_shard_dir(cfg.embed_root,"train"),cfg.train_csv,cfg.struct_train_root,True,cfg,"train")
    val_ds=ESMStructRowDataset(resolve_shard_dir(cfg.embed_root,"val"),cfg.val_csv,cfg.struct_val_root,False,cfg,"val")
    test_ds=ESMStructRowDataset(resolve_shard_dir(cfg.embed_root,"test"),cfg.test_csv,cfg.struct_test_root,False,cfg,"test")
    imm_dir=Path(cfg.immrep_shard_dir) if cfg.immrep_shard_dir else resolve_shard_dir(cfg.embed_root,"immrep_test")
    imm_ds=ESMStructRowDataset(imm_dir,cfg.immrep_csv,cfg.struct_immrep_root,False,cfg,"immrep_test") if cfg.immrep_csv else None
    train_loader=DataLoader(train_ds,cfg.batch_size,shuffle=True,num_workers=cfg.num_workers,collate_fn=collate,pin_memory=torch.cuda.is_available())
    val_loader=DataLoader(val_ds,cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,collate_fn=collate,pin_memory=torch.cuda.is_available())
    test_loader=DataLoader(test_ds,cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,collate_fn=collate,pin_memory=torch.cuda.is_available())
    imm_loader=None if imm_ds is None else DataLoader(imm_ds,cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,collate_fn=collate,pin_memory=torch.cuda.is_available())
    D,LT,LP,LH=infer_shapes(train_ds); print(f"Detected shapes | D={D} | L_T={LT} | L_P={LP} | L_H={LH}", flush=True)
    tcr=ESMProjectionHead(D,cfg.rL,cfg.rD,cfg.d,LT,cfg.dropout).to(device); pmhc=PMHCProjectionHead(D,cfg.rL,cfg.rD,cfg.d,LP,LH,cfg.R_PH,cfg.dropout).to(device); rel=RelationHead(cfg.d,cfg.relation_hidden,cfg.relation_dim,cfg.dropout).to(device)
    opt=torch.optim.AdamW(list(tcr.parameters())+list(pmhc.parameters())+list(rel.parameters()),lr=cfg.lr,weight_decay=cfg.weight_decay)
    best={"epoch":-1,"selection_value":-math.inf,"state":None}; bad=0; hist=[]
    for epoch in range(1,cfg.epochs+1):
        tcr.train(); pmhc.train(); rel.train(); run={}; steps=0
        for batch in train_loader:
            batch=move(batch,device); opt.zero_grad(set_to_none=True); zT=tcr(batch["emb_T"],batch["mask_T"]); zP=pmhc(batch["emb_P"],batch["mask_P"],batch["emb_H"],batch["mask_H"]); vloss,vparts=vicreg_loss(zT,zP,cfg); r=rel(zT,zP); sloss,sparts=boltz_neighbour_loss(r,batch["boltz_summary"],cfg); loss=vloss+cfg.lambda_struct*sloss; loss.backward()
            if cfg.grad_clip>0: torch.nn.utils.clip_grad_norm_(list(tcr.parameters())+list(pmhc.parameters())+list(rel.parameters()),cfg.grad_clip)
            opt.step(); steps+=1
            for d in [vparts,sparts,{"L_total":float(loss.detach().cpu())}]:
                for k,v in d.items(): run[k]=run.get(k,0.)+float(v)
        train={f"train_{k}":v/max(steps,1) for k,v in run.items()}; val=evaluate(val_loader,tcr,pmhc,rel,device,cfg,"val"); m=val["metrics"]; sel=selection(m,cfg)
        hist.append({"epoch":epoch,**train,**{f"val_{k}":v for k,v in m.items() if isinstance(v,(int,float,str))}}); pd.DataFrame(hist).to_csv(out/f"{cfg.run_tag}__history.csv",index=False)
        print(f"Epoch {epoch:03d} | total={train.get('train_L_total',float('nan')):.4f} | vicreg={train.get('train_L_vicreg',float('nan')):.4f} | struct={train.get('train_L_struct',float('nan')):.4f} | w_struct={train.get('train_weighted_struct',float('nan')):.4f} | corr_raw={train.get('train_corr_raw_dR_dB',float('nan')):.4f} | corr_scaled={train.get('train_corr_scaled_dR_dB',float('nan')):.4f} | tau={train.get('train_tau',float('nan')):.4f} | val_auroc={m['auroc']:.4f} | val_auprc={m['auprc']:.4f} | val_pep_weighted={m['weighted_per_peptide_auroc']:.4f} | pos_mse={m['pos_mse_mean']:.4f} | neg_mse={m['neg_mse_mean']:.4f} | gap={m['mse_gap_neg_minus_pos']:.4f} | sel[{cfg.selection_metric}]={sel:.4f}", flush=True)
        if not math.isnan(sel) and sel>best["selection_value"]:
            best={"epoch":epoch,"selection_value":sel,"state":{"tcr":copy.deepcopy(tcr.state_dict()),"pmhc":copy.deepcopy(pmhc.state_dict()),"rel":copy.deepcopy(rel.state_dict())},"metrics":m}; bad=0
            torch.save({"config":asdict(cfg),"state":best["state"],"epoch":epoch,"val_metrics":m,"selection_value":sel},out/f"{cfg.run_tag}__best.pt")
        else: bad+=1
        if epoch>=cfg.min_epochs and bad>=cfg.patience: print(f"Early stopping at epoch {epoch}",flush=True); break
    if best["state"]: tcr.load_state_dict(best["state"]["tcr"]); pmhc.load_state_dict(best["state"]["pmhc"]); rel.load_state_dict(best["state"]["rel"])
    val=evaluate(val_loader,tcr,pmhc,rel,device,cfg,"val"); test=evaluate(test_loader,tcr,pmhc,rel,device,cfg,"test"); imm=None if imm_loader is None else evaluate(imm_loader,tcr,pmhc,rel,device,cfg,"immrep_test")
    for name,res in [("val",val),("test",test),("immrep_test",imm)]:
        if res is None: continue
        res["predictions"].to_csv(out/f"{cfg.run_tag}__{name}_predictions.csv",index=False); res["per_peptide"].to_csv(out/f"{cfg.run_tag}__{name}_per_peptide_metrics.csv",index=False)
    summary={"config":asdict(cfg),"best_epoch":best["epoch"],"selection_value":best["selection_value"],"final_val_metrics":val["metrics"],"final_test_metrics":test["metrics"],"final_immrep_test_metrics":None if imm is None else imm["metrics"]}
    (out/f"{cfg.run_tag}__summary.json").write_text(json.dumps(summary,indent=2))
    torch.save({"config":asdict(cfg),"tcr_state_dict":tcr.state_dict(),"pmhc_state_dict":pmhc.state_dict(),"relation_state_dict":rel.state_dict(),"summary":summary}, out/f"{cfg.run_tag}__final.pt")
    print("Final validation metrics:",json.dumps(val["metrics"],indent=2),flush=True); print("Final test metrics:",json.dumps(test["metrics"],indent=2),flush=True)
    if imm: print("Final IMMREP test metrics:",json.dumps(imm["metrics"],indent=2),flush=True)
    print(f"Outputs written to: {out}",flush=True)
if __name__=="__main__": main()
