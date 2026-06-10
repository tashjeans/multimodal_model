#!/usr/bin/env python3
"""No-training Boltz summary-metric baselines from precomputed interface shards.

This script evaluates whether simple summaries of the compressed Boltz interface
representation separate binders from decoys before building heavier models.
It uses the existing structural shards only; no raw Boltz .npz files are opened.
"""
from __future__ import annotations

import argparse, json, math
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, auc


def first_existing_col(df, cols): return next((c for c in cols if c in df.columns), None)

def normalise_manifest(df:pd.DataFrame, source:str)->pd.DataFrame:
    if "pair_id" not in df.columns: raise ValueError(f"{source}: CSV must contain pair_id")
    out=df.copy(); out["pair_id"]=out["pair_id"].astype(str)
    lab=first_existing_col(out,["binding_flag","label","binder","target"]); out["binding_flag"]=1 if lab is None else out[lab].astype(int)
    pep=first_existing_col(out,["Peptide","peptide","pep_seq","peptide_seq"]); out["peptide_for_eval"]=out[pep].astype(str) if pep else out["pair_id"].astype(str)
    for col,cands in {"tcra_len":["tcra_len","tcr_alpha_len","alpha_len"],"tcrb_len":["tcrb_len","tcr_beta_len","beta_len"],"pep_len":["pep_len","peptide_len"],"hla_len":["hla_len","mhc_len"]}.items():
        c=first_existing_col(out,cands); out[col]=pd.to_numeric(out[c],errors="coerce").fillna(0).astype(int) if c else 1
    complete=(out.tcra_len>0)&(out.tcrb_len>0)&(out.pep_len>0)&(out.hla_len>0)
    print(f"{source}: rows={len(out)} | complete_alpha_beta_pmhc={int(complete.sum())}", flush=True)
    return out[complete].reset_index(drop=True)

class StructShardStore:
    def __init__(self,root:Path,source:str,cache_size:int=8):
        self.root=Path(root); p=self.root/"struct_shard_index.json"
        if not p.exists(): raise FileNotFoundError(f"{source}: missing {p}")
        payload=json.loads(p.read_text()); self.index=payload["index"]; self.cache_size=cache_size; self.cache=OrderedDict(); self.source=source
        self.cap_per_block=int(payload.get("cap_per_block",128)); self.n_max=int(payload.get("n_max",512)); self.dz=int(payload.get("dz",128))
        print(f"{source}: struct index={self.root} | examples={len(self.index)} | cap={self.cap_per_block} | n_max={self.n_max}", flush=True)
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
    tokens=tokens[mask]; type_ids=type_ids[mask]; outs=[]; tys=[]
    for tid in torch.unique(type_ids):
        sel=(type_ids==tid).nonzero(as_tuple=True)[0]
        if cap>0 and sel.numel()>cap:
            idx=torch.linspace(0,sel.numel()-1,steps=cap).round().long(); sel=sel[idx]
        outs.append(tokens[sel]); tys.append(type_ids[sel])
    return (torch.cat(outs,0),torch.cat(tys,0)) if outs else (tokens,type_ids)

def cosine(a,b,eps=1e-8):
    return float(torch.dot(a,b)/(a.norm()*b.norm()+eps))

def safe_ratio(a,b,eps=1e-8): return float(a/(b+eps))

def compute_features(tokens:torch.Tensor,type_ids:torch.Tensor)->Dict[str,float]:
    out={}
    norms=tokens.norm(dim=1) if tokens.numel() else torch.zeros(1)
    out["global_token_norm_mean"]=float(norms.mean()); out["global_token_norm_std"]=float(norms.std(unbiased=False))
    mean_all=tokens.mean(0) if tokens.shape[0] else torch.zeros(tokens.shape[-1])
    out["global_mean_norm"]=float(mean_all.norm()); out["global_feature_std"]=float(tokens.std(unbiased=False)) if tokens.numel() else 0.0
    means={}; counts={}
    for tid,name in [(0,"tcr_pep"),(1,"pep_tcr"),(2,"tcr_hla"),(3,"hla_tcr")]:
        m=(type_ids==tid)
        counts[name]=int(m.sum())
        x=tokens[m]
        if x.shape[0]==0:
            mu=torch.zeros(tokens.shape[-1]); nrm=torch.tensor([0.])
        else:
            mu=x.mean(0); nrm=x.norm(dim=1)
        means[name]=mu
        out[f"{name}_n"]=counts[name]
        out[f"{name}_mean_norm"]=float(mu.norm())
        out[f"{name}_token_norm_mean"]=float(nrm.mean())
        out[f"{name}_token_norm_std"]=float(nrm.std(unbiased=False))
    out["tcr_pep_direction_mse"]=float((means["tcr_pep"]-means["pep_tcr"]).pow(2).mean())
    out["tcr_hla_direction_mse"]=float((means["tcr_hla"]-means["hla_tcr"]).pow(2).mean())
    out["tcr_pep_direction_cosine"]=cosine(means["tcr_pep"],means["pep_tcr"])
    out["tcr_hla_direction_cosine"]=cosine(means["tcr_hla"],means["hla_tcr"])
    out["tcrpep_to_tcrhla_norm_ratio"]=safe_ratio(out["tcr_pep_mean_norm"],out["tcr_hla_mean_norm"])
    out["peptcr_to_hlatcr_norm_ratio"]=safe_ratio(out["pep_tcr_mean_norm"],out["hla_tcr_mean_norm"])
    out["tcrpep_tcrhla_cosine"]=cosine(means["tcr_pep"],means["tcr_hla"])
    out["peptcr_hlatcr_cosine"]=cosine(means["pep_tcr"],means["hla_tcr"])
    out["direction_mse_sum"]=out["tcr_pep_direction_mse"]+out["tcr_hla_direction_mse"]
    out["direction_cosine_mean"]=(out["tcr_pep_direction_cosine"]+out["tcr_hla_direction_cosine"])/2.0
    return out

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


def per_peptide_summary(y,s,peps,max_fpr=0.1):
    rows=[]; df=pd.DataFrame({"label":y,"score":s,"peptide":peps})
    for pep,g in df.groupby("peptide"):
        yy=g.label.to_numpy(int); ss=g.score.to_numpy(float); valid=len(np.unique(yy))==2
        pr=partial_auc_raw(yy,ss,max_fpr) if valid else float("nan")
        pm=partial_auc_mcclish(yy,ss,max_fpr) if valid else float("nan")
        rows.append({"peptide":pep,"n":len(g),"auroc":safe_auroc(yy,ss) if valid else float("nan"),f"auc{max_fpr:g}_raw":pr,f"auc{max_fpr:g}_raw_div_maxfpr":pr/max_fpr if valid else float("nan"),f"auc{max_fpr:g}_norm":pm if valid else float("nan"),f"auc{max_fpr:g}_mcclish":pm if valid else float("nan"),"valid":valid})
    tab=pd.DataFrame(rows); vt=tab[tab.valid].copy() if len(tab) else tab
    if len(vt)==0:
        return {"macro_per_peptide_auroc":float("nan"),"weighted_per_peptide_auroc":float("nan"),f"macro_per_peptide_auc{max_fpr:g}_raw":float("nan"),f"macro_per_peptide_auc{max_fpr:g}_raw_div_maxfpr":float("nan"),f"macro_per_peptide_auc{max_fpr:g}_norm":float("nan"),f"macro_per_peptide_auc{max_fpr:g}_mcclish":float("nan"),"n_valid_peptides":0}
    return {"macro_per_peptide_auroc":float(vt.auroc.mean()),"weighted_per_peptide_auroc":float(np.average(vt.auroc,weights=vt.n)),f"macro_per_peptide_auc{max_fpr:g}_raw":float(vt[f"auc{max_fpr:g}_raw"].mean()),f"macro_per_peptide_auc{max_fpr:g}_raw_div_maxfpr":float(vt[f"auc{max_fpr:g}_raw_div_maxfpr"].mean()),f"macro_per_peptide_auc{max_fpr:g}_norm":float(vt[f"auc{max_fpr:g}_mcclish"].mean()),f"macro_per_peptide_auc{max_fpr:g}_mcclish":float(vt[f"auc{max_fpr:g}_mcclish"].mean()),"n_valid_peptides":int(len(vt))}

def evaluate_metric_table(features:pd.DataFrame,cfg:"RunConfig",split:str)->pd.DataFrame:
    y=features.label.to_numpy(int); peps=features.peptide.to_numpy(str)
    ignore={"split","pair_id","peptide","label"}; rows=[]
    for col in [c for c in features.columns if c not in ignore and pd.api.types.is_numeric_dtype(features[c])]:
        base=features[col].replace([np.inf,-np.inf],np.nan).fillna(features[col].replace([np.inf,-np.inf],np.nan).median()).to_numpy(float)
        for sign,label in [(1,"+"),(-1,"-")]:
            s=sign*base; pr=partial_auc_raw(y,s,cfg.partial_auc_max_fpr); pep=per_peptide_summary(y,s,peps,cfg.partial_auc_max_fpr)
            pm=partial_auc_mcclish(y,s,cfg.partial_auc_max_fpr)
            rows.append({"split":split,"feature":col,"orientation":label,"auroc":safe_auroc(y,s),"auprc":safe_auprc(y,s),f"auc{cfg.partial_auc_max_fpr:g}_raw":pr,f"auc{cfg.partial_auc_max_fpr:g}_raw_div_maxfpr":pr/cfg.partial_auc_max_fpr if not math.isnan(pr) else float("nan"),f"auc{cfg.partial_auc_max_fpr:g}_norm":pm,f"auc{cfg.partial_auc_max_fpr:g}_mcclish":pm,**pep})
    out=pd.DataFrame(rows).sort_values("auroc",ascending=False).reset_index(drop=True)
    return out

@dataclass
class RunConfig:
    val_csv:str="/home/natasha/multimodal_model/data/val/val_multiview.csv"
    test_csv:str="/home/natasha/multimodal_model/data/test/test_multiview.csv"
    immrep_csv:str="/home/natasha/multimodal_model/data/immrep_test/immrep_test_multiview.csv"
    struct_val_root:str="/home/natasha/multimodal_model/outputs_data/val_struct_shards"
    struct_test_root:str="/home/natasha/multimodal_model/outputs_data/test_struct_shards"
    struct_immrep_root:str="/home/natasha/multimodal_model/outputs_data/immrep_test_struct_shards"
    out_dir:str="/home/natasha/multimodal_model/models/checkpoints/hpo_training/boltz_summary_metrics"
    run_tag:str="boltz_summary_metrics"
    tokens_per_interface_at_load:int=128
    struct_shard_cache_size:int=8
    partial_auc_max_fpr:float=0.1

def parse_args():
    p=argparse.ArgumentParser(); defaults=asdict(RunConfig())
    for k,v in defaults.items():
        arg="--"+k.replace("_","-")
        if isinstance(v,int): p.add_argument(arg,type=int,default=v)
        elif isinstance(v,float): p.add_argument(arg,type=float,default=v)
        else: p.add_argument(arg,default=v)
    return RunConfig(**vars(p.parse_args()))

def process_split(name,csv_path,root,cfg):
    meta=normalise_manifest(pd.read_csv(csv_path),name); store=StructShardStore(Path(root),name,cfg.struct_shard_cache_size)
    meta=meta[meta.pair_id.astype(str).isin(store.pair_ids())].reset_index(drop=True)
    rows=[]
    for i,row in meta.iterrows():
        if i and i%1000==0: print(f"{name}: processed {i}/{len(meta)}",flush=True)
        tok,ty,mask=store.get(str(row.pair_id)); tok,ty=subsample_per_type(tok,ty,mask,cfg.tokens_per_interface_at_load)
        rows.append({"split":name,"pair_id":str(row.pair_id),"peptide":str(row.peptide_for_eval),"label":int(row.binding_flag),**compute_features(tok,ty)})
    feat=pd.DataFrame(rows); metrics=evaluate_metric_table(feat,cfg,name)
    return feat,metrics

def main():
    cfg=parse_args(); out=Path(cfg.out_dir); out.mkdir(parents=True,exist_ok=True)
    print("="*80); print("Boltz summary metric evaluation"); print(json.dumps(asdict(cfg),indent=2)); print("="*80,flush=True)
    all_summary={"config":asdict(cfg)}
    for name,csv,root in [("val",cfg.val_csv,cfg.struct_val_root),("test",cfg.test_csv,cfg.struct_test_root),("immrep_test",cfg.immrep_csv,cfg.struct_immrep_root)]:
        if not csv: continue
        feat,met=process_split(name,csv,root,cfg)
        feat.to_csv(out/f"{cfg.run_tag}__{name}_features.csv",index=False)
        met.to_csv(out/f"{cfg.run_tag}__{name}_single_metric_aurocs.csv",index=False)
        all_summary[f"{name}_top10_by_auroc"]=met.head(10).to_dict(orient="records")
        print(f"[{name}] top metrics by AUROC:"); print(met.head(10).to_string(index=False), flush=True)
    (out/f"{cfg.run_tag}__summary.json").write_text(json.dumps(all_summary,indent=2))
    print(f"Outputs written to: {out}", flush=True)
if __name__=="__main__": main()
