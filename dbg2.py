import torch, numpy as np
from src.prepare import build_cache
from src.causal import Store
from src.modeling import Probe
from src.csb import train_csb
from src.train_eval import run_fixed, pack_metrics
dev=torch.device("cuda")
c=build_cache(".",0,42); st=Store(c,dev)
p=Probe(8,768).to(dev); p.load_state_dict(torch.load("results/csb/probe_robust.pt",weights_only=True)); p.eval()
for q in p.parameters(): q.requires_grad_(False)
tr=np.where(c["split"]==0)[0][:3000]; dv=np.where(c["split"]==0)[0][:3000]
hl=c["hlen"][dv]; L=np.arange(64)[None]
m=(L<hl[:,None])&(L>=hl[:,None]-4)
pr,rk=run_fixed(p,st,dv,m); print("LastK dev", pack_metrics(c["y"][dv],pr,rk,m.sum(1),hl))
train_csb(p,st,tr,dv,3,128,3e-4,0,dev,name="dbg")

