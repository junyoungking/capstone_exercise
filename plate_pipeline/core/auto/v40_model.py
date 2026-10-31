"""Controlled causal ST-GCN (spatial partitions) + causal TCN Phase Head.
Not an official-paper pretrained checkpoint or an exact teammate reproduction.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

EDGES=[(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
def adjacency():
    v=17;dist=np.full((v,v),1000,dtype=int);np.fill_diagonal(dist,0)
    for a,b in EDGES:dist[a,b]=dist[b,a]=1
    for k in range(v):dist=np.minimum(dist,dist[:,k,None]+dist[None,k,:])
    center=np.minimum(dist[:,11],dist[:,12]);a=np.zeros((3,v,v),np.float32)
    for j in range(v):
        for i in range(v):
            if i==j:a[0,i,j]=1
            elif dist[i,j]==1:a[1 if center[i]<=center[j] else 2,i,j]=1
    for k in range(3):a[k]/=np.maximum(a[k].sum(0,keepdims=True),1)
    return torch.tensor(a)

class FrameNorm(nn.Module):
    def __init__(self,c):super().__init__();self.norm=nn.LayerNorm(c)
    def forward(self,x):return self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
class TemporalConv(nn.Module):
    def __init__(self,ci,co,k=5,dilation=1):
        super().__init__();self.pad=(k-1)*dilation;self.conv=nn.Conv2d(ci,co,(k,1),dilation=(dilation,1))
    def forward(self,x):return self.conv(F.pad(x,(0,0,self.pad,0)))
class STBlock(nn.Module):
    def __init__(self,ci,co,a):
        super().__init__();self.register_buffer('a',a.clone());self.edge_importance=nn.Parameter(torch.ones_like(a));self.spatial=nn.Conv2d(ci,co*3,1)
        self.norm1=FrameNorm(co);self.temporal=TemporalConv(co,co);self.norm2=FrameNorm(co);self.res=nn.Identity() if ci==co else nn.Conv2d(ci,co,1);self.co=co
    def forward(self,x):
        b,_,t,v=x.shape;h=self.spatial(x).view(b,3,self.co,t,v)
        h=torch.einsum('bkctv,kvw->bctw',h,self.a*self.edge_importance)
        h=self.temporal(F.relu(self.norm1(h)));return F.relu(self.norm2(h)+self.res(x))
class CausalConv1d(nn.Module):
    def __init__(self,c,k=3,d=1):super().__init__();self.pad=(k-1)*d;self.conv=nn.Conv1d(c,c,k,dilation=d)
    def forward(self,x):return self.conv(F.pad(x,(self.pad,0)))
class TCNBlock(nn.Module):
    def __init__(self,c,d):
        super().__init__();self.c1=CausalConv1d(c,d=d);self.c2=CausalConv1d(c,d=d);self.n1=nn.LayerNorm(c);self.n2=nn.LayerNorm(c)
    def norm(self,x,n):return n(x.transpose(1,2)).transpose(1,2)
    def forward(self,x):
        h=F.relu(self.norm(self.c1(x),self.n1));h=self.norm(self.c2(h),self.n2);return F.relu(h+x)
class STGCN_TCN(nn.Module):
    def __init__(self,in_channels=8,base=32):
        super().__init__();a=adjacency();widths=[base,base,base,base*2,base*2,base*2];blocks=[];ci=in_channels
        for co in widths:blocks.append(STBlock(ci,co,a));ci=co
        self.backbone=nn.Sequential(*blocks);self.phase_tcn=nn.Sequential(*(TCNBlock(ci,d) for d in [1,2,4]));self.phase=nn.Conv1d(ci,3,1);self.action=nn.Conv1d(ci,3,1)
    def sequence(self,x):
        z=self.backbone(x).mean(-1);return self.action(z),self.phase(self.phase_tcn(z))
    def forward(self,x):
        a,p=self.sequence(x);return a[:,:,-1],p[:,:,-1]

def self_test():
    torch.manual_seed(42);m=STGCN_TCN(base=8).eval();x=torch.randn(2,8,32,17)
    with torch.no_grad():
        a,p=m(x);assert a.shape==p.shape==(2,3)
        xa=x.clone();xa[:,:,20:]+=20;a1,p1=m.sequence(x);a2,p2=m.sequence(xa)
        assert torch.allclose(a1[:,:,:20],a2[:,:,:20],atol=1e-5)
        assert torch.allclose(p1[:,:,:20],p2[:,:,:20],atol=1e-5)
    m.train();a,p=m(x);(a.square().mean()+p.square().mean()).backward();assert all(torch.isfinite(v.grad).all() for v in m.parameters() if v.grad is not None)
    return 'ST-GCN + TCN: shape / gradient / future-independence passed'
