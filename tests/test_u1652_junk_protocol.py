from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from src.dataset.teacher.val_dataloaders import PairedCrossViewDatasetEval
from src.utils.train_eval_utils import getdist_1652_val_and_get_recall

def test_nonquery_gallery_identity_is_junk(tmp_path):
    for label in ['0001','0002']:
        folder=tmp_path/label;folder.mkdir()
        Image.new('RGB',(4,4)).save(folder/'a.png')
    gallery=PairedCrossViewDatasetEval(tmp_path,'gallery',sample_ids={'0001'})
    assert [gallery[i][1] for i in range(2)]==[1,-1]
    query=PairedCrossViewDatasetEval(tmp_path,'query')
    assert [query[i][1] for i in range(2)]==[1,2]

def test_rank_full_gallery_before_removing_junk():
    # Ties make sort-then-remove observably different from remove-then-sort.
    labels=torch.tensor([1,-1,2,1,-1,3,4,1,-1,5,6,1,7,8,9,-1,10])
    scores=np.ones(len(labels),dtype=np.float32)
    order=np.argsort(scores)[::-1].copy()
    ranked=labels[order].numpy();ranked=ranked[ranked!=-1]
    good=np.flatnonzero(ranked==1)
    ap=sum(((i/r if r else 1)+(i+1)/(r+1))/2 for i,r in enumerate(good))/len(good)*100
    q=(torch.ones(1,1),torch.tensor([1]),None)
    g=(torch.ones(len(labels),1),labels,None)
    loaders=[SimpleNamespace(dataset=range(n)) for n in [1,len(labels)]]
    actual=getdist_1652_val_and_get_recall(torch.nn.Identity(),*loaders,torch.device('cpu'),precomputed_features=(*q,*g))
    expected=[float((good<k).any())*100 for k in [1,5,10]]+[ap]
    np.testing.assert_allclose(actual,expected,atol=1e-5)
