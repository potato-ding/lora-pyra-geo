from types import SimpleNamespace
import numpy as np
import torch
import cv2
import pytest
from PIL import Image
from src.dataset.teacher.val_dataloaders import build_sues200_val_dataloaders
from src.utils.train_eval_utils import run_sues_val_and_get_metrics
from src.evaluation.reference_metrics import sues_official_reference_metrics
from src.evaluation.certification import certify_pair, reuse_pair
import json

def test_loader_query_identity_filter_and_linear_resize(tmp_path):
    rgb=np.random.default_rng(5).integers(0,256,(11,13,3),dtype=np.uint8)
    for view in ['query_drone','query_satellite','gallery_drone','gallery_satellite']:
        for label in (['0002'] if view.startswith('query') else ['0001','0002']):
            p=tmp_path/'Testing/150'/view/label;p.mkdir(parents=True)
            Image.fromarray(rgb).save(p/'a.png')
    groups=build_sues200_val_dataloaders([7,7],2,0,str(tmp_path),['150'])
    for q,g in groups['150m'].values():
        assert q.dataset[0][1]==2
        assert [g.dataset[i][1] for i in range(2)]==[-1,2]
        expected=(cv2.resize(rgb,(7,7),interpolation=cv2.INTER_LINEAR_EXACT).astype(np.float32)/255.-np.array([.485,.456,.406]))/np.array([.229,.224,.225])
        np.testing.assert_allclose(q.dataset[0][0].permute(1,2,0).numpy(),expected,atol=1e-6)

def test_sues_excludes_distractors_matches_independent_reference():
    labels=torch.tensor([1,-1,2,1,-1,2,2,1,-1,2,2,1,2,2,2,-1,2])
    q=(torch.ones(2,1),torch.tensor([1,2]),None)
    g=(torch.ones(len(labels),1),labels,None)
    pair=[SimpleNamespace(dataset=range(n)) for n in [2,len(labels)]]
    got=run_sues_val_and_get_metrics(None,*pair,torch.device('cpu'),precomputed_features=(*q,*g))
    reference=sues_official_reference_metrics(q[0],q[1],g[0],g[1])
    for key in reference: assert got[key]==pytest.approx(reference[key],abs=1e-5)
    with pytest.raises(TypeError,match='horizontal_flip'):
        run_sues_val_and_get_metrics(None,*pair,torch.device('cpu'),horizontal_flip=True)

def test_old_sues_cache_is_rejected_before_reading_features(tmp_path):
    key='sues200_150m_D2S'
    (tmp_path/(key+'.npz')).write_bytes(b'old cache')
    (tmp_path/(key+'.json')).write_text(json.dumps({'horizontal_flip':True}))
    with pytest.raises(RuntimeError,match='incompatible'):
        certify_pair(None,None,'sues200','150m_D2S',torch.device('cpu'),tmp_path)
    with pytest.raises(RuntimeError,match='incompatible'):
        reuse_pair(None,None,'sues200','150m_D2S',torch.device('cpu'),tmp_path)
