"""Regression contracts for read-only TRAIN knowledge/BN diagnostics."""
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from src.tools import audit_d0_knowledge_absorption as audit


@pytest.mark.parametrize('relative', ['../test/query_drone/0001/x.jpg','../../SUES-200/150/x.jpg','../../GTA-UAV/train/x.jpg'])
def test_train_only_rejects_foreign_roots(tmp_path,relative):
    train=tmp_path/'U1652/train';train.mkdir(parents=True)
    with pytest.raises(ValueError): audit.train_guard(train/relative,train)


def test_train_path_and_symlink_guard(tmp_path):
    train=tmp_path/'train';train.mkdir()
    inside=train/'drone/0001/x.jpg'
    assert audit.train_guard(inside,train)==inside.resolve()
    external=tmp_path/'test';external.mkdir()
    (train/'escape').symlink_to(external,target_is_directory=True)
    with pytest.raises(ValueError): audit.train_guard(train/'escape/x.jpg',train)


def test_deterministic_disjoint_split():
    ids=[f'{i:04d}' for i in range(701)]
    a=audit.identity_split(ids);b=audit.identity_split(list(reversed(ids)))
    assert a==b
    assert len(a['fit'])==560 and len(a['heldout'])==141
    assert set(a['fit']).isdisjoint(a['heldout'])
    assert set(a['fit'])|set(a['heldout'])==set(ids)
    assert len(a['internal_fit'])==448 and len(a['internal_val'])==112
    assert set(a['internal_fit']).isdisjoint(a['internal_val'])
    assert set(a['internal_fit'])|set(a['internal_val'])==set(a['fit'])
    with pytest.raises(ValueError): audit.identity_split(ids[:-1])


def test_analytical_ridge_recovers_affine_mapping_without_optimizer(monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError('Optimizer forbidden')
    monkeypatch.setattr(torch.optim,'Adam',forbidden)
    monkeypatch.setattr(torch.optim,'SGD',forbidden)
    rng=np.random.default_rng(71)
    x=rng.normal(size=(100,8));w=rng.normal(size=(8,3));bias=rng.normal(size=3)
    coef,offset=audit.ridge_fit(x,x@w+bias,0)
    np.testing.assert_allclose(coef,w,atol=1e-12)
    np.testing.assert_allclose(offset,bias,atol=1e-12)


def test_probe_lambda_selection_cannot_see_heldout_targets():
    rng=np.random.default_rng(5)
    ids=[f'{i:04d}' for i in range(701)]
    split=audit.identity_split(ids)
    rows=[dict(pid=pid,domain=d) for d in audit.DOMAINS for pid in ids]
    x=rng.normal(size=(len(rows),6)); y=audit.unit(rng.normal(size=(len(rows),3)))
    c1,b1,_,r1=audit.fit_fresh_probe(x,y,rows,split)
    altered=y.copy()
    altered[[r['pid'] in split['heldout'] for r in rows]]*=-1
    c2,b2,_,r2=audit.fit_fresh_probe(x,altered,rows,split)
    np.testing.assert_array_equal(c1,c2);np.testing.assert_array_equal(b1,b2)
    assert r1['selected_lambda']==r2['selected_lambda']
    assert r1['internal_selection']==r2['internal_selection']
    with pytest.raises(ValueError):
        audit.fit_fresh_probe(x,y,[{**r,'path':'/forbidden/U1652/test/image.jpg'} for r in rows],split)


@pytest.mark.parametrize('raise_inside',[False,True])
def test_bn_restore_and_non_bn_eval_exact(raise_inside):
    model=nn.Sequential(nn.Linear(4,4),nn.BatchNorm1d(4),nn.Dropout(.9))
    model.train();initial=audit.state_hash(model)
    modes=[m.training for m in model.modules()]
    try:
        with audit.restored_bn(model):
            assert model[1].training and not model[0].training and not model[2].training
            with torch.no_grad(): model(torch.randn(16,4))
            assert model[1].num_batches_tracked==1
            if raise_inside: raise RuntimeError('controlled diagnostic failure')
    except RuntimeError:
        assert raise_inside
    assert audit.state_hash(model)==initial
    assert modes==[m.training for m in model.modules()]


def test_same_anchor_different_bn_context_changes_output_without_state_mutation():
    model=nn.BatchNorm1d(2).eval();before=audit.state_hash(model)
    anchor=torch.tensor([[1.,2.]])
    outputs=[]
    for companion in [torch.zeros(7,2),torch.ones(7,2)*10]:
        with audit.restored_bn(model),torch.no_grad():
            outputs.append(model(torch.cat([anchor,companion]))[0])
    assert not torch.equal(*outputs)
    assert audit.state_hash(model)==before


def test_context_manifest_matches_real_pairs_and_preserves_anchor():
    held=[f'{i:04d}' for i in range(141)]
    rows=[dict(pid=p,domain=d,path=f'/mock/{d}/{p}/{j}.jpg') for d in audit.DOMAINS for p in held
          for j in range(3 if d=='drone' else 1)]
    for size in [16,32,64]:
        anchors,contexts=audit.make_contexts(rows,held,size)
        assert contexts==audit.make_contexts(rows,held,size)[1]
        groups={}
        for c in contexts:
            assert len(c['row_indices'])==size
            assert len(set(c['pair_ids']))==size//2
            assert set(c['pair_ids'])<=set(held)
            items=[rows[i] for i in c['row_indices']]
            assert [r['pid'] for r in items[:size//2]]==[r['pid'] for r in items[size//2:]]
            assert all(r['domain']=='drone' for r in items[:size//2])
            assert all(r['domain']=='satellite' for r in items[size//2:])
            for pid in c['anchor_ids']:
                assert anchors[pid]['drone'] in c['row_indices'] and anchors[pid]['satellite'] in c['row_indices']
            groups.setdefault(c['group_start'],[]).append(c)
        assert all(len({tuple(c['row_indices']) for c in group})==8 for group in groups.values())
        assert sum(len(group[0]['anchor_ids']) for group in groups.values())==141


def test_metrics_and_geometry_identical_inputs():
    rng=np.random.default_rng(32);x=audit.unit(rng.normal(size=(20,5)))
    metric=audit.probe_metrics(x,x)
    assert metric['cosine_similarity_mean']==pytest.approx(1)
    assert metric['normalized_MSE']==pytest.approx(0,abs=1e-30)
    assert metric['R2']==pytest.approx(1)
    geom=audit.geometry_metrics(x@x.T,x@x.T,within=True)
    assert geom['Pearson']==pytest.approx(1) and geom['Spearman']==pytest.approx(1)
    assert geom['top_k_neighbor_overlap']=={'1':1.,'5':1.,'10':1.}
    assert geom['pair_count']==190
    context=np.repeat(x[:,None,:],8,axis=1)
    drift=audit.drift_arrays(context,x)
    assert np.max(np.abs(drift['one_minus_cosine']))<1e-14


def test_exact_official_target_semantics():
    from src.student.dual_stst import DualSTSTSupervision
    rng=torch.Generator().manual_seed(19)
    mean=torch.randn(768,generator=rng)
    basis=torch.linalg.qr(torch.randn(768,64,generator=rng)).Q
    owner=SimpleNamespace(teacher_mean=mean,top32_basis=basis[:,:32],random32_basis=basis[:,32:])
    desc=torch.nn.functional.normalize(torch.randn(10,768,generator=rng),dim=1)
    top,random=DualSTSTSupervision.teacher_targets(owner,desc)
    for actual,b in [(top,basis[:,:32]),(random,basis[:,32:])]:
        raw=(desc-mean)@b
        assert torch.equal(actual[1],raw)
        assert torch.equal(actual[0],torch.nn.functional.normalize(raw,dim=-1))


def test_output_schema_rejects_missing_and_unsafe_fields():
    with pytest.raises(ValueError): audit.validate_final_schema({})
    valid=dict(AUDIT_SOURCE_COMMIT='a'*40,TRAIN_ONLY=True,FIT_IDS=560,HELDOUT_IDS=141,
        IDENTITY_OVERLAP=0,ACTUAL_STUDENT_FORWARD_PATTERN='combined',BN_BATCH_SIZE_UNIQUE_VALUES=[64],
        TOP_PROBE_GAIN_3OF3=False,RANDOM_PROBE_GAIN_3OF3=False,D0_REDUCES_BN_DRIFT_3OF3=False,
        OPTIMIZER_STEP_CALLS=0,TRAINING_CHECKPOINT_CREATED=0,CHECKPOINTS_UNCHANGED=True,
        STST_ASSET_UNCHANGED=True,FORMAL_TRAINING_STARTED=False,AUDIT_COMPLETE=True,
        FACT=[],SUPPORTED_INTERPRETATION=[],UNRESOLVED=[])
    audit.validate_final_schema(json.loads(json.dumps(valid)))
    for key,value in [('OPTIMIZER_STEP_CALLS',1),('TRAIN_ONLY',False),('IDENTITY_OVERLAP',1),
                      ('CHECKPOINTS_UNCHANGED',False),('FORMAL_TRAINING_STARTED',True)]:
        with pytest.raises(ValueError): audit.validate_final_schema({**valid,key:value})


def test_audit_has_no_training_or_optimizer_entrypoint():
    import ast
    tree=ast.parse(inspect.getsource(audit))
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)]
    assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in ['step','backward'] for n in calls)
    assert not any(isinstance(n.func,ast.Attribute) and n.func.attr=='save' and
                   isinstance(n.func.value,ast.Name) and n.func.value.id=='torch' for n in calls)
