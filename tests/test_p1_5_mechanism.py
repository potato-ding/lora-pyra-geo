"""Part-I.5 measurement and immutability contracts."""
import ast,inspect,json
from pathlib import Path
from contextlib import ExitStack
import numpy as np
import pytest
import torch
from torch import nn
from src.tools import audit_p1_5_t128_r32_mechanism as a

def test_train_only_and_rejection(tmp_path):
    root=tmp_path/'train';root.mkdir();p=root/'drone'/'0001'/'a.jpg'
    assert a.train_guard(p,root)==p
    for split in ['test','SUES','GTA','val']:
        with pytest.raises(ValueError):a.train_guard(tmp_path/split/'a.jpg',root)

def test_no_update_no_checkpoint_writes():
    tree=ast.parse(inspect.getsource(a))
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            assert node.func.attr not in ['step','backward']
            assert not (node.func.attr=='save' and isinstance(node.func.value,ast.Name) and node.func.value.id=='torch')
    assert 'optimizer' not in {n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
    for name in ['prepare','worker','gradients','extract']:
        assert 'torch.save' not in inspect.getsource(getattr(a,name))

@pytest.mark.parametrize('error',[False,True])
def test_bn_all_buffer_restore_and_modes(error):
    m=nn.Sequential(nn.BatchNorm1d(3),nn.Dropout(.1))
    m.register_buffer('custom',torch.tensor(7))
    m.eval();before=a.old.state_hash(m)
    try:
        with a.restored_state(m,True):
            assert m.training and all(v.training for v in m.modules())
            m(torch.randn(64,3));m.custom.add_(5)
            assert m[0].num_batches_tracked==1
            if error:raise RuntimeError('exercise finally')
    except RuntimeError:
        assert error
    assert before==a.old.state_hash(m)
    assert not m.training and all(not v.training for v in m.modules())

def test_independent_branch_gradients_and_per_view():
    torch.manual_seed(4)
    z=torch.randn(64,512,requires_grad=True)
    heads={'top':(torch.randn(512,128),torch.randn(128)),'random':(torch.randn(512,32),torch.randn(32))}
    t=torch.nn.functional.normalize(torch.randn(64,128),dim=1)
    r=torch.nn.functional.normalize(torch.randn(64,32),dim=1)
    losses,(gt,gr)=a.branch_gradients(z,t,r,heads)
    p=torch.nn.functional.normalize(z@heads['top'][0]+heads['top'][1],dim=1)
    q=torch.nn.functional.normalize(z@heads['random'][0]+heads['random'][1],dim=1)
    total=a.DualSTSTSupervision._branch_loss(p,t,32)[0]+a.DualSTSTSupervision._branch_loss(q,r,32)[0]
    combined=torch.autograd.grad(total,z)[0].numpy()
    np.testing.assert_allclose(combined,gt+gr,atol=1e-8)
    assert z.grad is None
    values=a.gradient_arrays(gt,gr)
    assert values['grad_cos'].shape==(64,)
    assert a.stats(values['grad_cos'][:32])['n']==32 and a.stats(values['grad_cos'][32:])['n']==32
    assert losses[0]>0 and losses[1]>0

def test_gradient_cos_and_norm_formula():
    gt=np.array([[1,0],[0,1],[1,0]],float)
    gr=np.array([[1,0],[0,-2],[0,1]],float)
    v=a.gradient_arrays(gt,gr)
    np.testing.assert_allclose(v['grad_cos'],[1,-1,0])
    np.testing.assert_allclose(v['random_top_ratio'],[1,2,1])
    summary=a.summarize_grad(v)
    assert summary['fractions']['negative']==pytest.approx(1/3)
    assert summary['fractions']['near_zero_abs_le01']==pytest.approx(1/3)
    with pytest.raises(ValueError):a.gradient_arrays(np.zeros_like(gt),gr)

def test_angle_dispersion_and_wasserstein():
    np.testing.assert_allclose(a.angles([-1,0,1,1+1e-6]),[180,90,0,0])
    s=a.stats([1,2,3,4])
    assert s['mean']==2.5 and s['median']==2.5 and s['IQR']==1.5 and s['MAD']==1
    assert s['variance']==pytest.approx(5/3)
    assert a.wasserstein_distance([0,1],[1,2])==1
    assert a.wasserstein_distance([1,0],[2,1])==1
    assert a.ratio(1,0) is None

def toy_rows():
    rows=[]
    for i in range(40):
        pid=str(i)
        for domain,n in [('drone',2),('satellite',1)]:
            rows.extend(dict(pid=pid,domain=domain,path=str(a.TRAIN/domain/pid/f'{j}.jpg')) for j in range(n))
    split=dict(fit=[str(i) for i in range(8)],heldout=[str(i) for i in range(8,40)],
               internal_fit=[str(i) for i in range(4)],internal_val=[str(i) for i in range(4,8)])
    return rows,split

def test_matched_batch_manifest_deterministic():
    rows,split=toy_rows()
    x=a.make_batches(rows,split);y=a.make_batches(rows,split)
    assert x==y and x['same_all_seeds'] and len(x['batches'])==32
    for batch in x['batches']:
        assert len(batch['row_indices'])==64
        assert len(set(batch['identity_ids']))==32
        assert all(rows[i]['domain']=='drone' for i in batch['row_indices'][:32])
        assert all(rows[i]['domain']=='satellite' for i in batch['row_indices'][32:])
        assert batch['augmentation'] is False
        assert set(batch['identity_ids'])<=set(split['heldout'])

def test_common_relation_manifest_excludes_mismatched_positives():
    rows,split=toy_rows();z=np.random.default_rng(7).normal(size=(len(rows),8))
    first=a.make_relations(rows,split['heldout'],z)
    assert first==a.make_relations(rows,split['heldout'],z)
    assert first['pairs']['SAT_SAT']['positive']==[]
    for domain,sets in first['pairs'].items():
        for name,pairs in sets.items():
            for i,j in pairs:
                assert (rows[i]['pid']==rows[j]['pid'])==(name=='positive')
    t=a.relation_measure(z,first['pairs'])
    assert t['D2S']['positive']['cosine'].shape==(32,)

def test_ridge_no_heldout_influence():
    rows,split=toy_rows()
    rng=np.random.default_rng(5);x=rng.normal(size=(len(rows),5));y=rng.normal(size=(len(rows),4))
    weights,report=a.fit_heads(x,{'top':y},rows,split)
    altered=y.copy()
    held=np.array([r['pid'] in split['heldout'] for r in rows])
    altered[held]*=-30
    other,check=a.fit_heads(x,{'top':altered},rows,split)
    assert report['top']['lambda_value']==check['top']['lambda_value']
    assert np.array_equal(weights['top'][0],other['top'][0])
    assert np.array_equal(weights['top'][1],other['top'][1])
    bad=dict(split,heldout=split['heldout']+[split['fit'][0]])
    with pytest.raises(ValueError):a.fit_heads(x,{'top':y},rows,bad)

def test_random_untrained_head_forbidden():
    heads={k:(np.zeros((512,d)),np.zeros(d)) for k,d in [('top',128),('random',32)]}
    with pytest.raises(ValueError):a.require_diagnostic_heads('RANDOM_INITIALIZATION',heads)
    with pytest.raises(ValueError):a.require_diagnostic_heads('RIDGE_DIAGNOSTIC',{})
    a.require_diagnostic_heads('RIDGE_DIAGNOSTIC',heads)

def test_cross_seed_ddof():
    s=a.cross([1,2,3])
    assert s==dict(S0=1.,S1=2.,S2=3.,mean=2.,sample_std=1.)

def test_training_source_unchanged():
    import subprocess,hashlib
    from gbw_source_contract import before_gbw
    for path,sha in a.source_identity().items():
        # These later, separately tested modules did not exist in the historical
        # Part-I.5 seal. Keep the byte comparison for every pre-existing source.
        if path in ['src/student/gbw.py','src/student/gbw_smoke.py',
                    'src/student/part2.py','src/student/part2_input_audit.py',
                    'src/student/part2_smoke.py','src/student/part2_integration.py',
                    'src/student/part2_formal_smoke.py']:continue
        old=subprocess.check_output(['git','show','5cb84df6d041b0012968099ae089d456e5796a39:'+path],cwd=a.ROOT)
        assert old.decode()==before_gbw(path,(a.ROOT/path).read_text())

