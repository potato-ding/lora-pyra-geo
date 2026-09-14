"""The sole approved Part-I.5 branch coefficient control; historical default unchanged."""
GBW_NAME='P1.5-T128-R32-GBW-S0'
GBW_WEIGHTS=(1.247,.753)

def validate_coefficient_config(cfg,default_name):
    top=cfg.get('lambda_top',1.)
    random=cfg.get('lambda_random',1.)
    if type(top) not in (int,float) or type(random) not in (int,float):
        raise ValueError('Branch coefficients must be numbers')
    name=cfg.get('experiment_name',default_name)
    if name==GBW_NAME:
        if (top,random)!=GBW_WEIGHTS or cfg.get('seed')!=0 or cfg.get('part1_variant')!='p1_t128_r32_s0':
            raise ValueError('Only the fixed GBW S0 is certified')
        if cfg.get('top_dim')!=128 or cfg.get('random_layout')!='single32' or cfg.get('random_total_dim')!=32:
            raise ValueError('GBW requires exact Top128 + Random32_A')
        return GBW_NAME
    if (top,random)!=(1.,1.):
        raise ValueError('No other weighting candidate is certified')
    return default_name

def apply_branch_coefficients(cfg,unweighted,audit):
    lt=cfg.get('lambda_top',1.)
    lr=cfg.get('lambda_random',1.)
    if (lt,lr)==(1.,1.):
        # Return the original tensor; do not re-associate any historical reduction.
        return unweighted,{}
    validate_coefficient_config(cfg,cfg.get('experiment_name'))
    top=audit['top_loss'];random=audit['random_loss']
    if random is None:raise ValueError('GBW requires an active Random32 branch')
    weighted_top=lt*top;weighted_random=lr*random
    dual=weighted_top+weighted_random
    return dual,dict(raw_top_loss=top.detach(),raw_random_loss=random.detach(),
        lambda_top=lt,lambda_random=lr,unweighted_dual_loss=unweighted.detach(),
        weighted_top_loss=weighted_top.detach(),weighted_random_loss=weighted_random.detach(),
        weighted_dual_loss=dual.detach())

