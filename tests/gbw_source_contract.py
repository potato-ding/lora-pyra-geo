"""Strictly remove the two approved GBW insertion blocks for historical source guards."""
TRAIN_APPLY="""    gbw_metrics={}
    if cfg.get('part') == 'Part-I':
        from .gbw import apply_branch_coefficients
        kd,gbw_metrics=apply_branch_coefficients(cfg,kd,kd_audit)
"""
TRAIN_LOG="""        if gbw_metrics:
            metrics.update(gbw_metrics,retrieval_loss=info.detach(),total_loss=total.detach())
"""
PART_GUARD="""    from .gbw import validate_coefficient_config
    name=validate_coefficient_config(cfg,name)
"""
def before_gbw(path,text):
    blocks={'src/student/train.py':[TRAIN_APPLY,TRAIN_LOG],'src/student/part1.py':[PART_GUARD]}
    for block in blocks.get(str(path),[]):
        assert text.count(block)==1
        text=text.replace(block,'')
    return text

