"""Loss modules load only when explicitly selected by a consumer."""
from importlib import import_module
_EXPORTS={'PairInfoNCE':('pair_infonce','PairInfoNCE'),'nrkd':('nrkd','nrkd'),
 'margin_kd':('margin_kd','margin_kd'),'retrieval_distribution_kd':('retrieval_distribution_kd','retrieval_distribution_kd'),
 'local_covision_relation_kd':('local_covision_relation_kd','local_covision_relation_kd'),
 'residual_evidence_gates':('local_covision_relation_kd','residual_evidence_gates')}
__all__=list(_EXPORTS)
def __getattr__(name):
    if name not in _EXPORTS:raise AttributeError(name)
    module,attr=_EXPORTS[name]
    value=getattr(import_module('.'+module,__name__),attr)
    globals()[name]=value
    return value
