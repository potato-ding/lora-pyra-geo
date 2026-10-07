import pytest
from src.evaluation.protocols import RETRIEVAL_PROTOCOL, require_selection_protocol

def test_only_current_selection_protocol_is_accepted():
    require_selection_protocol(RETRIEVAL_PROTOCOL)
    for protocol in (None, "legacy", "sample4geo_query_ids_junk_v1"):
        with pytest.raises(RuntimeError,match="unified retrieval protocol"):
            require_selection_protocol(protocol)

@pytest.mark.parametrize('changes',[{'split_type':'cross-view'},{'query_mode':'S2D'},{'mode':'pos_semipos'}])
def test_gta_protocol_cannot_be_switched(changes):
    from src.dataset.teacher.val_dataloaders import build_gta_val_dataloaders
    with pytest.raises(ValueError,match='Unified GTA protocol'):
        build_gta_val_dataloaders(img_size=[224,224],data_dir='unused',**changes)
