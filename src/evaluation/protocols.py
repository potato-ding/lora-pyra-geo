"""The sole formal identity-retrieval protocol for University-1652 and SUES-200."""
RETRIEVAL_PROTOCOL = "query_identity_retrieval"
GTA_PROTOCOL = "gta_cross_area_d2s_pos_full_gallery_v1"


def require_selection_protocol(protocol):
    # Running/completed checkpoints already selected under these exact rules
    # carry this storage tag. It is a name equivalence, not a second protocol.
    checkpoint_storage_tag = "u1652_query_ids_junk_v1"
    if protocol not in (RETRIEVAL_PROTOCOL, checkpoint_storage_tag):
        raise RuntimeError("Checkpoint was not selected with the unified retrieval protocol")
