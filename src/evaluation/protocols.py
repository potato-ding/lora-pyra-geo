"""The sole formal identity-retrieval protocol for University-1652 and SUES-200."""
RETRIEVAL_PROTOCOL = "query_identity_retrieval"
GTA_PROTOCOL = "gta_cross_area_d2s_pos_full_gallery_v1"


def require_selection_protocol(protocol):
    # Running/completed checkpoints already selected under these exact rules
    # carry this storage tag. It is a name equivalence, not a second protocol.
    checkpoint_storage_tag = "u1652_query_ids_junk_v1"
    if protocol not in (RETRIEVAL_PROTOCOL, checkpoint_storage_tag):
        raise RuntimeError("Checkpoint was not selected with the unified retrieval protocol")


GTA_SAME_AREA_PROTOCOL = "gta_same_area_d2s_pos_full_gallery_v1"


def gta_eval_splits(selection):
    """Expand the CLI selection in a stable cross-area then same-area order."""
    if selection == "both":
        return ("cross-area", "same-area")
    if selection not in ("cross-area", "same-area"):
        raise ValueError("Unsupported GTA split: " + str(selection))
    return (selection,)


def gta_protocol(split):
    if split == "cross-area":
        return GTA_PROTOCOL
    if split == "same-area":
        return GTA_SAME_AREA_PROTOCOL
    raise ValueError("Unsupported GTA split: " + str(split))


def gta_result_name(split):
    gta_protocol(split)
    return "test_gta_" + split.replace("-", "_") + "_d2s.json"
