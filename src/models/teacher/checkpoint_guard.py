NEW_FUSION_MODES = {"none", "layerwise_soft_orth"}


def removed_fusion_hparam_keys():
    return {
        "local_" + "feature_layers",
        "use_" + "local_fusion",
        "use_" + "soft_orth_fusion",
        "soft_orth_" + "lambda_init",
        "gamma_" + "max",
        "gamma_19_" + "parallel_init",
        "gamma_19_" + "perp_init",
        "gamma_27_" + "parallel_init",
        "gamma_27_" + "perp_init",
        "gamma_36_" + "init",
    }


def reject_removed_fusion_hparams(hparams, source):
    mode = hparams.get("fusion_mode")
    if mode not in (None, "") and str(mode).strip().lower() not in NEW_FUSION_MODES:
        raise RuntimeError(
            f"{source} uses removed teacher fusion mode {mode!r}. "
            "The legacy teacher fusion architecture has been deleted; "
            "train or convert a layerwise_soft_orth checkpoint."
        )

    removed_enable_key = "use_" + "soft_orth_fusion"
    if bool(hparams.get(removed_enable_key, False)):
        raise RuntimeError(
            f"{source} enables the removed teacher fusion architecture. "
            "Legacy checkpoints are intentionally unsupported; "
            "train a new layerwise_soft_orth teacher."
        )


def reject_removed_fusion_state_dict(state_dict, source):
    removed_exact = {
        "lambda" + "_orth_raw",
        "gamma" + "_raw",
    }
    removed_prefixes = (
        "local_" + "cross_attn.",
        "local_" + "proj.",
    )
    new_gamma_keys = {"gamma_detail_raw", "gamma_sem_raw"}

    removed_keys = []
    for raw_key in state_dict:
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        if key in removed_exact or key.startswith(removed_prefixes):
            removed_keys.append(raw_key)
            continue
        if (
            key.startswith("gamma_")
            and key.endswith("_raw")
            and key not in new_gamma_keys
        ):
            removed_keys.append(raw_key)

    if removed_keys:
        raise RuntimeError(
            f"{source} contains parameters from the removed teacher fusion "
            f"architecture, for example {removed_keys[:5]}. "
            "Legacy checkpoints are intentionally unsupported; "
            "train a new layerwise_soft_orth teacher."
        )


def validate_fusion_state_matches_model(state_dict, model, source):
    normalized_keys = {
        key[7:] if key.startswith("module.") else key
        for key in state_dict
    }
    checkpoint_is_layerwise = "lambda19_raw" in normalized_keys
    model_is_layerwise = hasattr(model, "lambda19_raw")
    if checkpoint_is_layerwise != model_is_layerwise:
        checkpoint_mode = (
            "layerwise_soft_orth" if checkpoint_is_layerwise else "none"
        )
        model_mode = getattr(model, "fusion_mode", "none")
        raise RuntimeError(
            f"{source} was saved with fusion_mode={checkpoint_mode}, but the "
            f"constructed teacher uses fusion_mode={model_mode}. Keep the new "
            "bset_metricis.json beside the checkpoint or pass the matching "
            "--fusion_mode explicitly."
        )
