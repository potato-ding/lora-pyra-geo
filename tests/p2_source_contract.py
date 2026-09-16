"""Explicit approved P2 insertions for historical byte-exact source guards."""
BLOCKS=['    # P2_INTEGRATION_BEGIN\n    from .part2_integration import validate_config\n    is_p2 = validate_config(cfg)\n    # P2_INTEGRATION_END\n', "    # P2_INTEGRATION_BEGIN\n    if cfg.get('top_interface', 'linear') != 'linear':\n        from .part2_integration import prepare_top\n        prepare_top(supervision, cfg)\n    # P2_INTEGRATION_END\n", "    # P2_INTEGRATION_BEGIN\n    if cfg.get('top_interface', 'linear') != 'linear':\n        from .part2_integration import prepare_precision_groups\n        prepare_precision_groups(model, optimizer, cfg)\n    # P2_INTEGRATION_END\n", "    # P2_INTEGRATION_BEGIN\n    if cfg.get('top_interface', 'linear') != 'linear':\n        from .part2_integration import assert_precision\n        assert_precision(engine)\n    # P2_INTEGRATION_END\n", "        # P2_INTEGRATION_BEGIN\n        if cfg.get('top_interface', 'linear') != 'linear':\n            from .part2_integration import metadata\n            run_metadata.update(metadata(supervision))\n        # P2_INTEGRATION_END\n", "            # P2_INTEGRATION_BEGIN\n            if cfg.get('top_interface', 'linear') != 'linear' and step%200==0:\n                from .part2_integration import log_values\n                components.update(log_values(supervision))\n            # P2_INTEGRATION_END\n"]
def before_p2(path,text):
    if str(path) != 'src/student/train.py':return text
    # BNCC is a separately tested opt-in insertion; retain the original P2 source guard.
    import re
    text=re.sub(r'^([ ]*)# BNCC_BEGIN\n.*?^\1# BNCC_END\n','',text,flags=re.M|re.S)
    for block in BLOCKS:
        assert text.count(block)==1
        text=text.replace(block,'')
    assert text.count('            if not is_p2: validate_part1_config(cfg)')==1
    return text.replace('            if not is_p2: validate_part1_config(cfg)','            validate_part1_config(cfg)')
