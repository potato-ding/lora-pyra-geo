"""Explicit S3 dimensional controls, separate from immutable Top128/Random32."""
DIM_VARIANTS={
 "d1_top256_random32":{"target":"dual","top_dim":256,"random_dim":32,"top_head":"residual_mlp","random_head":"linear","allocation":"learnable"},
 "d2_top256_random64":{"target":"dual","top_dim":256,"random_dim":64,"top_head":"residual_mlp","random_head":"linear","allocation":"learnable"},
 "d3_top256":{"target":"top","top_dim":256,"random_dim":0,"top_head":"residual_mlp","branch_weight":1.},
 "d4_top512":{"target":"top","top_dim":512,"random_dim":0,"top_head":"residual_mlp","branch_weight":1.},
 "d5_top512_random32":{"target":"dual","top_dim":512,"random_dim":32,"top_head":"residual_mlp","random_head":"linear","allocation":"learnable"},
}

def is_dimension_config(cfg):return cfg.get("ablation",{}).get("variant") in DIM_VARIANTS
