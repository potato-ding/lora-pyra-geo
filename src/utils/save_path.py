import os
# 用于根据args返回路径
def get_save_pth(args):
    save_dir = os.path.join(
        'src/checkpoint',
        'dinov3' +
        (f'_lora{args.lora}' if (args.lora > 0) else '') +
        (f'_dora{args.dora}' if (args.dora > 0) else '') +
        ('_contrastive' if args.use_contrastive else '') +
        ('_triplet' if args.use_triplet else '') +
        (f'_{args.triplet_weight}w') + 
        (f'_{args.img_size}') + 
        (f'_mix' if args.use_mix else '') + 
        ('test')
    )
    return save_dir

def get_student_save_pth(args):
    save_dir = os.path.join(
        'src/checkpoint/student',
        ('_contrastive' if args.use_contrastive else '') +
        ('_triplet' if args.use_triplet else '') +
        (f'_{args.triplet_weight}w') + 
        (f'_{args.img_size}')
    )
    return save_dir