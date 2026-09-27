"""Teacher learning-rate scheduler construction."""

from transformers import get_cosine_schedule_with_warmup


def get_scheduler(scheduler_type, train_steps, optimizer, warmup_steps=0, lr_end=None):
    if scheduler_type != 'cosine':
        raise ValueError('Formal Teacher requires cosine scheduler')
    print(f"\nScheduler: cosine - train_steps: {train_steps} - warmup_steps: {warmup_steps}")
    return get_cosine_schedule_with_warmup(
        optimizer, num_training_steps=train_steps, num_warmup_steps=warmup_steps)


build_teacher_scheduler = get_scheduler
