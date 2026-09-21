"""Short boundary collective, then CPU-store polling during long rank0 work."""
import json
import time
import uuid
import torch.distributed as dist

def run_rank0_selection(action, *, wait_timeout=86400, poll_interval=0.2):
    if not (dist.is_available() and dist.is_initialized()):
        return action()
    rank = dist.get_rank()
    # All training steps have finished when this short collective returns.
    key = [str(uuid.uuid4()) if rank == 0 else None]
    dist.broadcast_object_list(key, src=0)
    store = dist.distributed_c10d._get_default_store()
    result_key = 'teacher_selection/' + key[0]
    if rank == 0:
        try:
            payload = dict(ok=True, result=action())
            encoded = json.dumps(payload, allow_nan=False)
        except BaseException as exc:
            encoded = json.dumps(dict(ok=False, error=repr(exc)))
        store.set(result_key, encoded)
    deadline = time.monotonic() + wait_timeout
    # check() is a short CPU-store request, not a pending NCCL collective.
    while not store.check([result_key]):
        if time.monotonic() > deadline:
            raise TimeoutError('Teacher rank0 selection exceeded wait_timeout')
        time.sleep(poll_interval)
    payload = json.loads(store.get(result_key).decode())
    # All ranks have consumed the result before deleting its per-epoch key.
    dist.barrier()
    if rank == 0:
        store.delete_key(result_key)
    if not payload['ok']:
        raise RuntimeError('Teacher rank0 selection failed: ' + payload['error'])
    return payload['result']
