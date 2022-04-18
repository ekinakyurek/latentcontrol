import os
import random
from typing import List, Optional, Union
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    return [seed]


def resume(
    model, optimizer, scheduler, path, loc=None, mark="epoch", load_optims=True
):

    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=loc)
        if not isinstance(checkpoint, dict):
            checkpoint = {"state_dict": checkpoint.state_dict()}

        model.load_state_dict(checkpoint["state_dict"], strict=False)

        if load_optims and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

        if load_optims and "scheduler" in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])

        print(
            "=> loaded checkpoint '{}' (epoch {})".format(
                path, checkpoint.get(mark, 0)
            )
        )
        del checkpoint
    else:
        print("=> no checkpoint found at '{}'".format(path))

    return model, optimizer, scheduler


def save_model(
    model,
    optimizer,
    scheduler,
    folder,
    mark="epoch",
    iter=0,
    gaccum=1,
    main_seed_state=None,
):
    np_seed_state = np.random.get_state()
    torch_seed_state = torch.random.get_rng_state()
    optimizer_state = optimizer.state_dict()
    model_state = model.state_dict()
    model_saved_state = {}
    for k in optimizer_state["param_groups"][0]["names"]:
        if k in model_state:
            model_saved_state[k] = model_state[k]

    torch.save(
        {
            "mark": iter,
            "gaccum": gaccum,
            "state_dict": model_saved_state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "np_seed_state": np_seed_state,
            "torch_seed_state": torch_seed_state,
            "main_seed_state": main_seed_state,
        },
        os.path.join(folder, f"iter-{iter}.pth.tar"),
    )


def split_seed(seed: Union[int, List]):
    if type(seed) == int:
        seed = [seed]
    rng = np.random.default_rng(seed)
    int1, int2 = rng.integers(0, np.iinfo(np.int32).max, size=2)
    new_seed = seed + [int2]
    old_seed = seed + [int1]
    return old_seed, new_seed


def worker_init_fn(worker_id, rank: int = 0, seed: Optional[List[int]] = None):
    if seed is None:
        np.random.seed(np.random.get_state()[1][0] + worker_id + rank)
    else:
        np.random.seed(seed + [worker_id, rank])


def save_evals(inputs, outputs, answers, folder, tag="test", iter=0):
    path = os.path.join(folder, f"iter-{iter}-{tag}.tsv")
    with open(path, "w") as f:
        for (i, o, a) in zip(inputs, outputs, answers):
            print(f"{i}\t{o}\t{a}", file=f)


def write_training_data_to_file(path, data):
    lines = "\n".join([str(d) for d in data])
    with open(path, "w") as handle:
        print(lines, file=handle)
