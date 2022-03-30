import os
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resume(model, optimizer, scheduler, path, loc=None, mark="epoch", load_optims=True):

    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=loc)
        if not isinstance(checkpoint, dict):
            checkpoint = {"state_dict": checkpoint.state_dict()}

        model.trainables().load_state_dict(checkpoint["state_dict"])

        if load_optims and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

        if load_optims and "scheduler" in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])

        print(
            "=> loaded checkpoint '{}' (epoch {})".format(path, checkpoint.get(mark, 0))
        )
        del checkpoint
    else:
        print("=> no checkpoint found at '{}'".format(path))

    return model, optimizer, scheduler


def save_model(model, optimizer, scheduler, folder, mark="epoch", iter=0, gaccum=1):

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
        },
        os.path.join(folder, f"iter-{iter}.pth.tar"),
    )


def save_evals(inputs, outputs, answers, folder, tag="test", iter=0):
    path = os.path.join(folder, f"iter-{iter}-{tag}.tsv")
    with open(path, "w") as f:
        for (i, o, a) in zip(inputs, outputs, answers):
            print(f"{i}\t{o}\t{a}", file=f)
