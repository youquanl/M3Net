import random
from collections.abc import Mapping, Sequence
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate


def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        # images = batch[0]["sam_features"] # useless
        offset = 0
        ind = 0
        pairing_points = []
        pairing_images = []
        grids_ = []
        batch_index = []
        batch_index.append(ind)
        flags = False
        pano_flags = False
        grid_flags = False
        for id , d in enumerate(batch):
            if "pairing_points" in d:
                flags = True
                pairing_points.append(d["pairing_points"])
                pairing_images.append(d["pairing_images"])
            if "grids" in d:
                grids_.append(d["grids"])
                batch[id].pop("grids")
                grid_flags = True
        
            if "org_coord" in d:
                pano_flags = True
        # if "pairing_points" in batch[0]:
        if flags:
            for batch_id in range(len(batch)):
                    pairing_points[batch_id][:] += offset
                    pairing_images[batch_id][:, 0] += batch_id 
                    offset += batch[batch_id]["discrete_coord"].shape[0]
                    # ind += batch[batch_id]["org_coord"].shape[0]
                    # batch_index.append(ind)
            pairing_points = torch.cat(pairing_points)
            pairing_images = torch.cat(pairing_images).int().contiguous()

            # batch.pop("grids")

        if pano_flags:
            for batch_id in range(len(batch)):
                ind += batch[batch_id]["org_coord"].shape[0]
                batch_index.append(ind)
        
        batch = {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
        # if "pairing_points" in batch:
        if flags:
            batch.pop("pairing_points")
            batch.pop("pairing_images")
            batch["pairing_points"] = pairing_points
            batch["pairing_images"] = pairing_images
        if grid_flags:
            batch["grids_"] = grids_
        
        if pano_flags:
            batch["batch_index"] = batch_index



        for key in batch.keys():
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)

        return batch
    else:
        return default_collate(batch)


def point_collate_fn(batch, mix_prob=0):
    assert isinstance(batch[0], Mapping)  # currently, only support input_dict, rather than input_list
    batch = collate_fn(batch)
    if "offset" in batch.keys():
        # Mix3d (https://arxiv.org/pdf/2110.02210.pdf)
        if random.random() < mix_prob:
            batch["offset"] = torch.cat([batch["offset"][1:-1:2], batch["offset"][-1].unsqueeze(0)], dim=0)
    return batch


def gaussian_kernel(dist2: np.array, a: float = 1, c: float = 5):
    return a * np.exp(-dist2 / (2 * c ** 2))