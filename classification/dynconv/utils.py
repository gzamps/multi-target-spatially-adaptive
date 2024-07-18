import torch.nn.functional as F
import torch

def apply_mask(x, mask):
    mask_hard = mask.hard
    assert mask_hard.shape[0] == x.shape[0]
    assert mask_hard.shape[2:4] == x.shape[2:4], (mask_hard.shape, x.shape)
    return mask_hard.float().expand_as(x) * x


def apply_inverse_mask(x, mask):
    mask_hard = 1 - mask.hard
    assert mask_hard.shape[0] == x.shape[0]
    assert mask_hard.shape[2:4] == x.shape[2:4], (mask_hard.shape, x.shape)
    return mask_hard.float().expand_as(x) * x


def apply_mask_scaled(x, mask, is_training=False):
    # print( x.shape)
    mask_hard = mask.hard
    # print ( mask_hard.shape)
    # ratio = torch.sum(mask.hard# keep_prob 
    total_elements_per_batch = x.shape[-1] * x.shape[-2]
    ones_sum_per_batch = torch.sum(mask_hard, dim=(2, 3)) / total_elements_per_batch # Sum along the height and width dimensions
    # print (ones_sum_per_batch.shape) 
    assert mask_hard.shape[0] == x.shape[0]
    assert mask_hard.shape[2:4] == x.shape[2:4], (mask_hard.shape, x.shape)
    if is_training:
        # print( "Training/w scaling")

        return mask_hard.float().expand_as(x)  * ones_sum_per_batch.unsqueeze(2).unsqueeze(3).float().expand_as(x) * x
    else:
        # print( "Inference/wO scaling")

        return mask_hard.float().expand_as(x)  *  x


def apply_mask_scaled_learnable(x, mask, factor, is_training=False):
    # print( "using scaler")
    # print( x.shape)
    mask_hard = mask.hard
    # print ( mask_hard.shape)
    # ratio = torch.sum(mask.hard# keep_prob 
    total_elements_per_batch = x.shape[-1] * x.shape[-2]
    ones_sum_per_batch = factor * torch.sum(mask_hard, dim=(2, 3)) / total_elements_per_batch # Sum along the height and width dimensions
    # print (ones_sum_per_batch.shape) 
    assert mask_hard.shape[0] == x.shape[0]
    assert mask_hard.shape[2:4] == x.shape[2:4], (mask_hard.shape, x.shape)
    
    # print( "Training/w scaling (learnable)")
    return mask_hard.float().expand_as(x)  * ones_sum_per_batch.unsqueeze(2).unsqueeze(3).float().expand_as(x) * x
    # if is_training:
    #     print( "Training/w scaling (learnable)")
    #     return mask_hard.float().expand_as(x)  * ones_sum_per_batch.unsqueeze(2).unsqueeze(3).float().expand_as(x) * x
    # else:
    #     print( "Inference/wO scaling (learnable)")

    #     return mask_hard.float().expand_as(x)  *  x

def ponder_cost_map(masks):
    """ takes in the mask list and returns a 2D image of ponder cost """
    assert isinstance(masks, list)
    out = None
    for mask in masks:
        m = mask['std'].hard
        assert m.dim() == 4
        m = m[0]  # only show the first image of the batch
        if out is None:
            out = m
        else:
            out += F.interpolate(m.unsqueeze(0),
                                 size=(out.shape[1], out.shape[2]), mode='nearest').squeeze(0)
    return out.squeeze(0).cpu().numpy()
