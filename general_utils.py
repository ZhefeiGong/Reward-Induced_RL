import numpy as np
import torch
import imageio
import torchvision.utils as vutils
import os
import random


#@func   : 
#@noteby : zhefei gong
class AttrDict(dict):
    __setattr__ = dict.__setitem__
    
    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)
    
    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self = d


#@func   : Without Use
#@noteby : zhefei gong
def get_padding(seq, replace_dim, size, val=0.0):
    """Returns padding tensor of same shape as seq, but with the target dimension replaced to 'size'.
       All values in returned array are set to 'val'."""
    seq_shape = seq.shape
    if isinstance(seq, torch.Tensor):
        return val * torch.ones(seq_shape[:replace_dim] + (size,) + seq_shape[replace_dim+1:], device=seq.device)
    else:
        return val * np.ones(seq_shape[:replace_dim] + (size,) + seq_shape[replace_dim + 1:])


#@func   : Without Use
#@noteby : zhefei gong
def stack_with_separator(tensors, dim, sep_width=2, sep_val=0.0):
    """Stacks list of tensors along given dimension, adds separator, brings to range [0...1]."""
    tensors = [(t + 1) / 2 if t.min() < 0.0 else t for t in tensors]
    stack_tensors = tensors[:1]
    if len(tensors) > 1:
        for tensor in tensors[1:]:
            assert tensor.shape == tensors[0].shape  # all stacked tensors must have same shape!
        separator = get_padding(stack_tensors[0], replace_dim=dim, size=sep_width, val=sep_val)
        for tensor in tensors[1:]:
            stack_tensors.extend([separator, tensor])
        stack_tensors = [np.concatenate(stack_tensors, axis=dim)]
    return stack_tensors[0]


#@func   : Without Use
#@noteby : zhefei gong
def make_image_seq_strip(imgs, n_logged_samples=5, sep_val=0.0):
    """Creates image strip where each row contains full rollout of sequence [each element of list makes one row]."""
    plot_imgs = stack_with_separator(imgs, dim=3, sep_val=sep_val)[:n_logged_samples]
    return stack_with_separator([t[:, 0] for t in np.split(plot_imgs, int(plot_imgs.shape[1] / 1), 1)],
                                dim=3, sep_val=sep_val)


#@func   : generate the gif figure
#@author : zhefei gong
def make_gif(imgs, path, fps_default = 10):
    return imageio.mimsave(path, imgs.astype(np.uint8), fps=fps_default)

#@func   : 
#@author : zhefei gong
def mean_distribution(imgs_pred_visual):
    mean_imgs_pred_visual = (imgs_pred_visual - imgs_pred_visual.min()) / (imgs_pred_visual.max() - imgs_pred_visual.min() + 1e-7) * 255.0
    return mean_imgs_pred_visual

#@func   : 
#@author : zhefei gong
def make_figure2(imgs_gt, imgs_pred, num_visual = 11):
    """
    @config :   1. imgs_pred and imgs_gt have the shape of [T,C,H,W]
    """
    assert len(imgs_pred) == len(imgs_gt)

    num_pics = len(imgs_pred)
    idxs = np.linspace(0, num_pics, num_visual, endpoint=False, dtype=int)
    imgs_pred_visual = mean_distribution(imgs_pred[idxs,:,:,:])
    imgs_gt_visual = imgs_gt[idxs,:,:,:] * 255.0

    # print(torch.unique(imgs_pred_visual[0,:,:,:]))
    # print(torch.unique(imgs_pred_visual[-1,:,:,:]))
    
    grid1 = vutils.make_grid(imgs_gt_visual, nrow=num_visual, pad_value=1)
    grid2 = vutils.make_grid(imgs_pred_visual, nrow=num_visual, pad_value=1)

    combined_grid = torch.cat((grid1, grid2), dim=1) 

    return combined_grid, idxs

#@func : 
def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or ``the entropy``.

    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor

#@func : 
def set_random_seed(num):
    pass

#@func : 
def set_seed(seed=42):
    random.seed(seed)  
    np.random.seed(seed)  
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
