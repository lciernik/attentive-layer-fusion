import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from src.tasks.attentive_probe import DimensionAlignment, AttentiveProbeModel



def _attention_probe_structure_check(model: torch.nn.Module):
    """ We assume that the model contains of a DimensionAlignment module, and an AttentiveProbeModel module.
    Both modules are expected to be wrapped in a Sequential module.
    """
    if not isinstance(model, torch.nn.Sequential):
        return 0 #raise ValueError("The model must be a Sequential module")
    if len(model._modules) != 2 or not isinstance(model[0], DimensionAlignment) or not isinstance(model[1], AttentiveProbeModel):
        raise ValueError("The model must be a Sequential module containing a DimensionAlignment module and an AttentiveProbeModel module")
    return 1
    

def get_attention_weights(model: torch.nn.Module, dataloader: DataLoader) -> np.ndarray:
    """
    Extracts and returns the attention weights from the AttentiveProbeModel during a forward pass over the dataloader.

    This function assumes that the model is a torch.nn.Sequential containing a DimensionAlignment module
    followed by an AttentiveProbeModel. It temporarily enables attention recording in the cross-attention
    block, runs the model over the provided dataloader, collects the recorded attention weights, and then
    disables attention recording.

    Args:
        model (torch.nn.Module): The model containing the attention mechanism.
        dataloader (DataLoader): The dataloader providing input features.

    Returns:
        np.ndarray: The stacked and squeezed attention weights recorded during the forward passes.
    """
    seq = _attention_probe_structure_check(model)

    if seq:
        def _set_record_attention():
            model[1].attn_block.cross_attn.record_attention = True
            model[1].attn_block.cross_attn.attn_rec = []
    
        def _reset_record_attention():
            model[1].attn_block.cross_attn.record_attention = False
            model[1].attn_block.cross_attn.attn_rec = []
    else:
        def _set_record_attention():
            model.attn_block.cross_attn.record_attention = True
            model.attn_block.cross_attn.attn_rec = []

        def _reset_record_attention():
            model.attn_block.cross_attn.record_attention = False
            model.attn_block.cross_attn.attn_rec = []

    attn_rec = None
    try:
        _set_record_attention()
        for features, _ in dataloader:
            model(features)
        if seq:
            attn_rec = np.stack(model[1].attn_block.cross_attn.attn_rec)
        else:
            # It can have different dim?
            #attn_rec = model.attn_block.cross_attn.attn_rec
            attn_rec = np.concatenate(model.attn_block.cross_attn.attn_rec,axis=0)

        #attn_rec = attn_rec.reshape(-1, *attn_rec.shape[2:])
        #attn_rec = np.squeeze(attn_rec)
    finally:
        _reset_record_attention()
    return attn_rec


def visualize_mean_std_attention_weights(
        weights: np.ndarray, 
        layers: list[str]|None=None,
        height: float=4,
        width: float=5,
        )->plt.Figure:

    mean_w = np.mean(weights, axis=0)
    std_w = np.std(weights, axis=0)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(2*width, height), sharex=True, sharey=True)
    sns.heatmap(mean_w, annot=True, ax=axes[0], cbar=False, cmap="Blues")
    sns.heatmap(std_w, annot=True, ax=axes[1], cbar=False, cmap="Oranges")
    axes[0].set_title("Mean attn. weights over all samples")
    axes[1].set_title("Std attn. weights over all samples")
    axes[0].set_ylabel("Heads")
    if layers is not None:
        for ax in axes:
            ax.set_xticks(np.arange(len(layers)) + 0.5)
            ax.set_xticklabels(layers)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    else:
        axes[0].set_xlabel("Input representations")
        axes[1].set_xlabel("Input representations")
    
    fig.tight_layout()
    return fig 



