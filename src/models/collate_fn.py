import torch
from torch.utils.data import DataLoader


def images_only_collate(batch: list[tuple[torch.Tensor, ...]]) -> torch.Tensor:
    """Collate function to return only the images from the batch.
    This is useful when we want to use the dataloader for feature extraction with thingsvision.
    """
    return torch.stack([item[0] for item in batch])


class CollateContextManager:
    """Context manager to replace the collate function of a dataloader with images only collate.
    This is useful when we want to use the dataloader for feature extraction with thingsvision.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self.dataloader = dataloader
        self.new_collate_fn = images_only_collate
        self.original_collate_fn = None
        self.should_replace = False

    def _check_dataloader_format(self) -> bool:
        sample_batch = next(iter(self.dataloader))
        return isinstance(sample_batch, (tuple, list)) and len(sample_batch) == 2

    def __enter__(self) -> DataLoader:
        """Enter the context manager and replace the collate function with images only collate,
        if the dataloader is in the correct format.
        """
        self.should_replace = self._check_dataloader_format()
        if self.should_replace:
            self.original_collate_fn = self.dataloader.collate_fn
            self.dataloader.collate_fn = self.new_collate_fn
        return self.dataloader

    def __exit__(self, *args) -> None:
        """Exit the context manager and restore the original collate function,
        if the dataloader was not in the correct format.
        """
        if self.should_replace and self.original_collate_fn is not None:
            self.dataloader.collate_fn = self.original_collate_fn
