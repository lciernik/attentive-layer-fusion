import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np
import torch.nn.functional
from loguru import logger
from src.utils.loss_utils import CKATorch
from src.utils.model_mapping import get_hash_from_model_id
from thingsvision.core.rsa import compute_rdm, correlate_rdms
from tqdm import tqdm

from src.utils.utils import check_models, load_features


class BaseModelSimilarity:
    def __init__(
        self,
        feature_root: str,
        subset_root: Optional[str],
        split: str = "train",
        device: str = "cuda",
        num_workers: int = 4,
        normalize: bool = True,
    ) -> None:
        self.feature_root = feature_root
        self.split = split
        self.device = device
        self.model_ids = []
        self.num_workers = num_workers
        self.subset_indices = self._load_subset_indices(subset_root)
        self.name = "Base"
        self.normalize = normalize

    def _load_subset_indices(self, subset_root) -> Optional[List[int]]:
        subset_path = os.path.join(subset_root, f"subset_indices_{self.split}.json")
        if not os.path.exists(subset_path):
            warnings.warn(
                f"Subset indices not found at {subset_path}. Continuing with full datasets."
            )
            return None
        with open(subset_path, "r") as f:
            subset_indices = json.load(f)
        return subset_indices

    def load_model_ids(self, model_ids: List[str]) -> None:
        assert os.path.exists(self.feature_root), "Feature root path non-existent"
        self.model_ids = check_models(self.feature_root, model_ids, self.split)
        self.model_ids_with_idx = [
            (i, model_id) for i, model_id in enumerate(self.model_ids)
        ]

    def _prepare_sim_matrix(self) -> np.ndarray:
        return np.ones((len(self.model_ids_with_idx), len(self.model_ids_with_idx)))

    def _load_feature(self, model_id: str) -> np.ndarray:
        raise NotImplementedError()

    def _compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        raise NotImplementedError()

    def compute_similarity_matrix(self, preload_feat_thresh:int|None=100) -> np.ndarray:
        sim_matrix = self._prepare_sim_matrix()
        num_workers = self.num_workers
        if len(self.model_ids_with_idx) < preload_feat_thresh:
            preloaded_features = {model: self._load_feature(model_id=model) for _, model in self.model_ids_with_idx}
        else:
            preloaded_features = {}
        for idx1, model1 in tqdm(
            self.model_ids_with_idx, desc=f"Computing {self.name} matrix"
        ):
            features_1 = preloaded_features[model1] if model1 in preloaded_features else self._load_feature(model_id=model1)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {}
                for idx2, model2 in self.model_ids_with_idx:
                    if idx1 < idx2:

                        future = executor.submit(
                            self._compute_similarity, features_1,
                            preloaded_features[model2] if model2 in preloaded_features else self._load_feature(model_id=model2)
                        )
                        futures[future] = (idx1, idx2)

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Pairwise similarity computation",
                ):
                    cidx1, cidx2 = futures[future]
                    rho = future.result()
                    sim_matrix[cidx1, cidx2] = rho
        upper_tri = np.triu(sim_matrix)
        sim_matrix = upper_tri + upper_tri.T - np.diag(np.diag(sim_matrix))
        return sim_matrix

    def get_model_ids(self) -> List[str]:
        return self.model_ids


class CKAModelSimilarity(BaseModelSimilarity):
    def __init__(
        self,
        feature_root: str,
        subset_root: Optional[str],
        split: str = "train",
        device: str = "cuda",
        kernel: str = "linear",
        backend: str = "torch",
        unbiased: bool = True,
        sigma: Optional[float] = None,
        num_workers: int = 4,
        normalize: bool =True,
    ) -> None:
        super().__init__(
            feature_root=feature_root,
            subset_root=subset_root,
            split=split,
            device=device,
            num_workers=num_workers,
            normalize=normalize,
        )
        self.kernel = kernel
        self.backend = backend
        self.unbiased = unbiased
        self.sigma = sigma


    # def _load_feature(self, model_id:str) -> np.ndarray:
    #     features = load_features(self.feature_root, model_id, self.split, self.subset_indices)
    #     return features

    # def _compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
    #     m = feat1.shape[0]
    #     cka = get_cka(backend=self.backend, m=m, kernel=self.kernel, unbiased=self.unbiased, device=self.device,
    #                     sigma=self.sigma)
    #     rho = cka.compare(X=feat1, Y=feat2)
    #     return rho

    def compute_similarity_matrix(self, preload_feat_thresh:int|None = 100) -> np.ndarray:
        sim_matrix = self._prepare_sim_matrix()
        # We can store all features (up to 100 models) in memory, so we don't need to load them multiple times
        if len(self.model_ids_with_idx) < preload_feat_thresh:
            preloaded_features = {model: load_features(
                            self.feature_root, model, self.split, self.subset_indices, self.normalize
                            ) for _, model in self.model_ids_with_idx}
        else:
            preloaded_features = {}
        for idx1, model1 in tqdm(self.model_ids_with_idx, desc="Computing CKA matrix"):
            features_i = (preloaded_features[model1] if model1 in preloaded_features else load_features(
                self.feature_root, model1, self.split, self.subset_indices, self.normalize
            ))
            m = features_i.shape[0]
            cka = CKATorch(m=m, kernel=self.kernel, unbiased=self.unbiased,
                           device=self.device, sigma=self.sigma, compile=False)

            for idx2, model2 in self.model_ids_with_idx:
                if idx1 >= idx2:
                    continue
                features_j = (preloaded_features[model2] if model2 in preloaded_features else load_features(
                            self.feature_root, model2, self.split, self.subset_indices, self.normalize
                ))
                assert features_i.shape[0] == features_j.shape[0], (
                    f"Number of features should be equal for CKA computation. (model1: {model1}, model2: {model2})"
                )


                rho = cka.compare(X=features_i, Y=features_j)
                sim_matrix[idx1, idx2] = rho
                sim_matrix[idx2, idx1] = rho

        return sim_matrix

    def get_name(self) -> str:
        method_name = (
            f"cka_kernel_{self.kernel}{'_unbiased' if self.unbiased else '_biased'}"
        )
        if self.kernel == "rbf":
            method_name += f"_sigma_{self.sigma}"
        return method_name

class CosineModelSimilarity(BaseModelSimilarity):
    def __init__(
        self,
        feature_root: str,
        subset_root: Optional[str],
        split: str = "train",
        device: str = "cuda",
        num_workers: int = 4,
        normalize: bool =True,
    ) -> None:
        super().__init__(
            feature_root=feature_root,
            subset_root=subset_root,
            split=split,
            device=device,
            num_workers=num_workers,
            normalize=normalize,
        )


    def _load_feature(self, model_id:str) -> np.ndarray:
        # TODO typing enforces the casting of a torch Tensor to numpy just to cast it back to torch in the next step...
        features = load_features(self.feature_root, model_id, self.split, self.subset_indices, self.normalize).numpy()
        return features

    def _compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        assert feat1.shape[0] == feat2.shape[0], (
            "Number of features should be equal for similarity computation."
        )
        if feat1.shape[1] != feat2.shape[1]:
            logger.warning(f"The feature dimensionality of the two models is different: {feat1.shape[1]} vs {feat2.shape[1]}. "
            )
            return np.nan
        return torch.nn.functional.cosine_similarity(torch.Tensor(feat1), torch.Tensor(feat2),dim=1).mean().item()

    def get_name(self) -> str:
        return f"cosine_similarity"

class RSAModelSimilarity(BaseModelSimilarity):
    def __init__(
        self,
        feature_root: str,
        subset_root: Optional[str],
        split: str = "train",
        device: str = "cuda",
        rsa_method: str = "correlation",
        corr_method: str = "spearman",
        num_workers: int = 4,
        normalize: bool = True,
    ) -> None:
        super().__init__(
            feature_root=feature_root,
            subset_root=subset_root,
            split=split,
            device=device,
            num_workers=num_workers,
            normalize=normalize,
        )
        self.rsa_method = rsa_method
        self.corr_method = corr_method
        self.name = "RSA"

    def _load_feature(self, model_id: str) -> np.ndarray:
        features = load_features(
            self.feature_root, model_id, self.split, self.subset_indices, self.normalize
        ).numpy()
        rdm_features = compute_rdm(features, method=self.rsa_method)
        return rdm_features

    def _compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        assert feat1.shape[0] == feat2.shape[0], (
            "Number of features should be equal for similarity computation."
        )

        return correlate_rdms(feat1, feat2, correlation=self.corr_method)

    def get_name(self):
        if self.rsa_method == "correlation":
            return f"rsa_method_{self.rsa_method}_corr_method_{self.corr_method}"
        else:
            return f"rsa_method_{self.rsa_method}"


def compute_sim_matrix(
    sim_method: str,
    feature_root: str,
    model_ids: List[str],
    split: str,
    subset_root: Optional[str] = None,
    kernel: str = "linear",
    rsa_method: str = "correlation",
    corr_method: str = "spearman",
    backend: str = "torch",
    unbiased: bool = True,
    device: str = "cuda",
    sigma: Optional[float] = None,
    num_workers: int = 4,
    normalize: bool = True,
    save_base_path: str | Path | None = None,
) -> Tuple[np.ndarray, List[str], str]:
    model_similarity = get_metric(
        sim_method=sim_method,
        feature_root=feature_root,
        split=split,
        subset_root=subset_root,
        kernel=kernel,
        rsa_method=rsa_method,
        corr_method=corr_method,
        backend=backend,
        unbiased=unbiased,
        device=device,
        sigma=sigma,
        num_workers=num_workers,
        normalize=normalize,
    )
    model_similarity.load_model_ids(model_ids)
    model_ids = model_similarity.get_model_ids()
    method_slug = model_similarity.get_name()
    
    if save_base_path is not None:
        model_ids_hash = get_hash_from_model_id(model_ids)
        out_path = save_base_path / method_slug / model_ids_hash
        out_res = out_path / "similarity_matrix.pt"
        out_res_model_ids = out_path / "model_ids.txt"

        if out_res.exists() and out_res_model_ids.exists():
            logger.info(f"\nSimilarity matrix already exists at {out_res}. Skipping computation ...\n")
            with open(out_res_model_ids, "r") as file:
                lines = file.readlines()
            lines = [line.strip() for line in lines]
            assert all([id1 == id2 for id1, id2 in zip(lines, model_ids)]), (
                "Model ids in the saved file do not match the provided model ids."
            )
            sim_mat = torch.load(out_res)
            return sim_mat, model_ids, method_slug

    logger.info("Compute similarity matrix")
    sim_mat = model_similarity.compute_similarity_matrix()
    if save_base_path is not None:

        if not out_path.exists():
            os.makedirs(out_path, exist_ok=True)
            logger.info(f"\nCreated path ({out_path}), where results are to be stored ...\n")

        logger.info(f"\nDump {sim_method.upper()} matrix to: {out_res}\n")
        torch.save(sim_mat, out_res)

        with open(out_res_model_ids, "w") as file:
            for string in model_ids:
                file.write(string + "\n")
    return sim_mat, model_ids, method_slug


def get_metric(
    sim_method: str,
    feature_root: str,
    split: str,
    subset_root: Optional[str] = None,
    kernel: str = "linear",
    rsa_method: str = "correlation",
    corr_method: str = "spearman",
    backend: str = "torch",
    unbiased: bool = True,
    device: str = "cuda",
    sigma: Optional[float] = None,
    num_workers: int = 4,
    normalize: bool = True,
):
    if sim_method == "cka":
        model_similarity = CKAModelSimilarity(
            feature_root=feature_root,
            subset_root=subset_root,
            split=split,
            device=device,
            kernel=kernel,
            backend=backend,
            unbiased=unbiased,
            sigma=sigma,
            num_workers=num_workers,
            normalize=normalize,
        )
    elif sim_method == "rsa":
        model_similarity = RSAModelSimilarity(
            feature_root=feature_root,
            subset_root=subset_root,
            split=split,
            device=device,
            rsa_method=rsa_method,
            corr_method=corr_method,
            num_workers=num_workers,
            normalize=normalize,
        )
    elif sim_method == "cosine":
        model_similarity = CosineModelSimilarity(
            feature_root=feature_root,
            subset_root=subset_root,
            split=split,
            device=device,
            num_workers=num_workers,
            normalize=normalize,
        )
    else:
        raise ValueError(f"Unknown similarity method: {sim_method}")

    return model_similarity
