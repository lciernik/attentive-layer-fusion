import hashlib
import re

model_id_to_abbrev_mapping = {
    # DINOv2 models
    "dinov2-vit-large-p14": "dv2L",
    "dinov2-vit-small-p14": "dv2S",
    "dinov2-vit-base-p14": "dv2B",
    "dinov2-vit-giant-p14": "dv2G",
    # DINO models
    "dino-vit-base-p16": "dinB",
    "dino-rn50": "dinr50",
    "dino-xcit-small-12-p16": "dinxs",
    "dino-xcit-medium-24-p16": "dinxm",
    "dino-vit-small-p16": "dinsm",
    # MAE models
    "mae-vit-base-p16": "maebs",
    "mae-vit-large-p16": "maelg",
    "mae-vit-huge-p14": "maehg",
    "mae-vit-large-p16_avg_pool": "maelp",
    # OpenCLIP models
    "OpenCLIP_RN50_openai": "oclr5",
    "OpenCLIP_ViT-B-32_openai": "oclb2",
    "OpenCLIP_ViT-B-16_openai": "oclb1",
    "OpenCLIP_ViT-B-16_laion400m_e32": "oclb2",
    "OpenCLIP_ViT-B-16_laion2b_s34b_b88k": "oclb3",
    "OpenCLIP_ViT-L-14_openai": "ocll1",
    "OpenCLIP_ViT-L-14_laion400m_e32": "ocll2",
    "OpenCLIP_ViT-L-14_laion2b_s32b_b82k": "ocll3",
    "OpenCLIP_ViT-B-16-SigLIP_webli": "oclsg",
    "OpenCLIP_EVA01-g-14_laion400m_s11b_b41k": "ocle1",
    "OpenCLIP_EVA01-g-14-plus_merged2b_s11b_b114k": "ocle2",
    "OpenCLIP_EVA02-L-14_merged2b_s4b_b131k": "ocle3",
    "OpenCLIP_EVA02-B-16_merged2b_s8b_b131k": "ocle4",
    # VGG models
    "vgg16": "vgg16",
    "vgg19": "vgg19",
    # ResNet models
    "resnet50": "rn50",
    "resnet152": "rn152",
    "resnext50_32x4d": "rnx50",
    "seresnet50": "sern5",
    # ConvNext models
    "convnext_base": "cnxbs",
    "convnext_large": "cnxlg",
    # EfficientNet models
    "efficientnet_b3": "effb3",
    "efficientnet_b4": "effb4",
    "efficientnet_b5": "effb5",
    "efficientnet_b6": "effb6",
    "efficientnet_b7": "effb7",
    # Self-supervised ResNet models
    "simclr-rn50": "simcl",
    "mocov2-rn50": "mocv2",
    "vicreg-rn50": "vicrg",
    "barlowtwins-rn50": "barlw",
    "pirl-rn50": "pirl5",
    "jigsaw-rn50": "jigs5",
    "rotnet-rn50": "rot50",
    "swav-rn50": "swav5",
    # Vision Transformer models
    "vit_small_patch16_224": "vits1",
    "vit_small_patch16_224.augreg_in1k": "vits2",
    "vit_base_patch16_224": "vitb1",
    "vit_base_patch16_224.augreg_in21k": "vitb2",
    "vit_large_patch16_224": "vitl1",
    "vit_large_patch16_224.augreg_in21k": "vitl2",
    "vit_huge_patch14_224.orig_in21k": "vith1",
    "vit_huge_patch14_clip_224.laion2b": "vith2",
    "vit_large_patch16_224_avg_pool": "vitlp",
    # BEIT models
    "beit_base_patch16_224": "beitb",
    "beit_base_patch16_224.in22k_ft_in22k": "beitf",
    "beit_large_patch16_224": "beitl",
    "beit_large_patch16_224.in22k_ft_in22k": "beitg",
    # DEIT models
    "deit3_base_patch16_224": "deit3",
    "deit3_base_patch16_224.fb_in22k_ft_in1k": "deit4",
    "deit3_large_patch16_224": "deit5",
    "deit3_large_patch16_224.fb_in22k_ft_in1k": "deit6",
    # Swin Transformer models
    "swin_base_patch4_window7_224": "swinb",
    "swin_base_patch4_window7_224.ms_in22k": "swinc",
    "swin_large_patch4_window7_224": "swinl",
    "swin_large_patch4_window7_224.ms_in22k": "swinm",
    # Other models
    "Kakaobrain_Align": "kakao",
}


for key, value in model_id_to_abbrev_mapping.copy().items():
    model_id_to_abbrev_mapping[key + "_cls"] = value + "_cls"
    model_id_to_abbrev_mapping[key + "_ap"] = value + "_ap"
    model_id_to_abbrev_mapping[key + "_at"] = value + "_at"

model_abbrev_to_id_mapping = {v: k for k, v in model_id_to_abbrev_mapping.items()}


def module_shortener(module: str) -> str:
    """Shorten the module name to make it more concise."""
    if module in ["visual", "norm"]:
        return "last"
    else:
        r"""
        >>> modules = [
        ...     "blocks.10.norm2",
        ...     "visual.transformer.resblocks.11.ln_1",
        ...     "visual.trunk.blocks.10.norm2",
        ... ]
        >>>['10.2', '11.1', '10.2']
        """
        match = re.search(r"\.(\d+)\.(?:norm|ln_)([12])$", module)
        if match:
            return f"{match.group(1)}.{match.group(2)}"
        else:
            raise ValueError(f"Module {module} is not a valid module name.")


def get_hash_from_model_id(model_ids: list[str]) -> str:
    """Returns a hash of the model ID.
    1. For the same model ID, return the model ID.
    2. For the same base model, but with both ap and cls, return the base model ID with "_both" suffix.
    3. For different model IDs, return a hash of the sorted model IDs.
    """

    tmp_model_ids = [mid.split("@")[0] for mid in model_ids]
    if all(m_id == tmp_model_ids[0] for m_id in tmp_model_ids):
        # All model IDs are the same
        return tmp_model_ids[0]
    elif check_same_base_model(tmp_model_ids, tmp_model_ids[0]):
        # All model IDs are the same base model, but with both ap and cls
        return tmp_model_ids[0].replace("_ap", "").replace("_cls", "") + "_both"
    else:
        # Not all model IDs are the same, create a hash
        sorted_model_ids = sorted(model_ids)
        return f"modelhash_{hashlib.sha256(' '.join(sorted_model_ids).encode('utf-8')).hexdigest()}"


def check_same_base_model(model_ids: list[str], base_model) -> bool:
    """Check if all model IDs in the list are the same base model."""
    return all(
        m_id.split("@", 1)[0].replace("_ap", "").replace("_cls", "")
        == base_model.split("@", 1)[0].replace("_ap", "").replace("_cls", "")
        for m_id in model_ids
    )


def split_model_module(
    model_id: str, use_abbrev: bool = True, always_return_tuple: bool = False
) -> str | tuple[str, str | None]:
    mid_module_name = model_id.split("@")
    if len(mid_module_name) == 1:
        mid = model_id_to_abbrev_mapping[mid_module_name[0]] if use_abbrev else mid_module_name[0]
        return (mid, None) if always_return_tuple else (mid,)
    elif len(mid_module_name) > 2:
        raise ValueError(
            f"Model ID {model_id} has more than one '@' symbol, which is not allowed. "
            f"Either enter model_id or model_id@module_name."
        )
    else:
        mid = model_id_to_abbrev_mapping[mid_module_name[0]] if use_abbrev else mid_module_name[0]
        return (mid, mid_module_name[1])


def compress_consecutive_sequences(suffixes):
    """Compress consecutive numerical sequences in suffixes.
    E.g., ['1.2', '2.2', '3.2', '4.2'] -> ['1.2_to_4.2']
    """
    # Separate numeric and non-numeric suffixes
    numeric_suffixes = []
    non_numeric_suffixes = []

    for suffix in suffixes:
        # Check if suffix matches pattern like "N.M" where N and M are numbers
        match = re.match(r"^(\d+)\.(\d+)$", suffix)
        if match:
            layer_num = int(match.group(1))
            block_num = int(match.group(2))
            numeric_suffixes.append((layer_num, block_num, suffix))
        else:
            non_numeric_suffixes.append(suffix)

    # Sort numeric suffixes by layer number, then block number
    numeric_suffixes.sort(key=lambda x: (x[0], x[1]))

    # Find consecutive sequences
    compressed = []
    if numeric_suffixes:
        current_sequence = [numeric_suffixes[0]]

        for i in range(1, len(numeric_suffixes)):
            prev_block, prev_layer_norm, _ = current_sequence[-1]
            curr_block, curr_layer_norm, curr_suffix = numeric_suffixes[i]

            if curr_layer_norm == prev_layer_norm and curr_block == prev_block + 1:
                current_sequence.append(numeric_suffixes[i])
            else:
                if len(current_sequence) >= 3:  # Only compress if 3+ consecutive
                    first_suffix = current_sequence[0][2]
                    last_suffix = current_sequence[-1][2]
                    compressed.append(f"{first_suffix}_to_{last_suffix}")
                else:
                    for _, _, suffix in current_sequence:
                        compressed.append(suffix)

                current_sequence = [numeric_suffixes[i]]

        if len(current_sequence) >= 3:
            first_suffix = current_sequence[0][2]
            last_suffix = current_sequence[-1][2]
            compressed.append(f"{first_suffix}_to_{last_suffix}")
        else:
            for _, _, suffix in current_sequence:
                compressed.append(suffix)

    compressed.extend(non_numeric_suffixes)

    return compressed
