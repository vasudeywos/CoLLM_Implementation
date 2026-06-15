import random


# Paper Section 7.1: all 15 templates
# w_i = target caption, w_j = neighbor caption
MODIFICATION_TEMPLATES = [
    "show {wi} instead of {wj}",
    "{wi} instead of {wj}",
    "show {wi} rather than {wj}",
    "{wi} rather than {wj}",
    "rather than {wj}, show {wi}",
    "rather than {wj}, {wi}",
    "instead of {wj}, {wi}",
    "{wj}, changed to {wi}",
    "not {wj}, but {wi}",
    "show {wi}, not {wj}",
    "{wj} is missing, {wi}",
    "{wi}, and {wj} is missing",
    "remove {wj}, add {wi}",
    "add {wi}, remove {wj}",
    "{wj} become {wi}",
]


def synthesize_modification_text(
    wi: str,
    wj: str,
) -> str:
    """
    Synthesize modification text from target caption wi and neighbor caption wj.
    
    Paper Section 7.1: randomly select one of 15 templates.
    Template describes change FROM neighbor (reference) TO target.
    
    Args:
        wi: caption of target image     (what we want)
        wj: caption of neighbor image   (what we have as reference)
    Returns:
        w*_i: synthesized modification text
    """
    template = random.choice(MODIFICATION_TEMPLATES)
    return template.format(wi=wi, wj=wj)


def get_modification_texts(
    captions_i: list,
    captions_j: list,
    synthesis_ratio: float = 0.75,
) -> list:
    """
    Generate modification texts for a batch.
    
    Paper Figure 7b ablation: text synthesis applied to 75% of samples.
    For remaining 25%: w*_i = w_i (just the target caption).
    
    Args:
        captions_i: list of target captions w_i     length B
        captions_j: list of neighbor captions w_j   length B
        synthesis_ratio: fraction to apply template (default 0.75)
    Returns:
        list of modification texts w*_i              length B
    """
    mod_texts = []
    for wi, wj in zip(captions_i, captions_j):
        if random.random() < synthesis_ratio:
            # Apply template synthesis
            mod_texts.append(synthesize_modification_text(wi, wj))
        else:
            # Use target caption directly
            mod_texts.append(wi)
    return mod_texts