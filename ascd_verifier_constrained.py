"""Independent visual-evidence constraint for token-level ASCD decisions.

This module deliberately does not alter ASCD's positive/negative attention
branches or contrastive logit formula.  It only decides whether a proposed
ASCD token replacement should be retained at a COCO-object-like position.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ascd_sumgd import find_aligned_word, generate_pos_tags


# The CHAIR evaluator's COCO objects and common aliases, flattened to word
# tokens.  This intentionally targets benchmark-relevant object positions,
# instead of running a visual gate on syntax or arbitrary content words.
_CHAIR_OBJECT_TERMS = """
person girl boy man woman kid child chef baker people adult rider children baby
worker passenger sister biker policeman cop officer lady cowboy bride groom male
female guy traveler mother father gentleman pitcher player skier snowboarder
skater skateboarder foreigner caller offender coworker trespasser patient
politician soldier grandchild serviceman walker drinker doctor bicyclist thief
buyer teenager student camper driver solider hunter shopper villager
bicycle bike unicycle minibike trike car automobile van minivan sedan suv
hatchback cab jeep coupe taxicab limo taxi motorcycle scooter motor moped
airplane jetliner plane air monoplane aircraft jet airbus biplane seaplane bus
minibus trolley train locomotive tramway caboose truck pickup lorry hauler
firetruck boat ship liner sailboat motorboat dinghy powerboat speedboat canoe
skiff yacht kayak catamaran pontoon houseboat vessel rowboat trawler ferryboat
watercraft tugboat schooner barge ferry sailboard paddleboat lifeboat freighter
steamboat riverboat battleship steamship traffic light street signal stop fire
hydrant sign parking meter bench pew bird ostrich owl seagull goose duck
parakeet falcon robin pelican waterfowl heron hummingbird mallard finch pigeon
sparrow seabird osprey blackbird fowl shorebird woodpecker egret chickadee
quail bluebird kingfisher buzzard willet gull swan bluejay flamingo cormorant
parrot loon gosling waterbird pheasant rooster sandpiper crow raven turkey
oriole cowbird warbler magpie peacock cockatiel lorikeet puffin vulture condor
macaw peafowl cockatoo songbird cat kitten feline tabby dog puppy beagle pup
chihuahua schnauzer dachshund rottweiler canine pitbull collie pug terrier
poodle labrador doggie doberman mutt doggy spaniel bulldog sheepdog weimaraner
corgi cocker greyhound retriever brindle hound whippet husky horse colt pony
racehorse stallion equine mare foal palomino mustang clydesdale bronc bronco
sheep lamb ram goat ewe cow cattle oxen ox calf holstein heifer buffalo bull
zebu bison elephant bear panda zebra giraffe backpack knapsack umbrella handbag
wallet purse briefcase tie bow suitcase luggage frisbee skis ski snowboard
sports ball kite baseball bat glove skateboard surfboard longboard skimboard
shortboard wakeboard tennis racket bottle wine glass cup fork knife pocketknife
knive spoon bowl container banana apple sandwich burger sub cheeseburger
hamburger orange broccoli carrot hot dog pizza donut doughnut bagel cake
cheesecake cupcake shortcake coffeecake pancake chair seat stool couch sofa
recliner futon loveseat settee chesterfield potted plant houseplant bed dining
table desk toilet urinal commode lavatory potty tv monitor televison television
laptop computer notebook netbook lenovo macbook mouse remote keyboard cell
phone mobile cellphone telephone phon smartphone iphone microwave oven stovetop
stove toaster sink refrigerator fridge freezer book clock vase scissors teddy
hair drier hairdryer toothbrush
"""
CHAIR_OBJECT_TERMS = frozenset(re.findall(r"[a-z]+", _CHAIR_OBJECT_TERMS))


def _normalise_object_word(word: str) -> str:
    normalised = re.sub(r"[^a-z]", "", word.lower())
    if not normalised:
        return ""
    # WordNet is already a CHAIR dependency and is present in the frozen env.
    try:
        from nltk.stem import WordNetLemmatizer

        return WordNetLemmatizer().lemmatize(normalised, pos="n")
    except (LookupError, ImportError):
        return normalised


def terminal_chair_object(
    tokenizer, prefix_token_ids: Sequence[int], candidate_token_id: int
) -> Optional[str]:
    """Return the terminal CHAIR-object word completed/continued by a token.

    A token can be a subword.  We therefore POS-align the decoded sequence and
    inspect the word that contains the candidate token, rather than treating a
    raw tokenizer fragment as an object label.
    """
    candidate_ids = [int(value) for value in prefix_token_ids] + [
        int(candidate_token_id)
    ]
    tagged = generate_pos_tags(tokenizer, candidate_ids)
    aligned = find_aligned_word(tagged, len(candidate_ids) - 1)
    if aligned is None:
        return None
    _, _, word, pos = aligned
    if pos not in {"NOUN", "PROPN"}:
        return None
    normalised = _normalise_object_word(word)
    return normalised if normalised in CHAIR_OBJECT_TERMS else None


def clip_caption_prompt(caption: str, max_words: int = 24) -> str:
    """Keep the candidate at the end of a CLIP-safe local caption context."""
    words = caption.strip().split()
    return "a photo of " + " ".join(words[-int(max_words) :])


@dataclass(frozen=True)
class VerifierDecision:
    selected_token_id: int
    reason: str
    eligible_object_position: bool
    ascd_object_word: Optional[str]
    vanilla_object_word: Optional[str]
    ascd_score: Optional[float]
    vanilla_score: Optional[float]
    score_delta: Optional[float]
    accepts_ascd: bool


def decide_verifier_constraint(
    *,
    ascd_token_id: int,
    vanilla_token_id: int,
    ascd_object_word: Optional[str],
    vanilla_object_word: Optional[str],
    ascd_score: Optional[float],
    vanilla_score: Optional[float],
    threshold: float,
    observe_only: bool = False,
) -> VerifierDecision:
    """Apply the frozen strict ``delta > threshold`` acceptance rule."""
    if int(ascd_token_id) == int(vanilla_token_id):
        return VerifierDecision(
            int(ascd_token_id), "same_token", False, ascd_object_word,
            vanilla_object_word, None, None, None, True,
        )
    eligible = bool(ascd_object_word or vanilla_object_word)
    if not eligible:
        return VerifierDecision(
            int(ascd_token_id), "non_object_position", False, ascd_object_word,
            vanilla_object_word, None, None, None, True,
        )
    if ascd_score is None or vanilla_score is None:
        raise ValueError("eligible verifier decisions require both CLIP scores")
    delta = float(ascd_score) - float(vanilla_score)
    accepts = bool(delta > float(threshold))
    if observe_only:
        return VerifierDecision(
            int(ascd_token_id), "observe_only", True, ascd_object_word,
            vanilla_object_word, float(ascd_score), float(vanilla_score), delta,
            accepts,
        )
    return VerifierDecision(
        int(ascd_token_id) if accepts else int(vanilla_token_id),
        "accepted_ascd" if accepts else "fallback_vanilla", True,
        ascd_object_word, vanilla_object_word, float(ascd_score),
        float(vanilla_score), delta, accepts,
    )


class CLIPVisualVerifier:
    """Frozen local OpenAI CLIP image/text cosine scorer for one image."""

    def __init__(self, model_path: str, device: str = "cuda"):
        from transformers import CLIPModel, CLIPProcessor

        self.device = torch.device(device)
        self.processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = CLIPModel.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.float16
        ).eval().to(self.device)
        self.image_features: Optional[torch.Tensor] = None

    @torch.inference_mode()
    def set_image(self, image) -> None:
        encoded = self.processor(images=image, return_tensors="pt")
        pixels = encoded["pixel_values"].to(self.device, dtype=torch.float16)
        features = self.model.get_image_features(pixel_values=pixels).float()
        self.image_features = torch.nn.functional.normalize(features, dim=-1)

    @torch.inference_mode()
    def score_pair(self, ascd_caption: str, vanilla_caption: str) -> Tuple[float, float]:
        if self.image_features is None:
            raise RuntimeError("CLIP visual verifier has no image feature")
        encoded = self.processor(
            text=[clip_caption_prompt(ascd_caption), clip_caption_prompt(vanilla_caption)],
            return_tensors="pt", padding=True, truncation=True, max_length=77,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        text_features = self.model.get_text_features(**encoded).float()
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        scores = torch.matmul(self.image_features, text_features.T)[0]
        return float(scores[0].item()), float(scores[1].item())


def build_audit_record(
    *, image_id: int, step: int, ascd_token_id: int, vanilla_token_id: int,
    ascd_token: str, vanilla_token: str, decision: VerifierDecision,
) -> Dict[str, object]:
    return {
        "image_id": int(image_id),
        "step": int(step),
        "ascd_token_id": int(ascd_token_id),
        "vanilla_token_id": int(vanilla_token_id),
        "ascd_token": ascd_token,
        "vanilla_token": vanilla_token,
        "selected_token_id": int(decision.selected_token_id),
        "reason": decision.reason,
        "eligible_object_position": bool(decision.eligible_object_position),
        "ascd_object_word": decision.ascd_object_word,
        "vanilla_object_word": decision.vanilla_object_word,
        "ascd_clip_score": decision.ascd_score,
        "vanilla_clip_score": decision.vanilla_score,
        "clip_score_delta": decision.score_delta,
        "accepts_ascd": bool(decision.accepts_ascd),
    }
