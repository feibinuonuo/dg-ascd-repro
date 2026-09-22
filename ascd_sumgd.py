"""Official SumGD-S token/POS alignment helpers."""

from functools import lru_cache


IMAGE_POS = frozenset(("NOUN", "ADJ", "NUM", "PROPN"))


def process_tokens(tokens):
    text = ""
    for token in tokens:
        if token.startswith("▁"):
            text += " " + token[1:]
        elif token in ("<0x0A>", "</s>", "<s>", "<unk>"):
            continue
        else:
            text += token
    return text.strip()


def align_tokens(model_tokens, words):
    """Match the released SumGD ``align_probabilities`` routine."""
    model_index = 0
    indices = []
    carried_word = ""
    for word in words:
        current = []
        combined = ""
        while model_index < len(model_tokens):
            token = (
                model_tokens[model_index]
                .replace("▁", "")
                .replace("<0x0A>", "")
                .replace("</s>", "")
                .replace("<unk>", "")
                .replace("<s>", "")
            )
            if token == "":
                model_index += 1
                continue
            combined += token
            carried_word += word
            current.append(model_index)
            model_index += 1
            if combined == word or combined == carried_word:
                carried_word = ""
                break
            if len(combined) > len(word):
                model_index -= 1
                break
        indices.append(current)
    return indices


@lru_cache(maxsize=1)
def _english_pipeline():
    import spacy

    return spacy.load("en_core_web_sm")


def generate_pos_tags(tokenizer, token_ids):
    model_tokens = tokenizer.convert_ids_to_tokens([int(value) for value in token_ids])
    document = _english_pipeline()(process_tokens(model_tokens))
    words = [token.text for token in document]
    alignments = align_tokens(model_tokens, words)
    return [
        (alignment, token.text, token.pos_)
        for alignment, token in zip(alignments, document)
    ]


def find_aligned_word(tagged_words, token_index):
    for word_index, (indices, word, pos) in enumerate(tagged_words):
        if int(token_index) in indices:
            return word_index, indices, word, pos
    return None
