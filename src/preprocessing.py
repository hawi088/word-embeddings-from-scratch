from pathlib import Path
from vocabulary import build_vocabulary


def read_corpus(file_path):
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as file:
        text = file.read()

    return text


def normalize_text(text):
    text = text.lower()
    return text


def split_into_sentences(text):
    sentences = text.split(".")
    sentences = [sentence.strip() for sentence in sentences]
    sentences = [sentence for sentence in sentences if sentence]

    return sentences


def tokenize(sentence):
    punctuation = ",.!?;:"

    for character in punctuation:
        sentence = sentence.replace(character, "")

    tokens = sentence.split()

    return tokens


if __name__ == "__main__":
    tokenized_sentences = []

    corpus_path = Path(__file__).parent.parent / "corpus" / "corpus.txt"

    text = read_corpus(corpus_path)
    text = normalize_text(text)
    sentences = split_into_sentences(text)

    for sentence in sentences:
        tokens = tokenize(sentence)
        tokenized_sentences.append(tokens)

    vocabulary = build_vocabulary(tokenized_sentences)

    total_tokens = 0

    for sentence in tokenized_sentences:
        total_tokens += len(sentence)

    print(f"Total tokens: {total_tokens}")
    print(f"Vocabulary size: {len(vocabulary)}")
    print(vocabulary)
