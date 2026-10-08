from pathlib import Path
from vocabulary import build_vocabulary , convert_pairs_to_ids
from preprocessing import read_corpus, normalize_text, split_into_sentences, tokenize
def generate_center_context(tokenized_sentences, window_size):
    pairs = []
    for sentence in tokenized_sentences:
        for center_index in range(len(sentence)):
            center_word = sentence[center_index]
            start = max(0,center_index - window_size)
            end = min(len(sentence) , center_index + window_size + 1)
            for context_index in range(start, end):
                if context_index == center_index:
                    continue
                context_word = sentence[context_index]
                pairs.append((center_word, context_word))
    return pairs

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

    pairs = generate_center_context(tokenized_sentences, window_size=2)

    indexed_pairs = convert_pairs_to_ids(pairs, vocabulary)

    print(f"Number of sentences: {len(tokenized_sentences)}")
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Number of center-context pairs: {len(pairs)}")

    print("\nFirst 20 word pairs:")
    for pair in pairs[:20]:
        print(pair)

    print("\nFirst 20 ID pairs:")
    for pair in indexed_pairs[:20]:
        print(pair)