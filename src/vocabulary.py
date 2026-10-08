def build_vocabulary(tokenized_sentences):
    vocabulary = []
    for sentence in tokenized_sentences:
        for word in sentence:
            if word not in vocabulary:
                vocabulary.append(word)
    return vocabulary
def find_word_id(vocabulary, word):
    for i in range(len(vocabulary)):
        if vocabulary[i] == word:
            return i

    return -1
def convert_pairs_to_ids(pairs, vocabulary):
    indexed_pairs = []

    for center_word, context_word in pairs:
        center_id = find_word_id(vocabulary, center_word)
        context_id = find_word_id(vocabulary, context_word)

        indexed_pairs.append((center_id, context_id))

    return indexed_pairs
def get_word_embedding(word, vocabulary, E):
    word_id = find_word_id(vocabulary, word)

    if word_id == -1:
        return None

    return E[word_id]