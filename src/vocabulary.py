def build_vocabulary(tokenized_sentences):
    vocabulary = []
    for sentence in tokenized_sentences:
        for word in sentence:
            if word not in vocabulary:
                vocabulary.append(word)
    return vocabulary