import json
import math
import random
from pathlib import Path

from preprocessing import (
    read_corpus,
    normalize_text,
    split_into_sentences,
    tokenize,
)
from vocabulary import (
    build_vocabulary,
    convert_pairs_to_ids,
    find_word_id,
    get_word_embedding, 
)
from pairs import generate_center_context

SEED = 42;  
TOLERANCE = 0.00001
def initialize_parameters(vocab_size, embedding_dim, seed=SEED):
    random.seed(seed)

    E = []
    for i in range(vocab_size):
        row = []
        for j in range(embedding_dim):
            row.append(random.uniform(-0.1, 0.1))
        E.append(row)

    U = []
    for i in range(embedding_dim):
        row = []
        for j in range(vocab_size):
            row.append(random.uniform(-0.1, 0.1))
        U.append(row)

    return E, U


def dot_product(vector_a, vector_b):
    result = 0.0
    for i in range(len(vector_a)):
        result += vector_a[i] * vector_b[i]
    return result


def calculate_scores(h, U):
    vocab_size = len(U[0]) #the horizontal length of U is the vocab size
    scores = []

    for j in range(vocab_size): #4 
        score = 0.0
        for k in range(len(h)): #the horizontal length of h is the embedding diamenion 
             
            score += h[k] * U[k][j]
        scores.append(score)

    return scores

def softmax(scores):
    m = max(scores)

    a = []
    for score in scores:
        a.append(math.exp(score - m))

    total = sum(a)

    probabilities = []
    for value in a:
        probabilities.append(value / total)

    return probabilities, m, total


def cross_entropy_loss(scores, target_id):
    probabilities, m, total = softmax(scores)
    loss = (m - scores[target_id]) + math.log(total)
    return loss, probabilities

def calculate_score_gradients(probabilities, target_id):
    gradients = list(probabilities)
    gradients[target_id] -= 1.0
    return gradients


def calculate_U_gradients(h, score_gradients): #score gradient are the results that we get fron the previous function 
    gradients = []
    for k in range(len(h)):
        row = []
        for j in range(len(score_gradients)):
            row.append(h[k] * score_gradients[j])
        gradients.append(row)
    return gradients


def calculate_E_gradients(U, score_gradients):
    gradients = []
    for k in range(len(U)):
        total = 0.0
        for j in range(len(score_gradients)):
            total += U[k][j] * score_gradients[j]
        gradients.append(total)
    return gradients


def update_parameters(E, U, center_id, E_gradient, U_gradients, learning_rate):
    for k in range(len(E[center_id])):
        E[center_id][k] -= learning_rate * E_gradient[k]

    for k in range(len(U)):
        for j in range(len(U[k])):
            U[k][j] -= learning_rate * U_gradients[k][j]


def train_one_pair(E, U, center_id, target_id, learning_rate):
    h = list(E[center_id])  # copy of the old embedding row

    scores = calculate_scores(h, U)
    loss, probabilities = cross_entropy_loss(scores, target_id)
    score_gradients = calculate_score_gradients(probabilities, target_id)

    # Both gradients are computed from the OLD E and U, before any update.
    U_gradients = calculate_U_gradients(h, score_gradients)
    E_gradient = calculate_E_gradients(U, score_gradients)

    update_parameters(E, U, center_id, E_gradient, U_gradients, learning_rate)

    return loss

def compute_dataset_loss(E, U, indexed_pairs):
    total_loss = 0.0

    for center_id, target_id in indexed_pairs:
        scores = calculate_scores(E[center_id], U)
        loss, _ = cross_entropy_loss(scores, target_id)
        total_loss += loss

    return total_loss / len(indexed_pairs)


def train_one_epoch(E, U, indexed_pairs, learning_rate):
    
    shuffled_pairs = list(indexed_pairs)
    random.shuffle(shuffled_pairs)

    for center_id, target_id in shuffled_pairs:
        train_one_pair(E, U, center_id, target_id, learning_rate)


def train(E, U, indexed_pairs, learning_rate, num_epochs):
    loss_history = []

    initial_loss = compute_dataset_loss(E, U, indexed_pairs)
    loss_history.append(initial_loss)
    print(f"Before training - mean loss: {initial_loss:.6f}")

    for epoch in range(num_epochs):
        train_one_epoch(E, U, indexed_pairs, learning_rate)
        epoch_loss = compute_dataset_loss(E, U, indexed_pairs)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs} - mean loss: {epoch_loss:.6f}")

    return loss_history

def cosine_similarity(vector_a, vector_b):
    dot = dot_product(vector_a, vector_b)

    magnitude_a = math.sqrt(sum(v * v for v in vector_a))
    magnitude_b = math.sqrt(sum(v * v for v in vector_b))

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    return dot / (magnitude_a * magnitude_b)


def find_nearest_neighbors(word, vocabulary, E, top_n):
    
    word_id = find_word_id(vocabulary, word)

    if word_id == -1:
        return None

    word_embedding = E[word_id]
    similarities = []

    for i in range(len(vocabulary)):
        if i == word_id:  # exclude the query itself
            continue
        similarity = cosine_similarity(word_embedding, E[i])
        similarities.append((vocabulary[i], similarity))

    similarities.sort(key=lambda item: item[1], reverse=True)

    return similarities[:top_n]


def print_nearest_neighbors(words, vocabulary, E, top_n=3):
    for word in words:
        neighbors = find_nearest_neighbors(word, vocabulary, E, top_n)

        if neighbors is None:
            print(f"'{word}' is not in the vocabulary.")
            continue

        print(f"Nearest neighbors of '{word}':")
        for neighbor, similarity in neighbors:
            print(f"  {neighbor}: {similarity:.4f}")


def evaluate_word_pairs(evaluation_pairs, vocabulary, E):
    results = []

    for word_a, word_b in evaluation_pairs:
        embedding_a = get_word_embedding(word_a, vocabulary, E)
        embedding_b = get_word_embedding(word_b, vocabulary, E)

        if embedding_a is None or embedding_b is None:
            print(f"Skipped ({word_a}, {word_b}): word not in vocabulary.")
            continue

        similarity = cosine_similarity(embedding_a, embedding_b)
        results.append((word_a, word_b, similarity))

    return results


def average_similarity(results):
    if len(results) == 0:
        return None
    return sum(similarity for _, _, similarity in results) / len(results)

def save_model(path, vocabulary, E, U):
    data = {"vocabulary": vocabulary, "E": E, "U": U}
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False)


def load_model(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data["vocabulary"], data["E"], data["U"]

def numerical_gradient_check(E, U, center_id, target_id, epsilon=0.00001):
    def pair_loss():
        scores = calculate_scores(E[center_id], U)
        loss, _ = cross_entropy_loss(scores, target_id)
        return loss

    h = list(E[center_id])
    scores = calculate_scores(h, U)
    _, probabilities = cross_entropy_loss(scores, target_id)
    e = calculate_score_gradients(probabilities, target_id)
    analytic_U = calculate_U_gradients(h, e)
    analytic_h = calculate_E_gradients(U, e)

    max_difference = 0.0

    for k in range(len(h)):
        original = E[center_id][k]
        E[center_id][k] = original + epsilon
        loss_plus = pair_loss()
        E[center_id][k] = original - epsilon
        loss_minus = pair_loss()
        E[center_id][k] = original
        numeric = (loss_plus - loss_minus) / (2 * epsilon)
        max_difference = max(max_difference, abs(numeric - analytic_h[k]))

    for k in range(len(U)):
        for j in range(len(U[0])):
            original = U[k][j]
            U[k][j] = original + epsilon
            loss_plus = pair_loss()
            U[k][j] = original - epsilon
            loss_minus = pair_loss()
            U[k][j] = original
            numeric = (loss_plus - loss_minus) / (2 * epsilon)
            max_difference = max(max_difference, abs(numeric - analytic_U[k][j]))

    return max_difference

def close(a, b, tolerance=TOLERANCE):
    return abs(a - b) <= tolerance


def lists_close(list_a, list_b, tolerance=TOLERANCE):
    if len(list_a) != len(list_b):
        return False
    for a, b in zip(list_a, list_b):
        if not close(a, b, tolerance):
            return False
    return True


def report(name, passed):
    print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    return passed


def make_tiny_model():
    vocabulary = ["cats", "eat", "food"]
    E = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    U = [[0.0, 1.0, -1.0], [1.0, 0.0, 1.0]]
    return vocabulary, E, U


def run_checks():
    results = []

    # 1. Context window
    pairs = generate_center_context([["a", "b", "c", "d"]], 1)
    expected = [("a", "b"), ("b", "a"), ("b", "c"), ("c", "b"), ("c", "d"), ("d", "c")]
    cross = generate_center_context([["a", "b"], ["c", "d"]], 1)
    no_cross = ("b", "c") not in cross and ("c", "b") not in cross
    results.append(report("Context window", pairs == expected and no_cross))

    # 2. Forward pass
    vocabulary, E, U = make_tiny_model()
    h = E[0]
    scores = calculate_scores(h, U)
    results.append(report("Forward pass: scores [0, 1, -1]",
                          lists_close(scores, [0.0, 1.0, -1.0])))

    # 3. Probabilities and loss
    loss, probabilities = cross_entropy_loss(scores, 1)
    passed = (
        lists_close(probabilities, [0.244728, 0.665241, 0.090031])
        and close(loss, 0.407606)
        and close(sum(probabilities), 1.0)
        and all(p >= 0 for p in probabilities)
    )
    results.append(report("Probabilities and loss", passed))

    # 4. Gradients
    e = calculate_score_gradients(probabilities, 1)
    grad_U = calculate_U_gradients(h, e)
    grad_h = calculate_E_gradients(U, e)
    passed = (
        lists_close(grad_U[0], [0.244728, -0.334759, 0.090031])
        and lists_close(grad_U[1], [0.0, 0.0, 0.0])
        and lists_close(grad_h, [-0.424790, 0.334759])
    )
    results.append(report("Gradients", passed))

    # 5. One update (learning rate 0.1)
    vocabulary, E, U = make_tiny_model()
    train_one_pair(E, U, 0, 1, 0.1)
    new_scores = calculate_scores(E[0], U)
    new_loss, _ = cross_entropy_loss(new_scores, 1)
    passed = (
        lists_close(E[0], [1.042479, -0.033476])
        and close(new_loss, 0.361859)
        and E[1] == [0.0, 1.0]
        and E[2] == [1.0, 1.0]
    )
    results.append(report("One update", passed))

    # 6. Extreme scores
    extreme_loss, extreme_probs = cross_entropy_loss([1000.0, 0.0], 1)
    results.append(report("Extreme scores: loss ~ 1000",
                          close(extreme_loss, 1000.0)))

    # 7. Shift invariance
    base_scores = [0.0, 1.0, -1.0]
    shifted_scores = [s + 100.0 for s in base_scores]
    base_loss, base_probs = cross_entropy_loss(base_scores, 1)
    shifted_loss, shifted_probs = cross_entropy_loss(shifted_scores, 1)
    passed = lists_close(base_probs, shifted_probs) and close(base_loss, shifted_loss)
    results.append(report("Shift invariance", passed))

    # 8. Saved model (optional)
    vocabulary, E, U = make_tiny_model()
    save_model("test_model.json", vocabulary, E, U)
    loaded_vocabulary, loaded_E, loaded_U = load_model("test_model.json")
    original_neighbors = find_nearest_neighbors("cats", vocabulary, E, 2)
    loaded_neighbors = find_nearest_neighbors("cats", loaded_vocabulary, loaded_E, 2)
    same_order = [w for w, _ in original_neighbors] == [w for w, _ in loaded_neighbors]
    same_sims = lists_close([s for _, s in original_neighbors],
                            [s for _, s in loaded_neighbors])
    loaded_scores = calculate_scores(loaded_E[0], loaded_U)
    _, loaded_probs = cross_entropy_loss(loaded_scores, 1)
    passed = (
        loaded_vocabulary == vocabulary
        and same_order and same_sims
        and lists_close(loaded_probs, probabilities)
    )
    results.append(report("Saved model (optional)", passed))

    # 9. Numerical gradient check (optional)
    vocabulary, E, U = make_tiny_model()
    difference = numerical_gradient_check(E, U, 0, 1)
    results.append(report(
        f"Numerical gradient check (optional), max diff {difference:.2e}",
        difference <= TOLERANCE))

    print(f"\n{sum(results)}/{len(results)} checks passed.\n")
    return all(results)

def find_corpus_path():
    here = Path(__file__).parent
    candidates = [
        here.parent / "corpus" / "corpus.txt",
        here / "corpus.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_training_data(window_size):
    text = read_corpus(find_corpus_path())
    text = normalize_text(text)
    sentences = split_into_sentences(text)

    tokenized_sentences = []
    for sentence in sentences:
        tokenized_sentences.append(tokenize(sentence))

    vocabulary = build_vocabulary(tokenized_sentences)
    pairs = generate_center_context(tokenized_sentences, window_size=window_size)
    indexed_pairs = convert_pairs_to_ids(pairs, vocabulary)

    return tokenized_sentences, vocabulary, indexed_pairs


if __name__ == "__main__":
    embedding_dim = 10
    learning_rate = 0.1
    num_epochs = 20
    window_size = 2

    print(" Implementation checks")
    run_checks()

    print("Loading corpus")
    tokenized_sentences, vocabulary, indexed_pairs = load_training_data(window_size)
    vocab_size = len(vocabulary)

    print("Seed:", SEED)
    print("Number of sentences:", len(tokenized_sentences))
    print("Vocabulary size:", vocab_size)
    print("Number of training pairs:", len(indexed_pairs))
    print(f"Embedding dim: {embedding_dim}, window: {window_size}, "
          f"learning rate: {learning_rate}, epochs: {num_epochs}\n")

    E, U = initialize_parameters(vocab_size, embedding_dim)

    print("Training")
    loss_history = train(E, U, indexed_pairs, learning_rate, num_epochs)

    # Cosine  checks
    print("\nCosine  checks")
    print(cosine_similarity([1, 1], [2, 2]))    # 1
    print(cosine_similarity([1, 0], [0, 1]))    # 0
    print(cosine_similarity([1, 0], [-1, 0]))   # -1
    print(cosine_similarity([0, 0], [1, 2]))    # 0

    # Five query words, three nearest vectors each (plus one unknown word)
    print("\n Nearest neighbors")
    query_words = ["bosonni", "mukkeen", "lafa", "qilleensa", "manca’iinsa", "computer"]
    print_nearest_neighbors(query_words, vocabulary, E, top_n=3)

    # Related vs unrelated pairs
    related_pairs = [
        ("qilleensa", "kaarboon"),
        ("qilleensa", "ooksiijiinii"),
        ("bosona", "mukkeen"),
        ("bosona", "manca’iinsa"),
    ]
    unrelated_pairs = [
        ("qilleensa", "manneen"),
        ("kaarboon", "ijaarsaa"),
        ("ooksiijiinii", "qubsumaa"),
        ("bosona", "waajjira"),
    ]

    print("\nRelated pairs")
    related_results = evaluate_word_pairs(related_pairs, vocabulary, E)
    for word_a, word_b, similarity in related_results:
        print(f"{word_a} - {word_b}: {similarity:.4f}")

    print("\n Unrelated pairs")
    unrelated_results = evaluate_word_pairs(unrelated_pairs, vocabulary, E)
    for word_a, word_b, similarity in unrelated_results:
        print(f"{word_a} - {word_b}: {similarity:.4f}")

    print("\nAverage similarity:")
    print("Related:", average_similarity(related_results))
    print("Unrelated:", average_similarity(unrelated_results))

    # Optional: save the trained model and reload it
    save_model("trained_model.json", vocabulary, E, U)
    loaded_vocabulary, loaded_E, loaded_U = load_model("trained_model.json")
    original = find_nearest_neighbors("bosonni", vocabulary, E, 3)
    reloaded = find_nearest_neighbors("bosonni", loaded_vocabulary, loaded_E, 3)
    print("\nReloaded model gives same neighbors:",
          [w for w, _ in original] == [w for w, _ in reloaded])
