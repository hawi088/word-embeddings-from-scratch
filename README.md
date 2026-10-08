# Word Embeddings from Scratch: Afaan Oromo Skip-gram

A small skip-gram neural network that learns word embeddings for **Afaan Oromo**,
written in pure Python (lists, loops, `math`, `random`). The model, gradients and SGD
are implemented by hand. No NumPy, no pretrained embeddings, no automatic
differentiation and no neural-network libraries are used.

## Project structure

```
.
├── corpus/
│   └── corpus.txt          # Afaan Oromo training text
├── src/
│   ├── preprocessing.py    # read, lowercase, split sentences, tokenize
│   ├── vocabulary.py       # vocabulary, word -> id, pairs -> ids, embedding lookup
│   ├── pairs.py            # center-context pair generation
│   └── model.py            # network, training, checks, nearest neighbors, save/load
├── requirements.txt
└── README.md
```

## How to run

Requires Python 3 only (standard library). Run from inside `src/`:

```bash
cd src
python -X utf8 model.py
```

(`-X utf8` makes the console print Afaan Oromo characters correctly on Windows.)

This will, in order:

1. Run the implementation checks on the tiny "cats eat food" example.
2. Load and preprocess the corpus, build the vocabulary and the training pairs.
3. Train the model, printing the mean loss before training and after every epoch.
4. Print cosine sanity checks, the nearest neighbors of the query words, and
   related vs. unrelated word-pair similarities.
5. Save the trained model to `trained_model.json`, reload it, and confirm the
   neighbors match.

To inspect only the data pipeline, run `python pairs.py` or `python preprocessing.py`.

## Corpus and preprocessing

- **Language:** Afaan Oromo
- **Content:** a short text about forests and the environment (forest cover,
  deforestation, trees, carbon and oxygen in the air, land use).
- **Size:** 37 sentences, 440 tokens, 307 distinct words.
- **Preprocessing:** lowercase the text, split sentences on `.`, remove the
  punctuation `, . ! ? ; :`, split tokens on whitespace. The apostrophe (’) is kept,
  because it is part of Afaan Oromo spelling (e.g. `manca’iinsa`).
- **Vocabulary:** every distinct token, in order of first appearance, with no
  frequency cutoff. Size: **307**.
- **Pairs:** center-context pairs from a window of 2 words on each side. Pairs are
  built per sentence, so none cross sentence boundaries. Total: **1,538 pairs**.

## Model

Row-vector notation, where `V` is the vocabulary size and `d` the embedding dimension.

| Step | Equation |
|---|---|
| Embedding lookup | `h = E[i]` (shape 1 × d) |
| Scores | `s = hU` (shape 1 × V) |
| Stable softmax | `m = max(s)`, `a[j] = exp(s[j] - m)`, `p[j] = a[j] / sum(a)` |
| Loss (one pair) | `L = (m - s[t]) + ln(sum(a))` |
| Score error | `e[j] = p[j] - 1` if `j = t`, otherwise `e[j] = p[j]` |
| Gradients | `grad_U[k,j] = h[k] e[j]`, `grad_h[k] = sum_j U[k,j] e[j]` |
| Update | `U <- U - lr * grad_U`, `E[i] <- E[i] - lr * grad_h` |
| Dataset loss | `J = (1/N) sum_n L_n`, evaluated at fixed weights |

- `E` has shape V × d (307 × 10) and `U` has shape d × V (10 × 307). Both are
  initialized with `random.uniform(-0.1, 0.1)` using **seed 42**.
- Both gradients are computed from the old `E` and `U` before either matrix is
  updated. Only the selected row of `E` is updated.
- Pairs are shuffled at the start of every epoch.
- The loss uses the max-shifted form above, so it stays finite even for very large scores.

## Hyperparameters

| Parameter | Value |
|---|---|
| Embedding dimension (d) | 10 |
| Context window | 2 |
| Learning rate | 0.1 |
| Epochs | 20 |
| Seed | 42 |

## Implementation checks

`run_checks()` in `model.py` runs the assignment's checks (tolerance `0.00001`) on the
tiny example and prints PASS/FAIL for each:

| # | Check | Required | Result |
|---|---|---|---|
| 1 | Context window (`[a,b,c,d]`, window 1, no cross-sentence pairs) | yes | PASS |
| 2 | Forward pass (scores `[0, 1, -1]`) | yes | PASS |
| 3 | Probabilities and loss (`L ≈ 0.407606`) | yes | PASS |
| 4 | Gradients (`grad_U`, `grad_h`) | yes | PASS |
| 5 | One update (`E[cats] ≈ [1.042479, -0.033476]`, loss ≈ 0.361859) | yes | PASS |
| 6 | Extreme scores (`[1000, 0]`, loss ≈ 1000) | yes | PASS |
| 7 | Shift invariance (add 100 to all scores) | yes | PASS |
| 8 | Saved model (JSON save/reload) | optional | PASS |
| 9 | Numerical gradient check (ε = 0.00001) | optional | PASS (max difference 6.76e-12) |

All 9 of 9 checks pass.

## Training results

Mean training loss over the same 1,538 pairs, evaluated with the weights fixed:

| Stage | Mean loss |
|---|---|
| Before training | 5.727243 |
| Epoch 1 | 5.720042 |
| Epoch 2 | 5.707847 |
| Epoch 3 | 5.676715 |
| Epoch 4 | 5.585237 |
| Epoch 5 | 5.349282 |
| Epoch 6 | 4.976310 |
| Epoch 7 | 4.571905 |
| Epoch 8 | 4.182521 |
| Epoch 9 | 3.814812 |
| Epoch 10 | 3.492662 |
| Epoch 11 | 3.214375 |
| Epoch 12 | 3.017491 |
| Epoch 13 | 2.837915 |
| Epoch 14 | 2.685751 |
| Epoch 15 | 2.587068 |
| Epoch 16 | 2.511381 |
| Epoch 17 | 2.440234 |
| Epoch 18 | 2.398246 |
| Epoch 19 | 2.342649 |
| Epoch 20 | 2.327644 |

The loss falls steadily from 5.727 to 2.328. The starting value is almost exactly
ln(307) ≈ 5.727, which is the loss of a model that gives every word the same
probability, as expected for small random initial weights. Training was slow for the
first three epochs and then dropped quickly.

## Nearest neighbors

Cosine similarity on the rows of `E`; the query word itself is excluded.

```
Nearest neighbors of 'bosonni':
  misoomaatinillee: 0.8408
  maqaa: 0.7594
  nicirama: 0.7540
Nearest neighbors of 'mukkeen':
  tuuta: 0.7872
  ilaallata: 0.7466
  jiruuti: 0.7333
Nearest neighbors of 'lafa':
  qonnaa: 0.8424
  uffisee: 0.8220
  madaalamu: 0.8192
Nearest neighbors of 'qilleensa':
  dhabuun: 0.7894
  hoʼisa: 0.7411
  ooksaayidiin: 0.6789
Nearest neighbors of 'manca’iinsa':
  malees: 0.7328
  kanaafis: 0.7131
  tattaaffiin: 0.6975
'computer' is not in the vocabulary.
```

The unknown word `computer` is handled without a crash, and the cosine function
returns 0 for a zero-norm vector (`cosine([0, 0], [1, 2]) = 0`).

### Related vs. unrelated pairs

| Pair | Type | Cosine |
|---|---|---|
| qilleensa – kaarboon | related | 0.5227 |
| qilleensa – ooksiijiinii | related | 0.3267 |
| bosona – mukkeen | related | 0.5167 |
| bosona – manca’iinsa | related | -0.0081 |
| qilleensa – manneen | unrelated | -0.0802 |
| kaarboon – ijaarsaa | unrelated | -0.5676 |
| ooksiijiinii – qubsumaa | unrelated | -0.2394 |
| bosona – waajjira | unrelated | -0.0974 |

| Group | Average cosine similarity |
|---|---|
| Related pairs | 0.3395 |
| Unrelated pairs | -0.2461 |

After training, the model reloaded from `trained_model.json` gives the same nearest
neighbors as the original.

## Discussion

The training itself works: all checks pass and the loss falls from 5.73 to 2.33.
The embeddings, however, are only weakly meaningful, and a lower loss does not by
itself show good semantic quality.

- **Neighbors are mostly noisy.** Several are not clearly related to the query:
  for `bosonni` the closest word is `misoomaatinillee`, and for `manca’iinsa` the
  neighbors are function-like words (`malees`, `kanaafis`). A few look sensible,
  such as `lafa` with `qonnaa` (farming) and `qilleensa` with `ooksaayidiin`
  (oxide), but these could also be chance.
- **Related pairs score higher than unrelated pairs** (0.34 against -0.25), which
  points in the right direction. With only four pairs per group this is weak evidence,
  and one related pair (`bosona – manca’iinsa`) scored about 0.
- **The corpus is very small for this task.** There are 307 distinct words in only
  440 tokens, so most words appear once or twice. The model has little evidence about
  any single word, and part of the loss reduction is probably the model fitting the
  specific pairs it was given rather than learning general meaning.
- **Possible improvements:** a larger corpus, handling of the many inflected forms
  of the same word (for example `bosona`, `bosonni`, `bosonaan`, `bosonaa`), more epochs, and
  comparing different window sizes or embedding dimensions.

## Optional extensions

- **JSON save/load** of `E`, `U` and the vocabulary (`save_model`, `load_model`),
  verified by check 8 and by the reload test at the end of `model.py`.
- **Numerical gradient check** (`numerical_gradient_check`), verified by check 9.
- **Related vs. unrelated pair evaluation** (`evaluate_word_pairs`).

## Limitations

- Vocabulary lookup is a linear search, which is fine for a small corpus but slow for a large one.
- Sentences are split only on `.`, so text that uses other sentence endings is kept as longer sentences.
- No subsampling or negative sampling: the full softmax is computed for every pair.
- Different spellings and inflections of one word are separate vocabulary entries
  (for example `qilleensa` and `qillensa`).
