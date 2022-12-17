# 1 Word Embeddings
## 1.1 Distributional Hypothesis
Words which frequently appear in similar contexts have similar meaning.

## 1.2 Word2Vec
### 1.2.1 Objective Function
- Take a huge text corpus.
- Go over the text with a sliding window, moving one word at a time.
- For the central word $P(w_t)$, compute probabilities of context words:
  - Maximize the data likelihood:
  $$L(\theta)=\prod_{t=1}^T\prod_{-m\leqq j\leqq m,\ j\neq 0}P(w_{t+j}|w_t,\theta)$$
  - Equivalent to minimizing the negative log-likelihood:
  $$Loss(\theta)=-\frac{1}{T}\sum_{t=1}^T\sum_{-m\leqq j\leqq m,\ j\neq 0}\log P(w_{t+j}|w_t,\theta)$$

### 1.2.2 How to compute $P(w_{t+j}|w_t,\theta)$
- For each word $w$, we have **two vectors**:
  - $v_w$ when it is a central word
  - $u_w$ when it is a context word

- For the central word $c$ and context word $o$:
  $$P(o|c)=\frac{\exp(u_o^Tv_c)}{\sum_{w\in V}\exp(u_w^Tv_c)}$$
  - $\exp(u_o^Tv_c)$ measures the similarity of $o$ and $c$ .
  - $\sum_{w\in V}\exp(u_w^Tv_c)$ measures the average similarity of $c$ and random word $w$ in the vocabulary $V$.

Training the objective function with such a probability increases the similarity of context words and central words, while reducing others.

### 1.2.3 Faster training: negative sampling
Use a subset of words in $V$ instead of all other words in $V$.

### 1.2.3 Word2Vec Variants: Skip-Gram and CBOW
- Skip-Gram is what we did so far
- CBOW predicts **central** from sum of context

### 1.2.4 Standard Hyperparameters
- Model: Skip-Gram with Negative Sampling (SGNS)
- Number of negative examples: 2-5 for huge datasets
- Embedding dimensionality: 300
- Sliding window: 5-10

## 1.3 Practical tips
- Vocabulary is chosen in advance. Therefore, some tokens may be “unknown" (UNK).
- Initialize with pretrained word embeddings and then fine-tune the embeddings.