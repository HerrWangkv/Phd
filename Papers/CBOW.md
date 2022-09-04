# Continuous Bag of Words Model
- Given: Some text
- Goal: Each word needs to be represented as an embedding (vector)

## Intuition
- The meaning of a word can be obtained by understanding its context.
- Mathematically, the word embedding $\boldsymbol{v}_c$ should be close to the mean of its context: 
  $$\boldsymbol{v}_c\approx\frac{\boldsymbol{v}_{c-m}+\boldsymbol{v}_{c-m+1}+...+\boldsymbol{v}_{c+m}}{2m}$$

## Algorithm
1. Build up a set $V$ with all unique words.
2. Generate one-hot encodings $\boldsymbol{x}_{1},...,\boldsymbol{x}_{|V|}\in\mathbb{R}^{|V|}$for all unique words existed in the text.
3. Randomly initialize a network with two layers, whose parameters are respectively $\mathcal{V}\in\mathbb{R}^{n\times|V|}$ and $\mathcal{U}\in\mathbb{R}^{|V|\times n}$. Those two matrices are called **input word embedding** matrix and **output word embedding** matrix. We hope the $i^{\th}$ column and $i^{\th}$ row could represent the $i^{\th}$ word of $V$.
4. For each window consisted of $2m+1$ words, whose center word's one-hot encoding is expressed as $\boldsymbol{y}=\boldsymbol{x}_{c}$:
   1. Calculate the input word embeddings of $2m$ words around the center word:
    $$\boldsymbol{v}_{c-m}=\mathcal{V}\boldsymbol{x}_{c-m},\ ...,\ \boldsymbol{v}_{c+m}=\mathcal{V}\boldsymbol{x}_{c+m}\in\mathbb{R}^{n}$$
   2. Calculate the mean of those $2m$ word embeddings as **context embedding**:
    $$\boldsymbol{\hat v}=\frac{\boldsymbol{v}_{c-m}+\boldsymbol{v}_{c-m+1}+...+\boldsymbol{v}_{c+m}}{2m}$$
   3. Multiply this context embedding with the output word embedding matrix to get a **score vector**:
   $$\boldsymbol{z}=\mathcal{U}\boldsymbol{\hat v}\in\mathbb{R}^{|V|}$$
   4. Calculate the softmax output $\boldsymbol{\hat y}=softmax(\boldsymbol{z})$ of the score vector, which basically represents the similarity between each word embedding in $V$ and the context embedding.
   5. Use cross entropy loss to optimize $\mathcal{V}$ and $\mathcal{U}$:
   $$L(\boldsymbol{\hat y},\boldsymbol{y})=-\sum_{j=1}^{|V|}\boldsymbol{y}_j\log(\boldsymbol{\hat y}_j)$$
5. Since $\mathcal{V}$ is closer to the input layer and has thus less noise, $\mathcal{V}$ is finally utilized as the word embedding matrix.

## Properties
1. The more similar two word embeddings are, the closer their meanings are. Similarity can be represented using either inner product or cosine similarity.
2. Subtracting two word embeddings can generate a new vector. One famous example is "king - man + woman = queen". A simple explanation is that different dimensions may contain different information. In other words, some dimensions in the word embedding of word "king" stand for "royal" and others stand for "man". 