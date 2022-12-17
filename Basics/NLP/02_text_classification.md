# 2 Text Classification
## 2.1 Examples
- spam / not spam
- hashtags (topic)
- Sentiment(情绪)

## 2.2 Generative vs. Discriminative models
### 2.2.1 Generative models
- Learn data distribution $p(x,y)=p(x,y)\cdot p(y)$
- $y = \argmax_k p(x,y)$

### 2.2.2 Discriminative models
- Learn boundary between classes $p(y|x)$
- $y=\argmax_k p(y|x)$

## 2.3 Classical methods
Naive Bayes, Logistic Regression, SVM

## 2.4 Neural methods
### 2.4.1 (Weighted) BOW
Use weights (e.g. tf-idf weights) to calculate the weighted sum of all word embeddings:
$$tf-idf(w,d,D)=tf(w,d)\cdot idf(w,D)$$
- $tf(w,d)$ means the **term frequency** of word $w$ in text $d$.
- $idf(w,D)$ means the **inverse document frequency**, which increases when less texts inside corpus $D$ contain word $w$.

### 2.4.2 Convolutional models
TDNN + Pooling

### 2.4.3 Recurrent models
The final state outputs the context vector, which represents the whole sentence.

## 2.5 Practical tips: Data augmentation
- Word dropout:
  - Pick several words randomly
    - replace them with UNK (unknown).
    - replace them with random words.
- Use external resource (e.g. thesaurus): Pick words where you have synonyms. 
- Use separate models fo paraphrasing (e.g. double translation)

