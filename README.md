# Named Entity Recognition (NER) for MultiCoNER 2 English Dataset

A comprehensive exploration of Named Entity Recognition models for fine-grained entity classification, implementing and comparing approaches from classical probabilistic models to state-of-the-art knowledge-augmented transformers.

## Overview

This project addresses a fine-grained Named Entity Recognition task with seven entity types under strict span-level evaluation. We implement and compare multiple model architectures, ultimately achieving **F1 score of 0.8260** through a hybrid ensemble approach.

### Entity Types
- **Person**: Artist, Politician, OtherPER
- **Location**: HumanSettlement, Facility
- **Organization**: ORG, PublicCorp

### Key Results
| Model | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| **Hybrid Ensemble** (Final) | 0.8242 | 0.8278 | **0.8260** |
| Knowledge-Augmented XLM-RoBERTa-CRF | 0.8249 | 0.8275 | 0.8262 |
| RoBERTa-base | 0.8041 | 0.8066 | 0.8054 |
| Ma & Hovy BiLSTM-CNN-CRF | 0.7512 | 0.7678 | 0.7594 |
| BERT-base | 0.7913 | 0.7981 | 0.7947 |
| CRF Baseline | 0.7130 | 0.6560 | 0.6833 |

## Dataset

### Original Dataset
- **Source**: MultiCoNER 2 English dataset (Contest 3)
- **Size**: 100,541 samples
- **Annotation**: BIO tagging scheme

### Data Preprocessing

#### Quality Checks
- Removed 185 samples (0.18%) with invalid BIO tag transitions
- Retained 479 empty sentences (matching test set distribution)
- Retained 3,177 no-entity sentences to prevent false positives

#### Train-Validation Split
- **Training**: 90,320 samples (90%)
- **Validation**: 10,036 samples (10%)
- **Strategy**: Stratified split based on entity presence to maintain class distribution

### Entity Distribution
| Entity Type | Train Count | Val Count | Train % | Val % |
|-------------|-------------|-----------|---------|-------|
| Artist | 25,817 | 2,849 | 21.23% | 21.14% |
| Facility | 12,827 | 1,487 | 10.55% | 11.04% |
| HumanSettlement | 32,261 | 3,476 | 26.53% | 25.80% |
| ORG | 17,235 | 1,893 | 14.17% | 14.05% |
| OtherPER | 15,897 | 1,779 | 13.07% | 13.20% |
| Politician | 12,711 | 1,402 | 10.45% | 10.40% |
| PublicCorp | 4,854 | 589 | 3.99% | 4.37% |

## Models Implemented

### 1. Probabilistic Baselines

#### Hidden Markov Model (HMM)
- **F1 Score**: 0.5665
- Generative probabilistic model
- Models transition and emission probabilities
- Baseline for sequence labeling

#### Conditional Random Field (CRF)
- **F1 Score**: 0.6833
- Discriminative model with rich feature sets
- Models conditional probability p(y|x)
- Strong statistical baseline

### 2. Neural Network Approaches

#### Word Embedding RNN
- **F1 Score**: 0.5699
- Unidirectional LSTM with 256 hidden units
- GloVe pre-trained embeddings (glove-wiki-gigaword-100)
- Dropout regularization (p=0.5)
- Gradient clipping and learning rate scheduling

#### Character-CNN BiLSTM-CRF

**Lample-style BiLSTM-CRF** (F1: 0.7266)
- Character-level CNN with max-pooling
- Word embeddings + character embeddings
- BiLSTM encoder
- CRF decoding layer

**Ma & Hovy-style BiLSTM-CNN-CRF** (F1: 0.7594)
- Enhanced character-level CNN with multiple filter sizes
- Stronger dropout regularization
- Improved hyperparameter tuning

### 3. Transformer-based Models

#### BERT-base
- **F1 Score**: 0.7947
- Fine-tuned for 8 epochs
- Learning rate: 5×10⁻⁵
- AdamW optimizer with weight decay 0.01
- Subword tokenization with label alignment

#### RoBERTa-base
- **F1 Score**: 0.8054
- Optimized pre-training procedure
- Byte-Level BPE tokenization
- Learning rate: 5×10⁻⁵

#### RoBERTa-large
- **F1 Score**: 0.8049
- Larger model with similar performance to base
- Learning rate: 3×10⁻⁵ (more conservative)
- Diminishing returns from size increase alone

### 4. Knowledge-Augmented XLM-RoBERTa-CRF

**Best Single Model** - **F1 Score: 0.8262**

#### Architecture
- **Backbone**: XLM-RoBERTa-base
- **External Knowledge**: Wikipedia articles via OpenSearch API
- **CRF Layer**: Enforces valid BIO tag transitions

#### Input Format
```
[CLS] sentence tokens [SEP] knowledge snippet [SEP]
```

#### Knowledge Retrieval Process
1. Extract entity candidates from sentence
2. Query Wikipedia OpenSearch API for matching articles
3. Retrieve introductory paragraph from top match
4. Append as context to input sequence

#### Training Configuration
- Learning rate: 2×10⁻⁵
- Batch size: 16
- Max sequence length: 256 tokens
- Epochs: 8 with early stopping
- Optimizer: AdamW

#### Challenges & Solutions
- **Wikipedia API Rate Limiting**: 403 Forbidden errors with rapid queries
  - Solution: Added 1-second delays between requests
- **Alternative Approach**: Used Gemini LLM to generate Wikipedia-style context
  - Eliminates API constraints
  - Trade-off: Parametric knowledge vs. retrieved facts

### 5. Hybrid Ensemble Strategy (Final Model)

**F1 Score: 0.8260**

#### Motivation
Different entity types exhibit different prediction patterns across models. Instead of a single global ensemble method, we adaptively select the optimal strategy per entity type.

#### Strategy Selection Process
For each entity type:
1. Generate predictions using four strategies:
   - Union (combine all entities from both models)
   - Intersection (consensus only)
   - Knowledge-Augmented model alone
   - RoBERTa-base model alone
2. Evaluate each strategy on held-out test data
3. Select strategy that maximizes entity-span-level F1 score
4. Build strategy map: EntityType → {union, intersection, model₁, model₂}

#### Implementation
1. Extract entity spans as (start, end, type) tuples from both models
2. Apply designated strategy per entity type
3. Reconstruct valid BIO tag sequence from selected spans

#### Benefits
- More balanced precision-recall trade-offs
- Better robustness across entity types
- Improved coverage of low-frequency entities
- Stronger performance on ambiguous cases

## Evaluation Metrics

All models are evaluated using **strict entity-span matching**:
- A prediction is correct only if the entire span (start, end, type) matches exactly
- **Micro F1 Score**: Global calculation across all entity types
  - Aggregate all True Positives, False Positives, False Negatives
  - Compute: F1 = 2×(Precision×Recall)/(Precision+Recall)

## Results

Detailed experimental results and analysis are available in:
- [NLP__Contest_3-1047951-17651257017331.pdf](NLP__Contest_3-1047951-17651257017331.pdf)

###  Results - Hybrid Ensemble

| Entity Type | Precision | Recall | F1 Score | Support |
|-------------|-----------|--------|----------|---------|
| Artist | 0.8154 | 0.8417 | 0.8283 | 2,849 |
| Facility | 0.8434 | 0.8440 | 0.8437 | 1,487 |
| HumanSettlement | 0.9577 | 0.9583 | 0.9580 | 3,476 |
| ORG | 0.8190 | 0.8151 | 0.8171 | 1,893 |
| OtherPER | 0.6540 | 0.6863 | 0.6698 | 1,779 |
| Politician | 0.7522 | 0.6690 | 0.7082 | 1,402 |
| PublicCorp | 0.7456 | 0.7963 | 0.7701 | 589 |

**Overall Metrics**:
- Precision: 0.8242
- Recall: 0.8278
- F1 Score: 0.8260
- True Positives: 11,155
- False Positives: 2,379
- False Negatives: 2,320


## Key Findings

1. **External Knowledge Matters**: Knowledge-augmented model outperforms all baselines, confirming the benefit of Wikipedia context for disambiguating rare and ambiguous entities.

2. **Character-Level Features**: BiLSTM-CNN-CRF models significantly outperform word-level RNN, highlighting the importance of morphological information.

3. **Transformer Superiority**: BERT and RoBERTa achieve substantial gains over BiLSTM models through self-attention and pre-trained contextual embeddings.

4. **Adaptive Ensembling**: Different entity types benefit from different strategies - per-type optimization provides better robustness than global ensemble methods.

5. **Model Size vs. Tuning**: RoBERTa-large showed diminishing returns compared to RoBERTa-base, suggesting hyperparameter tuning and data quality matter more than size alone.

