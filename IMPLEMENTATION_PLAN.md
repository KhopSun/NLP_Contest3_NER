# NER Contest - Comprehensive Implementation Plan

## Table of Contents
1. [Overview](#overview)
2. [Project Status](#project-status)
3. [Model Implementation Plans](#model-implementation-plans)
4. [Final Deliverables](#final-deliverables)
5. [Timeline Estimates](#timeline-estimates)
6. [Resources & References](#resources--references)

---

## Overview

This document outlines the complete implementation plan for the NLP Named Entity Recognition (NER) Contest. The goal is to build and compare multiple NER models, ordered from simplest to most advanced, to achieve the best entity-span level F1 score.

**Key Requirements:**
- Minimum 4 model/feature combinations
- At least 2 must involve deep learning training
- Entity-span level evaluation (strict matching)
- Report documenting all experiments (≤4 pages)

**Our Approach:** Implement 11 models (exceeds requirements!) to demonstrate thorough experimentation.

---

## Project Status

### ✅ Completed (3/17)
1. ✅ Research best NER methods and approaches online
2. ✅ Create `1_EDA.ipynb` for data exploration, cleaning, and splitting
3. ✅ Implement entity-span level evaluation metrics in `utils.py`

### ⏳ Pending (14/17)

**Phase 2: Model Building (11 models)**
- Model 0: HMM Baseline
- Model 1: CRF Baseline
- Model 2: Word Embeddings + RNN
- Model 3: Word Embeddings + POS Features + RNN
- Model 4: Character-CNN + BiLSTM-CRF
- Model 5: Fine-tuned BERT
- Model 6: BERT + CRF Hybrid
- Model 7: Attention-Based BiLSTM-CRF
- Model 8: LLM + Basic Prompting
- Model 9: LLM + Few-Shot Prompting
- Model 10: Ensemble

**Phase 3: Final Deliverables (3 tasks)**
- Compare all models and select final model
- Generate predictions on test set
- Write the report (≤4 pages)

---

## Model Implementation Plans

### Model 0: HMM (Hidden Markov Model)

**Complexity:** Low | **Training Required:** Yes (classical) | **Expected F1:** 75-82%

#### Implementation Steps

1. **Feature Engineering**
   - Use word tokens as observations
   - States = BIO tags
   - Learn transition probabilities (tag → tag)
   - Learn emission probabilities (tag → word)

2. **Libraries**
   - `hmmlearn` or `nltk.tag.hmm`
   - `sklearn` for preprocessing

3. **Training Approach**
   - Estimate transition matrix from training data
   - Estimate emission matrix from training data
   - Use Viterbi algorithm for decoding
   - Handle unknown words with smoothing

4. **Key Challenges**
   - Poor handling of unknown words (OOV problem)
   - No context beyond immediate neighbors
   - Simple probabilistic assumptions
   - Struggles with rare entity types

5. **Hyperparameters**
   - Smoothing parameter for OOV words
   - No complex tuning required

**Notebook:** `3_HMM.ipynb`

**References:**
- NLTK HMM Tagger: https://www.nltk.org/api/nltk.tag.hmm.html
- Hidden Markov Models for NER: https://www.depends-on-the-definition.com/named-entity-recognition-with-hidden-markov-model/

---

### Model 1: CRF (Conditional Random Fields)

**Complexity:** Medium | **Training Required:** Yes (classical) | **Expected F1:** 85-88%

#### Implementation Steps

1. **Feature Engineering** (Critical for CRF!)

   **Word-level features:**
   - `word.lower()` - lowercase form
   - `word.isupper()` - all uppercase?
   - `word.istitle()` - title case?
   - `word.isdigit()` - contains digits?
   - `word[:3]`, `word[-3:]` - prefix/suffix (3 chars)

   **Word shape features:**
   - Pattern: "Xxxxx" → capitalized word
   - Pattern: "XXXXX" → all caps
   - Pattern: "XxXxX" → mixed case
   - Pattern: "dd/dd/dddd" → date-like

   **Context window features:**
   - Previous word features: `word[-1].lower()`, `word[-1].isupper()`
   - Next word features: `word[+1].lower()`, `word[+1].isupper()`
   - Bigram features: `(word[-1], word[0])`, `(word[0], word[+1])`

   **Position features:**
   - `BOS` - beginning of sentence
   - `EOS` - end of sentence

   **Optional advanced features:**
   - POS tags (from spaCy)
   - Gazetteers (lists of known entities)
   - Brown clusters
   - Word embeddings

2. **Libraries**
   - `sklearn-crfsuite` (recommended - scikit-learn compatible)
   - `python-crfsuite` (alternative)

3. **Training**
   ```python
   import sklearn_crfsuite
   from sklearn_crfsuite import metrics

   crf = sklearn_crfsuite.CRF(
       algorithm='lbfgs',
       c1=0.1,        # L1 regularization
       c2=0.1,        # L2 regularization
       max_iterations=100,
       all_possible_transitions=True
   )
   crf.fit(X_train, y_train)
   ```

4. **Hyperparameters to Tune**
   - `c1` (L1): 0.01, 0.1, 1.0
   - `c2` (L2): 0.01, 0.1, 1.0
   - `max_iterations`: 50-200
   - Feature templates

5. **Key Advantages**
   - Automatically enforces valid BIO sequences
   - Can use rich, hand-crafted feature sets
   - Fast training and inference
   - Interpretable (can inspect feature weights)
   - No need for GPU

6. **BIO Sequence Enforcement**
   - CRF learns transition scores between tags
   - `all_possible_transitions=True` allows model to learn valid patterns
   - Viterbi decoding ensures globally optimal tag sequence

**Notebook:** `4_CRF.ipynb`

**References:**
- sklearn-crfsuite Documentation: https://sklearn-crfsuite.readthedocs.io/
- CRF Tutorial for NER: https://towardsdatascience.com/named-entity-recognition-with-conditional-random-fields-94cc61eb8f69
- Feature Engineering for CRF: https://www.chokkan.org/software/crfsuite/tutorial.html

---

### Model 2: Word Embeddings + RNN

**Complexity:** Medium | **Training Required:** Yes (deep learning) ✅ | **Expected F1:** 85-89%

#### Implementation Steps

1. **Word Embeddings**
   - **Pre-trained GloVe embeddings** (recommended)
     - GloVe 6B 100d: 400K vocabulary, 100-dimensional
     - GloVe 6B 300d: 400K vocabulary, 300-dimensional
     - Download: https://nlp.stanford.edu/projects/glove/
   - **Alternative:** Word2Vec, FastText

   **Vocabulary handling:**
   - Build vocabulary from training data
   - Map words to pre-trained embeddings
   - Unknown words: use random initialization or `<UNK>` token

2. **Architecture**
   ```
   Input (word indices)
       ↓
   Embedding Layer (frozen or fine-tuned)
       ↓
   LSTM/GRU Layer (128-256 units, bidirectional optional)
       ↓
   Dropout (0.3-0.5)
       ↓
   Dense Layer (num_tags)
       ↓
   Softmax
   ```

3. **Libraries**
   - **PyTorch** (recommended) or TensorFlow/Keras
   - `gensim` for loading pre-trained embeddings
   - `torch.nn.LSTM` or `torch.nn.GRU`

4. **Implementation (PyTorch)**
   ```python
   import torch
   import torch.nn as nn

   class RNN_NER(nn.Module):
       def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags):
           super().__init__()
           self.embedding = nn.Embedding(vocab_size, embedding_dim)
           self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                               batch_first=True, bidirectional=False)
           self.dropout = nn.Dropout(0.5)
           self.fc = nn.Linear(hidden_dim, num_tags)

       def forward(self, x):
           embeds = self.embedding(x)
           lstm_out, _ = self.lstm(embeds)
           lstm_out = self.dropout(lstm_out)
           return self.fc(lstm_out)
   ```

5. **Training**
   - **Loss:** CrossEntropyLoss (ignore padding tokens with `ignore_index=-100`)
   - **Optimizer:** Adam (lr=0.001)
   - **Batch size:** 32
   - **Epochs:** 10-20
   - **Early stopping:** Monitor validation F1, patience=3
   - **Gradient clipping:** `torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)`

6. **Hyperparameters**
   - Embedding dimension: 100, 300
   - Hidden dimension: 128, 256
   - Number of layers: 1, 2
   - Dropout: 0.3, 0.5
   - Learning rate: 0.0001, 0.001, 0.01
   - Bidirectional: True/False

7. **Data Preparation**
   ```python
   # Convert tokens to indices
   word2idx = build_vocab(train_tokens)
   X_train = [[word2idx.get(w, word2idx['<UNK>']) for w in sent] for sent in train_tokens]

   # Convert tags to indices
   tag2idx = {tag: idx for idx, tag in enumerate(unique_tags)}
   y_train = [[tag2idx[tag] for tag in sent] for sent in train_tags]
   ```

8. **Key Features**
   - Contextual understanding via recurrent connections
   - Pre-trained semantic knowledge from embeddings
   - Handles variable-length sequences
   - Can capture some long-range dependencies

**Notebook:** `5_Word_Emb_RNN.ipynb`

**References:**
- GloVe Embeddings: https://nlp.stanford.edu/projects/glove/
- PyTorch LSTM Tutorial: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
- NER with LSTM: https://www.depends-on-the-definition.com/sequence-tagging-lstm-crf/

---

### Model 3: Word Embeddings + POS + RNN

**Complexity:** Medium-High | **Training Required:** Yes (deep learning) ✅ | **Expected F1:** 87-90%

#### Implementation Steps

1. **Feature Extraction**

   **Word embeddings:**
   - GloVe 300d (richer representations)

   **POS (Part-of-Speech) tags:**
   - Extract using spaCy:
     ```python
     import spacy
     nlp = spacy.load("en_core_web_sm")

     def get_pos_tags(tokens):
         doc = nlp(" ".join(tokens))
         return [token.pos_ for token in doc]
     ```
   - POS tag set: NOUN, VERB, ADJ, ADV, PROPN, etc.
   - Create POS vocabulary and embeddings (learnable, 50d)

2. **Architecture**
   ```
   Input: (word_indices, pos_indices)
       ↓                    ↓
   Word Embedding        POS Embedding
   (300d, frozen)        (50d, learnable)
       ↓                    ↓
       └────── Concatenate ──────┘
                 ↓
         BiLSTM (256 units)
                 ↓
           Dropout (0.5)
                 ↓
         Dense (num_tags)
                 ↓
             Softmax
   ```

3. **Implementation (PyTorch)**
   ```python
   class RNN_POS_NER(nn.Module):
       def __init__(self, vocab_size, pos_vocab_size,
                    word_emb_dim, pos_emb_dim, hidden_dim, num_tags):
           super().__init__()
           self.word_embedding = nn.Embedding(vocab_size, word_emb_dim)
           self.pos_embedding = nn.Embedding(pos_vocab_size, pos_emb_dim)

           self.lstm = nn.LSTM(word_emb_dim + pos_emb_dim, hidden_dim,
                               batch_first=True, bidirectional=True)
           self.dropout = nn.Dropout(0.5)
           self.fc = nn.Linear(hidden_dim * 2, num_tags)  # *2 for bidirectional

       def forward(self, words, pos_tags):
           word_embeds = self.word_embedding(words)
           pos_embeds = self.pos_embedding(pos_tags)

           # Concatenate features
           combined = torch.cat([word_embeds, pos_embeds], dim=-1)

           lstm_out, _ = self.lstm(combined)
           lstm_out = self.dropout(lstm_out)
           return self.fc(lstm_out)
   ```

4. **Training**
   - Same optimizer/loss as Model 2
   - May need slightly lower learning rate (lr=0.0005)
   - POS embeddings learned during training (not pre-trained)
   - Monitor both word and POS embedding quality

5. **Data Preparation**
   ```python
   # Extract POS tags for all sentences
   train_pos = [get_pos_tags(tokens) for tokens in train_tokens]
   val_pos = [get_pos_tags(tokens) for tokens in val_tokens]

   # Create POS vocabulary
   pos2idx = build_pos_vocab(train_pos)

   # Convert to indices
   X_train_pos = [[pos2idx.get(p, pos2idx['<UNK>']) for p in sent]
                   for sent in train_pos]
   ```

6. **Key Advantages**
   - **Linguistic features improve accuracy**: POS helps disambiguate word meanings
   - **Example benefits:**
     - "Apple" (PROPN) → likely B-ORG
     - "apple" (NOUN) → likely O
     - Proper nouns (PROPN) → strong signal for entities
   - **+2-3% F1 improvement** over basic RNN
   - Still relatively fast training

7. **Why POS Helps NER**
   - Named entities are often proper nouns (PROPN)
   - Verbs (VERB) are rarely entities
   - POS provides syntactic context
   - Helps with unseen words (POS pattern recognition)

**Notebook:** `6_Word_POS_RNN.ipynb`

**References:**
- spaCy POS Tagging: https://spacy.io/usage/linguistic-features#pos-tagging
- POS Features for NER: https://aclanthology.org/W03-0419.pdf
- Multi-feature RNN: https://arxiv.org/abs/1603.01354

---

### Model 4: Character-CNN + BiLSTM-CRF

**Complexity:** High | **Training Required:** Yes (deep learning) ✅ | **Expected F1:** 90-91%

#### Implementation Steps

1. **Character-Level Embeddings**

   **Why characters matter:**
   - Handle out-of-vocabulary (OOV) words
   - Capture morphology: prefixes (un-, pre-), suffixes (-tion, -ness)
   - Useful for: proper names, compound words, misspellings

   **Architecture for character features:**
   ```
   Word: "Washington"
   ↓
   Characters: ['W', 'a', 's', 'h', 'i', 'n', 'g', 't', 'o', 'n']
   ↓
   Character Embeddings (25d each)
   ↓
   1D CNN (filters=30, kernel_size=3)
   ↓
   Max Pooling → Single vector per word (30d)
   ```

2. **Full Architecture**
   ```
   Input: (words, characters_per_word)
       ↓                           ↓
   Word Embedding          Character Embedding
   (GloVe 300d)            (25d) → CNN → MaxPool (30d)
       ↓                           ↓
       └───── Concatenate (330d) ──────┘
                    ↓
           BiLSTM (256 forward + 256 backward)
                    ↓
              Dropout (0.5)
                    ↓
            Linear (num_tags)
                    ↓
               CRF Layer
   ```

3. **Implementation (PyTorch)**
   ```python
   import torch
   import torch.nn as nn
   from torchcrf import CRF

   class CharCNN_BiLSTM_CRF(nn.Module):
       def __init__(self, vocab_size, char_vocab_size, word_emb_dim,
                    char_emb_dim, char_hidden_dim, lstm_hidden_dim, num_tags):
           super().__init__()

           # Word embeddings
           self.word_embedding = nn.Embedding(vocab_size, word_emb_dim)

           # Character embeddings + CNN
           self.char_embedding = nn.Embedding(char_vocab_size, char_emb_dim)
           self.char_cnn = nn.Conv1d(char_emb_dim, char_hidden_dim,
                                      kernel_size=3, padding=1)

           # BiLSTM
           self.lstm = nn.LSTM(word_emb_dim + char_hidden_dim,
                               lstm_hidden_dim,
                               batch_first=True,
                               bidirectional=True)

           # Output layers
           self.dropout = nn.Dropout(0.5)
           self.fc = nn.Linear(lstm_hidden_dim * 2, num_tags)

           # CRF layer
           self.crf = CRF(num_tags, batch_first=True)

       def char_repr(self, chars):
           # chars: (batch, seq_len, max_char_len)
           batch_size, seq_len, max_char_len = chars.size()

           # Reshape to (batch * seq_len, max_char_len)
           chars = chars.view(-1, max_char_len)

           # Character embeddings
           char_embeds = self.char_embedding(chars)  # (batch*seq, max_char, emb_dim)
           char_embeds = char_embeds.transpose(1, 2)  # (batch*seq, emb_dim, max_char)

           # CNN
           char_cnn_out = self.char_cnn(char_embeds)  # (batch*seq, hidden, max_char)
           char_features = torch.max(char_cnn_out, dim=2)[0]  # Max pooling

           # Reshape back
           char_features = char_features.view(batch_size, seq_len, -1)
           return char_features

       def forward(self, words, chars, tags=None, mask=None):
           # Word embeddings
           word_embeds = self.word_embedding(words)

           # Character features
           char_features = self.char_repr(chars)

           # Concatenate
           combined = torch.cat([word_embeds, char_features], dim=-1)

           # BiLSTM
           lstm_out, _ = self.lstm(combined)
           lstm_out = self.dropout(lstm_out)

           # Emission scores
           emissions = self.fc(lstm_out)

           # CRF
           if tags is not None:
               # Training: compute loss
               loss = -self.crf(emissions, tags, mask=mask)
               return loss
           else:
               # Inference: decode
               return self.crf.decode(emissions, mask=mask)
   ```

4. **Libraries**
   - **PyTorch** + **pytorch-crf**: `pip install pytorch-crf`
   - Alternative: TensorFlow + tensorflow-addons (CRF)
   - GitHub: https://github.com/kmkurn/pytorch-crf

5. **Training**
   - **Loss:** CRF negative log-likelihood (computed by CRF layer)
   - **Optimizer:** Adam (lr=0.001)
   - **Gradient clipping:** `torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)`
   - **Batch size:** 16-32
   - **Epochs:** 20-30
   - **Early stopping:** Monitor validation F1

6. **Data Preparation**
   ```python
   # Character vocabulary
   char2idx = {c: i for i, c in enumerate(
       'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?:;-'
   )}
   char2idx['<PAD>'] = len(char2idx)
   char2idx['<UNK>'] = len(char2idx)

   # Convert words to character indices
   def word_to_char_ids(word, max_char_len=20):
       char_ids = [char2idx.get(c, char2idx['<UNK>']) for c in word[:max_char_len]]
       # Pad to max_char_len
       char_ids += [char2idx['<PAD>']] * (max_char_len - len(char_ids))
       return char_ids

   # For each sentence
   X_train_chars = [
       [word_to_char_ids(word) for word in sentence]
       for sentence in train_tokens
   ]
   ```

7. **Key Advantages**
   - **Handles OOV words**: Can represent unseen words via character patterns
   - **Captures morphology**: Recognizes common prefixes/suffixes
   - **CRF enforces valid BIO**: Transition constraints learned automatically
   - **Research-proven**: 90.94% F1 on CoNLL-2003 (English), 91.47% on CoNLL++
   - **Better boundary detection**: CRF prevents invalid transitions (e.g., O → I-)

8. **CRF Benefits**
   - Learns transition scores: P(B-Person | O), P(I-Person | B-Person), etc.
   - Viterbi decoding finds globally optimal tag sequence
   - Automatically prevents: O → I-, B-Person → I-Location, etc.

**Notebook:** `7_CharCNN_BiLSTM_CRF.ipynb`

**References:**
- Character-level CNN Paper: https://arxiv.org/abs/1508.01991
- BiLSTM-CRF for NER: https://arxiv.org/abs/1508.01991
- PyTorch CRF Implementation: https://pytorch-crf.readthedocs.io/
- Tutorial: https://towardsdatascience.com/named-entity-recognition-ner-meeting-industrys-requirement-by-applying-state-of-the-art-deep-698d2b3b4ede
- Research Results: https://domino.ai/blog/named-entity-recognition-ner-challenges-and-model

---

### Model 5: Fine-tuned BERT

**Complexity:** Medium (HuggingFace makes it easy!) | **Training Required:** Yes (fine-tuning) ✅ | **Expected F1:** 90%+

#### Implementation Steps

1. **Model Selection**
   - **`bert-base-cased`** (recommended for NER)
     - 12 layers, 768 hidden, 12 attention heads
     - 110M parameters
     - Preserves capitalization (important for NER!)
   - **Alternative:** `bert-large-cased` (340M params, if GPU memory allows)
   - **Why cased?** Capitalization is crucial for entity recognition
     - "Apple" (company) vs "apple" (fruit)
     - Proper nouns vs common nouns

2. **Tokenization Challenge: Subword Alignment**

   BERT uses WordPiece tokenization, which splits words into subwords:
   ```
   Word:  "Washington"
   BERT:  ["Wash", "##ing", "##ton"]

   Problem: How to assign labels?
   ```

   **Solution:** Label first subword token, ignore rest
   ```
   Original:
   Tokens: ["Washington", "visited"]
   Labels: ["B-Person", "O"]

   After BERT tokenization:
   Tokens: ["Wash", "##ing", "##ton", "visited"]
   Labels: ["B-Person", -100, -100, "O"]

   # -100 = ignore in loss computation
   ```

3. **Architecture**
   ```
   Input Token IDs
       ↓
   BERT Encoder (12 layers, fine-tuned)
       ↓
   Hidden States (768d per token)
       ↓
   Dropout (0.1)
       ↓
   Linear Layer (num_tags)
       ↓
   Softmax → Tag probabilities
   ```

4. **Libraries**
   - **Transformers** (HuggingFace): `pip install transformers`
   - **Datasets** (HuggingFace): `pip install datasets`
   - **PyTorch** or **TensorFlow**
   - **Accelerate** (optional, for multi-GPU): `pip install accelerate`

5. **Implementation (HuggingFace)**
   ```python
   from transformers import (
       AutoTokenizer,
       AutoModelForTokenClassification,
       TrainingArguments,
       Trainer,
       DataCollatorForTokenClassification
   )

   # Load tokenizer and model
   model_name = "bert-base-cased"
   tokenizer = AutoTokenizer.from_pretrained(model_name)

   # Number of tags
   label_list = ['O', 'B-Politician', 'I-Politician', ...]  # All your tags
   num_tags = len(label_list)
   label2id = {label: i for i, label in enumerate(label_list)}
   id2label = {i: label for label, i in label2id.items()}

   model = AutoModelForTokenClassification.from_pretrained(
       model_name,
       num_labels=num_tags,
       id2label=id2label,
       label2id=label2id
   )

   # Tokenize and align labels
   def tokenize_and_align_labels(examples):
       tokenized_inputs = tokenizer(
           examples["tokens"],
           truncation=True,
           is_split_into_words=True,
           padding=True,
           max_length=128
       )

       labels = []
       for i, label in enumerate(examples["ner_tags"]):
           word_ids = tokenized_inputs.word_ids(batch_index=i)
           label_ids = []
           previous_word_idx = None

           for word_idx in word_ids:
               if word_idx is None:
                   # Special tokens (CLS, SEP, PAD)
                   label_ids.append(-100)
               elif word_idx != previous_word_idx:
                   # First token of word
                   label_ids.append(label2id[label[word_idx]])
               else:
                   # Subsequent tokens of same word
                   label_ids.append(-100)
               previous_word_idx = word_idx

           labels.append(label_ids)

       tokenized_inputs["labels"] = labels
       return tokenized_inputs

   # Apply to dataset
   tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True)
   tokenized_val = val_dataset.map(tokenize_and_align_labels, batched=True)

   # Training arguments
   training_args = TrainingArguments(
       output_dir="./bert_ner",
       learning_rate=5e-5,
       per_device_train_batch_size=16,
       per_device_eval_batch_size=16,
       num_train_epochs=3,
       weight_decay=0.01,
       evaluation_strategy="epoch",
       save_strategy="epoch",
       load_best_model_at_end=True,
       warmup_steps=500,
       fp16=True,  # Mixed precision for speed
       logging_steps=100
   )

   # Data collator
   data_collator = DataCollatorForTokenClassification(tokenizer)

   # Trainer
   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=tokenized_train,
       eval_dataset=tokenized_val,
       tokenizer=tokenizer,
       data_collator=data_collator,
       compute_metrics=compute_metrics  # Custom function using utils.py
   )

   # Train!
   trainer.train()
   ```

6. **Training Configuration**
   - **Learning rate:** 5e-5 (BERT paper recommendation)
     - Too high: model forgets pre-training
     - Too low: slow convergence
   - **Batch size:** 16 (adjust based on GPU memory)
   - **Epochs:** 3-4 (BERT overfits quickly!)
   - **Warm-up:** 500 steps (gradual learning rate increase)
   - **Weight decay:** 0.01 (L2 regularization)
   - **Mixed precision (fp16):** 2x faster training, half memory usage

7. **Hyperparameters**
   | Parameter | Options | Recommended |
   |-----------|---------|-------------|
   | Learning rate | 2e-5, 3e-5, 5e-5 | 5e-5 |
   | Batch size | 8, 16, 32 | 16 |
   | Epochs | 3, 4, 5 | 3 |
   | Max length | 128, 256, 512 | 128 |
   | Warmup steps | 0, 500, 1000 | 500 |

8. **Compute Metrics (Entity-Span F1)**
   ```python
   from utils import evaluate_entity_spans

   def compute_metrics(p):
       predictions, labels = p
       predictions = np.argmax(predictions, axis=2)

       # Remove ignored index (-100) and convert to tags
       true_tags = []
       pred_tags = []

       for pred_seq, label_seq in zip(predictions, labels):
           true_seq = []
           pred_seq_clean = []

           for pred, label in zip(pred_seq, label_seq):
               if label != -100:
                   true_seq.append(id2label[label])
                   pred_seq_clean.append(id2label[pred])

           true_tags.append(true_seq)
           pred_tags.append(pred_seq_clean)

       # Use our entity-span evaluation
       results = evaluate_entity_spans(true_tags, pred_tags)
       return {
           "precision": results["precision"],
           "recall": results["recall"],
           "f1": results["f1"]
       }
   ```

9. **Key Advantages**
   - **State-of-the-art contextual embeddings**: Each token representation depends on entire sentence
   - **Transfer learning**: Massive pre-training on 3.3B words (BookCorpus + Wikipedia)
   - **Bidirectional context**: Attention mechanism sees both left and right context
   - **Easy implementation**: HuggingFace abstracts complexity
   - **Strong baseline**: ~90% F1 with minimal tuning

10. **Common Issues & Solutions**
    - **CUDA OOM:** Reduce batch size to 8, or use gradient accumulation
    - **Overfitting:** BERT overfits quickly, use early stopping
    - **Slow training:** Use fp16 mixed precision, reduce max_length
    - **Subword alignment:** Ensure -100 labels for non-first subword tokens

**Notebook:** `8_BERT_Finetuning.ipynb`

**References:**
- BERT Paper: https://arxiv.org/abs/1810.04805
- HuggingFace BERT NER Tutorial: https://huggingface.co/docs/transformers/tasks/token_classification
- Token Classification Guide: https://huggingface.co/course/chapter7/2
- Fine-tuning BERT for NER: https://www.freecodecamp.org/news/getting-started-with-ner-models-using-huggingface/
- BERT Best Practices: https://mccormickml.com/2019/07/22/BERT-fine-tuning/

---

### Model 6: BERT + CRF Hybrid

**Complexity:** High | **Training Required:** Yes (fine-tuning) ✅ | **Expected F1:** 90-92%

#### Implementation Steps

1. **Why Add CRF to BERT?**

   **BERT alone:**
   - Predicts each token independently
   - Can produce invalid BIO sequences: O → I-Person
   - No explicit transition modeling

   **BERT + CRF:**
   - CRF layer learns transition constraints
   - Viterbi decoding ensures valid sequences
   - Better boundary detection
   - **+1-2% F1 improvement** in research

2. **Architecture**
   ```
   Input Token IDs
       ↓
   BERT Encoder (fine-tuned)
       ↓
   Hidden States (768d per token)
       ↓
   Dropout (0.1)
       ↓
   Linear Layer → Emission Scores (num_tags)
       ↓
   CRF Layer (learns transition matrix)
       ↓
   Viterbi Decoding → Tag Sequence
   ```

3. **Implementation (PyTorch + pytorch-crf)**
   ```python
   from transformers import BertModel, BertConfig
   from torchcrf import CRF
   import torch.nn as nn

   class BertCRF(nn.Module):
       def __init__(self, bert_model_name, num_tags):
           super().__init__()

           # BERT encoder
           self.bert = BertModel.from_pretrained(bert_model_name)
           self.dropout = nn.Dropout(0.1)

           # Emission layer (replaces softmax)
           self.classifier = nn.Linear(self.bert.config.hidden_size, num_tags)

           # CRF layer
           self.crf = CRF(num_tags, batch_first=True)

       def forward(self, input_ids, attention_mask, labels=None):
           # BERT forward pass
           outputs = self.bert(
               input_ids=input_ids,
               attention_mask=attention_mask
           )
           sequence_output = outputs[0]  # (batch, seq_len, hidden_size)
           sequence_output = self.dropout(sequence_output)

           # Emission scores
           emissions = self.classifier(sequence_output)  # (batch, seq_len, num_tags)

           if labels is not None:
               # Training: compute CRF loss
               # Convert attention_mask to byte mask for CRF
               mask = attention_mask.byte()

               # Filter out -100 labels (ignored tokens)
               labels_masked = labels.clone()
               labels_masked[labels == -100] = 0  # CRF requires valid indices

               loss = -self.crf(emissions, labels_masked, mask=mask, reduction='mean')
               return loss
           else:
               # Inference: CRF decoding
               mask = attention_mask.byte()
               predictions = self.crf.decode(emissions, mask=mask)
               return predictions
   ```

4. **Training**
   ```python
   from transformers import AdamW, get_linear_schedule_with_warmup

   model = BertCRF('bert-base-cased', num_tags=len(label_list))

   # Optimizer
   optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

   # Learning rate scheduler
   total_steps = len(train_dataloader) * num_epochs
   scheduler = get_linear_schedule_with_warmup(
       optimizer,
       num_warmup_steps=500,
       num_training_steps=total_steps
   )

   # Training loop
   for epoch in range(num_epochs):
       model.train()
       for batch in train_dataloader:
           input_ids = batch['input_ids']
           attention_mask = batch['attention_mask']
           labels = batch['labels']

           # Forward pass
           loss = model(input_ids, attention_mask, labels)

           # Backward pass
           loss.backward()
           torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
           optimizer.step()
           scheduler.step()
           optimizer.zero_grad()
   ```

5. **Libraries**
   - `transformers` (HuggingFace)
   - `pytorch-crf`: `pip install pytorch-crf`
   - PyTorch

6. **Key Implementation Details**

   **Handling -100 labels:**
   - BERT uses -100 to ignore subword tokens
   - CRF requires valid label indices
   - Solution: Mask out -100 labels using attention mask

   **Attention mask as CRF mask:**
   - CRF needs to know valid tokens vs padding
   - Use BERT's attention_mask (1 = valid, 0 = padding)

7. **Hyperparameters**
   - Same as BERT fine-tuning (Model 5)
   - May benefit from slightly more epochs (4-5)
   - CRF adds ~1M parameters (transition matrix)

8. **Key Advantages**
   - **Better boundary detection**: CRF prevents partial entities
   - **Valid BIO sequences**: Automatically enforced
   - **Improved consistency**: Global sequence optimization
   - **Research-backed**: Multiple papers show +1-2% F1 improvement
   - **Best of both worlds**: BERT's representations + CRF's structure

9. **CRF Transition Matrix**
   ```
   After training, CRF learns transitions like:

   High probability:
   - O → B-Person (start of entity)
   - B-Person → I-Person (continue entity)
   - I-Person → O (end entity)

   Low/zero probability:
   - O → I-Person (invalid: inside without begin)
   - B-Person → I-Location (invalid: type change)
   - I-Person → I-Location (invalid: type change)
   ```

**Notebook:** `9_BERT_CRF.ipynb`

**References:**
- BERT + CRF Paper: https://arxiv.org/abs/1902.03006
- PyTorch CRF: https://pytorch-crf.readthedocs.io/
- Implementation Guide: https://towardsdatascience.com/bert-crf-for-named-entity-recognition-ner-c0a71a8f5c3e
- CRF Layer Benefits: https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/

---

### Model 7: Attention-Based BiLSTM-CRF

**Complexity:** Very High | **Training Required:** Yes (deep learning) ✅ | **Expected F1:** 92-93%

#### Implementation Steps

1. **Why Attention for NER?**

   **Problem:** Standard BiLSTM processes sentence sequentially
   - May miss long-range dependencies
   - "Barack Obama, the 44th president, visited Paris" - need to connect "Barack Obama" with "president"

   **Solution:** Self-attention mechanism
   - Each token attends to all other tokens in sentence/document
   - Learn which context words are relevant
   - Captures entity co-reference and relationships

2. **Architecture**
   ```
   Input: (words, characters)
       ↓
   Word Embedding (GloVe 300d) + Character CNN (30d)
       ↓
   Concatenate → (330d)
       ↓
   BiLSTM (256 forward + 256 backward = 512d)
       ↓
   Self-Attention Layer
   - Query/Key/Value projections
   - Scaled dot-product attention
   - Multi-head attention (optional)
       ↓
   Context-aware representations (512d)
       ↓
   Linear (num_tags)
       ↓
   CRF Layer → Predictions
   ```

3. **Self-Attention Mechanism**
   ```python
   class SelfAttention(nn.Module):
       def __init__(self, hidden_dim, num_heads=8):
           super().__init__()
           self.hidden_dim = hidden_dim
           self.num_heads = num_heads
           self.head_dim = hidden_dim // num_heads

           assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

           # Linear projections
           self.query = nn.Linear(hidden_dim, hidden_dim)
           self.key = nn.Linear(hidden_dim, hidden_dim)
           self.value = nn.Linear(hidden_dim, hidden_dim)

           self.fc_out = nn.Linear(hidden_dim, hidden_dim)
           self.dropout = nn.Dropout(0.1)

       def forward(self, x, mask=None):
           batch_size, seq_len, hidden_dim = x.size()

           # Linear projections and split into multiple heads
           Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
           K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
           V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

           # Transpose for attention computation
           Q = Q.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
           K = K.transpose(1, 2)
           V = V.transpose(1, 2)

           # Scaled dot-product attention
           scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

           # Apply mask (for padding)
           if mask is not None:
               mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
               scores = scores.masked_fill(mask == 0, -1e9)

           # Attention weights
           attention = torch.softmax(scores, dim=-1)
           attention = self.dropout(attention)

           # Apply attention to values
           context = torch.matmul(attention, V)  # (batch, heads, seq_len, head_dim)

           # Concatenate heads
           context = context.transpose(1, 2).contiguous()
           context = context.view(batch_size, seq_len, hidden_dim)

           # Final linear projection
           output = self.fc_out(context)
           return output, attention
   ```

4. **Full Model Implementation**
   ```python
   class AttentionBiLSTMCRF(nn.Module):
       def __init__(self, vocab_size, char_vocab_size, word_emb_dim, char_emb_dim,
                    char_hidden_dim, lstm_hidden_dim, num_tags, num_heads=8):
           super().__init__()

           # Embeddings (same as Model 4)
           self.word_embedding = nn.Embedding(vocab_size, word_emb_dim)
           self.char_embedding = nn.Embedding(char_vocab_size, char_emb_dim)
           self.char_cnn = nn.Conv1d(char_emb_dim, char_hidden_dim, kernel_size=3, padding=1)

           # BiLSTM
           self.lstm = nn.LSTM(
               word_emb_dim + char_hidden_dim,
               lstm_hidden_dim,
               batch_first=True,
               bidirectional=True
           )

           # Self-Attention
           self.attention = SelfAttention(lstm_hidden_dim * 2, num_heads=num_heads)

           # Output layers
           self.dropout = nn.Dropout(0.5)
           self.fc = nn.Linear(lstm_hidden_dim * 2, num_tags)

           # CRF
           self.crf = CRF(num_tags, batch_first=True)

       def char_repr(self, chars):
           # Same as Model 4
           ...

       def forward(self, words, chars, tags=None, mask=None):
           # Embeddings
           word_embeds = self.word_embedding(words)
           char_features = self.char_repr(chars)
           combined = torch.cat([word_embeds, char_features], dim=-1)

           # BiLSTM
           lstm_out, _ = self.lstm(combined)

           # Self-Attention
           attention_out, attention_weights = self.attention(lstm_out, mask=mask)

           # Residual connection (optional but recommended)
           attention_out = attention_out + lstm_out

           attention_out = self.dropout(attention_out)

           # Emission scores
           emissions = self.fc(attention_out)

           # CRF
           if tags is not None:
               loss = -self.crf(emissions, tags, mask=mask)
               return loss
           else:
               return self.crf.decode(emissions, mask=mask)
   ```

5. **Training**
   - **Similar to Model 4** but with attention
   - **Learning rate:** 0.001 (may need to be lower: 0.0005)
   - **Epochs:** 25-35 (attention adds complexity, may need more training)
   - **Gradient clipping:** 5.0
   - **Batch size:** 16 (attention increases memory usage)
   - **Attention heads:** 4, 8 (experiment)

6. **Hyperparameters**
   | Parameter | Value |
   |-----------|-------|
   | Word embedding | GloVe 300d |
   | Char embedding | 25d |
   | Char CNN filters | 30 |
   | LSTM hidden | 256 (bidirectional = 512 total) |
   | Attention heads | 8 |
   | Dropout | 0.5 |
   | Learning rate | 0.0005 |
   | Batch size | 16 |

7. **Key Advantages**
   - **Document-level context**: Can relate entities across entire sentence
   - **Long-range dependencies**: Attention doesn't suffer from vanishing gradients
   - **Interpretability**: Attention weights show which words model focuses on
   - **Research SOTA**: 92.57% F1 on BioCreative V CDR dataset
   - **Entity co-reference**: Helps with repeated entities in long text

8. **Attention Benefits for NER**
   ```
   Example: "Apple Inc announced that Apple will release..."

   Without attention:
   - Second "Apple" may be tagged differently
   - Limited context from first mention

   With attention:
   - Second "Apple" attends to first "Apple Inc"
   - Consistent tagging across mentions
   - Model learns entity co-reference patterns
   ```

9. **Computational Cost**
   - **Memory:** O(n²) for attention scores (n = sequence length)
   - **Training time:** ~1.5-2x slower than BiLSTM-CRF
   - **Inference:** Still fast enough for production
   - **Tip:** Use batch size 16 instead of 32

**Notebook:** `10_Attention_BiLSTM_CRF.ipynb`

**References:**
- Attention Mechanism Paper: https://arxiv.org/abs/1706.03762 (Transformer)
- Attention for NER: https://academic.oup.com/bioinformatics/article/34/8/1381/4657076
- Research Results (92.57% F1): https://academic.oup.com/bioinformatics/article/34/8/1381/4657076
- Self-Attention Explained: https://jalammar.github.io/illustrated-transformer/
- Multi-Head Attention: https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853

---

### Model 8: LLM + Basic Prompting

**Complexity:** Low (implementation) | **Training Required:** No ❌ | **Expected F1:** 65-73%

#### Implementation Steps

1. **LLM Selection**

   **Option A: OpenAI GPT-4**
   - Best performance
   - API: `openai` Python package
   - Cost: ~$0.03 per 1K tokens (input) + $0.06 per 1K tokens (output)
   - Estimated cost for 10K validation samples: ~$20-30

   **Option B: OpenAI GPT-3.5-turbo**
   - Cheaper alternative
   - API: `openai` Python package
   - Cost: ~$0.001 per 1K tokens
   - Estimated cost: ~$2-5
   - Performance: ~20% worse than GPT-4

   **Option C: Anthropic Claude**
   - Good alternative to GPT-4
   - API: `anthropic` Python package
   - Similar pricing to GPT-4

   **Option D: Open-source LLMs**
   - Llama 2, Mistral, etc.
   - Free but requires local GPU
   - Performance varies

2. **Prompt Design (Basic)**

   **Simple Instruction Prompt:**
   ```
   Task: Named Entity Recognition (NER) using BIO tagging scheme.

   Entity Types and Tags:
   - B-Politician, I-Politician: Politician names
   - B-Artist, I-Artist: Artist names
   - B-Facility, I-Facility: Buildings, venues, landmarks
   - B-HumanSettlement, I-HumanSettlement: Cities, towns, villages
   - B-OtherPER, I-OtherPER: Other person names
   - B-PublicCorp, I-PublicCorp: Public corporations
   - B-ORG, I-ORG: Other organizations
   - O: Not an entity

   Instructions:
   - Tag each token with exactly one BIO tag
   - B- marks the beginning of an entity
   - I- marks inside/continuation of an entity
   - O marks tokens that are not entities

   Input tokens: ["Barack", "Obama", "visited", "Paris", "."]

   Output the tags as a JSON list, one tag per token:
   ```

3. **Implementation (OpenAI)**
   ```python
   import openai
   import json
   from time import sleep

   openai.api_key = "your-api-key"

   def create_prompt(tokens):
       """Create basic NER prompt"""
       prompt = f'''Task: Named Entity Recognition (NER) using BIO tagging scheme.

   Entity Types:
   - B-Politician, I-Politician: Politicians
   - B-Artist, I-Artist: Artists
   - B-Facility, I-Facility: Buildings, venues
   - B-HumanSettlement, I-HumanSettlement: Cities, towns
   - B-OtherPER, I-OtherPER: Other person names
   - B-PublicCorp, I-PublicCorp: Public corporations
   - B-ORG, I-ORG: Organizations
   - O: Not an entity

   Tag each token with exactly one BIO tag.

   Input tokens: {json.dumps(tokens)}

   Output ONLY a JSON list of tags, one per token. Example format:
   ["B-Politician", "I-Politician", "O", "B-HumanSettlement", "O"]

   Output:'''
       return prompt

   def predict_ner_basic(tokens, model="gpt-4"):
       """Predict NER tags using basic prompting"""
       prompt = create_prompt(tokens)

       try:
           response = openai.ChatCompletion.create(
               model=model,
               messages=[
                   {"role": "system", "content": "You are an expert NER system. Output only valid JSON."},
                   {"role": "user", "content": prompt}
               ],
               temperature=0,  # Deterministic
               max_tokens=500
           )

           # Parse response
           output = response.choices[0].message.content.strip()

           # Try to parse as JSON
           tags = json.loads(output)

           # Validate length
           if len(tags) != len(tokens):
               print(f"Warning: Length mismatch. Got {len(tags)}, expected {len(tokens)}")
               # Pad or truncate
               if len(tags) < len(tokens):
                   tags += ['O'] * (len(tokens) - len(tags))
               else:
                   tags = tags[:len(tokens)]

           return tags

       except json.JSONDecodeError:
           print(f"Failed to parse JSON: {output}")
           # Fallback: all O
           return ['O'] * len(tokens)

       except Exception as e:
           print(f"Error: {e}")
           return ['O'] * len(tokens)

   # Batch prediction with rate limiting
   def predict_batch(tokens_list, model="gpt-4", delay=0.5):
       """Predict for multiple sentences"""
       predictions = []

       for i, tokens in enumerate(tokens_list):
           if i % 10 == 0:
               print(f"Processed {i}/{len(tokens_list)}")

           pred_tags = predict_ner_basic(tokens, model=model)
           predictions.append(pred_tags)

           # Rate limiting
           sleep(delay)

       return predictions
   ```

4. **Implementation (Anthropic Claude)**
   ```python
   import anthropic

   client = anthropic.Anthropic(api_key="your-api-key")

   def predict_ner_claude(tokens):
       prompt = create_prompt(tokens)

       message = client.messages.create(
           model="claude-3-5-sonnet-20241022",
           max_tokens=500,
           temperature=0,
           messages=[
               {"role": "user", "content": prompt}
           ]
       )

       output = message.content[0].text
       tags = json.loads(output)
       return tags
   ```

5. **Challenges & Solutions**

   **Problem 1: Output Formatting**
   - LLM may not follow JSON format exactly
   - May add explanations, formatting

   **Solution:**
   - Explicit JSON instructions in prompt
   - Parse with error handling
   - Fallback to regex extraction

   **Problem 2: Token Misalignment**
   - LLM may merge/split tokens
   - Output length ≠ input length

   **Solution:**
   - Emphasize "one tag per token"
   - Validate and pad/truncate as needed

   **Problem 3: Invalid BIO Sequences**
   - LLM may generate I- without B-

   **Solution:**
   - Post-processing to fix sequences (use code from EDA)
   - Or add explicit BIO rules to prompt

6. **Post-Processing**
   ```python
   def fix_bio_sequences(tags):
       """Fix invalid BIO sequences"""
       fixed_tags = []
       prev_tag = 'O'

       for tag in tags:
           if tag.startswith('I-'):
               entity_type = tag[2:]
               # Check if previous was B- or I- of same type
               if not (prev_tag == f'B-{entity_type}' or prev_tag == f'I-{entity_type}'):
                   # Fix: change I- to B-
                   tag = f'B-{entity_type}'

           fixed_tags.append(tag)
           prev_tag = tag

       return fixed_tags
   ```

7. **Cost Estimation**
   ```python
   # Approximate tokens per sample
   avg_tokens_per_sample = 200  # input prompt + output
   num_samples = 10000

   # GPT-4
   gpt4_cost = (num_samples * avg_tokens_per_sample * 0.03) / 1000
   print(f"GPT-4 estimated cost: ${gpt4_cost:.2f}")

   # GPT-3.5
   gpt35_cost = (num_samples * avg_tokens_per_sample * 0.001) / 1000
   print(f"GPT-3.5 estimated cost: ${gpt35_cost:.2f}")
   ```

8. **Key Characteristics**
   - **Lower F1 than trained models**: 65-73% vs 90%+
   - **No training required**: Zero-shot capability
   - **Slow inference**: API latency + rate limits
   - **Cost**: Can be expensive for large datasets
   - **Good for**: Quick baseline, comparison

9. **Performance by LLM**
   | Model | Expected F1 | Cost |
   |-------|-------------|------|
   | GPT-4 | ~70-73% | $$$ |
   | GPT-3.5 | ~65-68% | $ |
   | Claude 3 | ~70-73% | $$$ |
   | Llama 2 70B | ~60-65% | Free (local) |

**Notebook:** `11_LLM_Basic_Prompting.ipynb`

**References:**
- OpenAI API: https://platform.openai.com/docs/api-reference
- Anthropic Claude: https://docs.anthropic.com/
- Prompting Guide: https://www.promptingguide.ai/
- LLM for NER Research: https://arxiv.org/abs/2408.15796 (CoT prompting: 73% F1)

---

### Model 9: LLM + Few-Shot Prompting

**Complexity:** Medium | **Training Required:** No ❌ | **Expected F1:** 85-95%

#### Implementation Steps

1. **Few-Shot Learning Concept**

   Instead of just instructions, provide **examples** in the prompt:
   ```
   Here are some examples:

   Example 1:
   Tokens: ["Barack", "Obama", "visited", "Paris"]
   Tags: ["B-Politician", "I-Politician", "O", "B-HumanSettlement"]

   Example 2:
   Tokens: ["Apple", "Inc", "released", "iPhone"]
   Tags: ["B-PublicCorp", "I-PublicCorp", "O", "O"]

   Now tag this:
   Tokens: [your input]
   ```

   **Why it works:**
   - LLMs learn patterns from examples
   - Better understanding of task format
   - Improved accuracy: +10-20% F1

2. **Example Selection Strategy**

   **Option A: Random Selection**
   - Pick 3-5 random examples from training data
   - Simple but effective

   **Option B: Representative Selection**
   - Choose examples covering all entity types
   - Ensure diversity in sentence structure

   **Option C: Similar Examples (k-NN)**
   - Find examples most similar to input
   - Use sentence embeddings for similarity
   - Best performance but more complex

3. **PromptNER Technique**

   **Research finding:** Asking LLM to explain WHY improves accuracy

   **Enhanced prompt with explanations:**
   ```
   Example:
   Tokens: ["Barack", "Obama", "visited", "Paris"]
   Tags: ["B-Politician", "I-Politician", "O", "B-HumanSettlement"]
   Explanation: "Barack Obama" is a full name of a politician (2 tokens: B-Politician, I-Politician). "visited" is a verb, not an entity. "Paris" is a city (B-HumanSettlement).

   Benefits:
   - +4% F1 on CoNLL
   - +9% F1 on GENIA
   - +24% F1 on TweetNER
   ```

4. **Implementation (Few-Shot + PromptNER)**
   ```python
   def create_fewshot_prompt(tokens, examples):
       """
       Create few-shot prompt with explanations

       Args:
           tokens: List of tokens to tag
           examples: List of (tokens, tags, explanation) tuples
       """

       prompt = """Task: Named Entity Recognition (NER) using BIO tagging.

   Entity Types:
   - B-Politician, I-Politician: Politician names
   - B-Artist, I-Artist: Artist names
   - B-Facility, I-Facility: Buildings, venues, landmarks
   - B-HumanSettlement, I-HumanSettlement: Cities, towns, villages
   - B-OtherPER, I-OtherPER: Other person names
   - B-PublicCorp, I-PublicCorp: Public corporations
   - B-ORG, I-ORG: Other organizations
   - O: Not an entity

   Here are some examples:

"""

       # Add few-shot examples
       for i, (ex_tokens, ex_tags, ex_explanation) in enumerate(examples, 1):
           prompt += f"Example {i}:\n"
           prompt += f"Tokens: {json.dumps(ex_tokens)}\n"
           prompt += f"Tags: {json.dumps(ex_tags)}\n"
           prompt += f"Explanation: {ex_explanation}\n\n"

       # Add current input
       prompt += f"Now tag this sentence:\n"
       prompt += f"Tokens: {json.dumps(tokens)}\n\n"
       prompt += f"Provide:\n"
       prompt += f"1. Tags: A JSON list of BIO tags\n"
       prompt += f"2. Explanation: Brief explanation of your tagging\n\n"
       prompt += f"Output format:\n"
       prompt += f'{{"tags": [...], "explanation": "..."}}\n'

       return prompt

   def select_representative_examples(train_data, num_examples=5):
       """
       Select diverse examples covering all entity types
       """
       from collections import defaultdict

       # Group examples by entity types present
       examples_by_type = defaultdict(list)

       for sample in train_data:
           tokens = sample['tokens']
           tags = sample['ner_tags']

           # Get entity types in this sample
           entity_types = set()
           for tag in tags:
               if tag.startswith('B-'):
                   entity_types.add(tag[2:])

           for entity_type in entity_types:
               examples_by_type[entity_type].append(sample)

       # Select one example per entity type (up to num_examples)
       selected = []
       for entity_type, samples in examples_by_type.items():
           if len(selected) >= num_examples:
               break
           # Pick shortest example (easier to fit in prompt)
           samples.sort(key=lambda x: len(x['tokens']))
           selected.append(samples[0])

       # Create explanations
       examples_with_explanations = []
       for sample in selected:
           tokens = sample['tokens']
           tags = sample['ner_tags']
           explanation = generate_explanation(tokens, tags)
           examples_with_explanations.append((tokens, tags, explanation))

       return examples_with_explanations

   def generate_explanation(tokens, tags):
       """
       Generate explanation for the tagging
       """
       from utils import extract_entities

       entities = extract_entities(tokens, tags)

       if not entities:
           return "No entities in this sentence."

       explanations = []
       for entity_text, entity_type, start, end in entities:
           span_len = end - start + 1
           if span_len == 1:
               explanations.append(f'"{entity_text}" is a {entity_type}')
           else:
               explanations.append(f'"{entity_text}" is a {entity_type} (spans {span_len} tokens)')

       return ". ".join(explanations) + "."

   def predict_ner_fewshot(tokens, examples, model="gpt-4"):
       """
       Predict NER tags using few-shot prompting
       """
       prompt = create_fewshot_prompt(tokens, examples)

       try:
           response = openai.ChatCompletion.create(
               model=model,
               messages=[
                   {"role": "system", "content": "You are an expert NER system."},
                   {"role": "user", "content": prompt}
               ],
               temperature=0,
               max_tokens=1000  # More tokens for explanation
           )

           output = response.choices[0].message.content.strip()

           # Try to parse JSON output
           result = json.loads(output)
           tags = result.get('tags', [])
           explanation = result.get('explanation', '')

           # Validate and fix if needed
           if len(tags) != len(tokens):
               tags = fix_length(tags, tokens)

           tags = fix_bio_sequences(tags)

           return tags, explanation

       except Exception as e:
           print(f"Error: {e}")
           return ['O'] * len(tokens), ""
   ```

5. **Chain-of-Thought (CoT) Prompting**

   **Alternative approach:** Ask LLM to reason step-by-step

   ```python
   def create_cot_prompt(tokens):
       prompt = f"""Task: Named Entity Recognition

   Tokens: {json.dumps(tokens)}

   Step 1: Identify potential entities in the sentence.
   Step 2: Classify each entity by type.
   Step 3: Assign BIO tags to each token.

   Let's work through this step by step:

   Step 1 - Entities:
   Step 2 - Classification:
   Step 3 - BIO Tags:

   Final answer (JSON list):"""
       return prompt
   ```

   **Research:** CoT achieves 73% F1 vs 65% without CoT

6. **Self-Consistency (Advanced)**

   **Technique:** Sample multiple predictions and vote

   ```python
   def predict_with_self_consistency(tokens, examples, num_samples=5):
       """
       Generate multiple predictions and take majority vote
       """
       all_predictions = []

       for _ in range(num_samples):
           # Use temperature > 0 for diversity
           response = openai.ChatCompletion.create(
               model="gpt-4",
               messages=[...],
               temperature=0.7,  # Sampling
               max_tokens=1000
           )

           tags = parse_response(response)
           all_predictions.append(tags)

       # Majority voting per token
       final_tags = []
       for i in range(len(tokens)):
           token_tags = [pred[i] for pred in all_predictions]
           # Most common tag
           final_tag = max(set(token_tags), key=token_tags.count)
           final_tags.append(final_tag)

       return final_tags
   ```

   **Trade-off:** 5x more API calls, but +2-3% F1

7. **Hyperparameters**
   - **Number of examples:** 3, 5, 7 (more = better but longer prompt)
   - **Temperature:** 0 (deterministic) or 0.7 (with self-consistency)
   - **Max tokens:** 1000-2000 (need space for explanation)
   - **Model:** GPT-4 > GPT-3.5 (significant difference)

8. **Performance Expectations**
   | Technique | Expected F1 | Improvement |
   |-----------|-------------|-------------|
   | Basic prompting | 65-73% | Baseline |
   | Few-shot (3 examples) | 75-82% | +10-15% |
   | Few-shot (5 examples) | 80-85% | +15-20% |
   | PromptNER (with explanation) | 85-90% | +20-25% |
   | GPT-4o + ensemble | 95% | +30% |

9. **Cost Considerations**
   - Few-shot prompts are longer → higher cost
   - Typical prompt: 300-500 tokens
   - Self-consistency: 5x cost
   - Budget accordingly!

10. **Example Selection Best Practices**
    - Include all entity types
    - Vary sentence lengths (short + long)
    - Include multi-word entities
    - Show edge cases if possible
    - Keep examples concise (fit in context)

**Notebook:** `12_LLM_FewShot_Prompting.ipynb`

**References:**
- PromptNER Paper: https://arxiv.org/abs/2305.15444
  - +4% F1 on CoNLL, +9% on GENIA, +24% on TweetNER
- GPT-NER Paper: https://arxiv.org/abs/2304.10428
  - GPT-4o: 49% correct spans vs GPT-3.5: 28%
- Few-Shot NER Evaluation: https://arxiv.org/abs/2408.15796
  - CoT: 73% F1 vs standard: 65% F1
- Medical NER with GPT-4: https://arxiv.org/abs/2505.08704
  - GPT-4o + prompt ensemble: 95% F1
- Few-Shot Prompting Guide: https://www.promptingguide.ai/techniques/fewshot

---

### Model 10: Ensemble

**Complexity:** Medium | **Training Required:** Uses trained models | **Expected F1:** Best overall (+1-3%)

#### Implementation Steps

1. **Ensemble Strategy for NER**

   **Why ensemble?**
   - Different models make different errors
   - Combining predictions reduces variance
   - Typically +1-3% F1 over best individual
   - More robust and reliable

2. **Model Selection for Ensemble**

   Choose 3-5 best performing models with **diversity**:

   **Good combination:**
   - BERT (transformer, contextual)
   - BiLSTM-CRF with Char-CNN (handles OOV)
   - LLM Few-Shot (external knowledge)
   - BERT+CRF (structural constraints)

   **Why diversity matters:**
   - Similar models make similar errors
   - Different architectures capture different patterns
   - Complementary strengths

3. **Ensemble Method 1: Span-Level Voting**

   **Approach:** Extract entity spans, vote on each span

   ```python
   from utils import extract_entities
   from collections import Counter

   def ensemble_span_voting(tokens, predictions_list, threshold=0.5):
       """
       Ensemble by voting on entity spans

       Args:
           tokens: List of tokens
           predictions_list: List of tag sequences from different models
           threshold: Minimum fraction of models that must agree (0.5 = majority)

       Returns:
           Final tag sequence
       """
       num_models = len(predictions_list)

       # Extract spans from each model
       all_spans = []
       for pred_tags in predictions_list:
           entities = extract_entities(tokens, pred_tags)
           spans = [(start, end, entity_type) for _, entity_type, start, end in entities]
           all_spans.append(set(spans))

       # Count votes for each span
       span_votes = Counter()
       for spans in all_spans:
           for span in spans:
               span_votes[span] += 1

       # Keep spans with enough votes
       min_votes = int(num_models * threshold)
       final_spans = {span for span, votes in span_votes.items() if votes >= min_votes}

       # Convert spans back to BIO tags
       final_tags = ['O'] * len(tokens)

       for start, end, entity_type in sorted(final_spans):
           final_tags[start] = f'B-{entity_type}'
           for i in range(start + 1, end + 1):
               final_tags[i] = f'I-{entity_type}'

       return final_tags
   ```

4. **Ensemble Method 2: Weighted Voting**

   **Approach:** Weight models by validation F1 scores

   ```python
   def ensemble_weighted_voting(tokens, predictions_list, model_weights):
       """
       Weighted voting by model performance

       Args:
           tokens: List of tokens
           predictions_list: List of tag sequences
           model_weights: List of weights (e.g., validation F1 scores)
       """
       from collections import defaultdict

       # Normalize weights
       total_weight = sum(model_weights)
       weights = [w / total_weight for w in model_weights]

       # Count weighted votes for each span
       span_votes = defaultdict(float)

       for pred_tags, weight in zip(predictions_list, weights):
           entities = extract_entities(tokens, pred_tags)
           for _, entity_type, start, end in entities:
               span = (start, end, entity_type)
               span_votes[span] += weight

       # Keep spans with weight >= threshold
       threshold = 0.5  # Majority of total weight
       final_spans = {span for span, weight in span_votes.items() if weight >= threshold}

       # Convert to BIO tags
       final_tags = ['O'] * len(tokens)
       for start, end, entity_type in sorted(final_spans):
           final_tags[start] = f'B-{entity_type}'
           for i in range(start + 1, end + 1):
               final_tags[i] = f'I-{entity_type}'

       return final_tags
   ```

5. **Ensemble Method 3: Token-Level Voting**

   **Approach:** Vote on each token independently

   ```python
   def ensemble_token_voting(predictions_list):
       """
       Token-level majority voting

       Simpler but may produce invalid BIO sequences
       """
       from collections import Counter

       num_tokens = len(predictions_list[0])
       final_tags = []

       for i in range(num_tokens):
           # Get all predictions for this token
           token_predictions = [pred[i] for pred in predictions_list]

           # Majority vote
           most_common = Counter(token_predictions).most_common(1)[0][0]
           final_tags.append(most_common)

       # Fix invalid BIO sequences
       final_tags = fix_bio_sequences(final_tags)

       return final_tags
   ```

6. **Ensemble Method 4: Confidence-Based**

   **Approach:** Use model confidence scores

   ```python
   def ensemble_confidence_based(tokens, predictions_with_confidence):
       """
       Select predictions based on confidence scores

       Args:
           tokens: List of tokens
           predictions_with_confidence: List of (tags, confidence_scores) tuples
       """
       num_tokens = len(tokens)
       final_tags = []

       for i in range(num_tokens):
           # Get predictions and confidences for this token
           token_preds = []
           for tags, confidences in predictions_with_confidence:
               token_preds.append((tags[i], confidences[i]))

           # Select prediction with highest confidence
           best_tag = max(token_preds, key=lambda x: x[1])[0]
           final_tags.append(best_tag)

       # Fix BIO sequences
       final_tags = fix_bio_sequences(final_tags)

       return final_tags
   ```

7. **Full Implementation**
   ```python
   class NERPEnsemble:
       def __init__(self, models, model_names, weights=None):
           """
           Args:
               models: List of trained models
               model_names: List of model names (for debugging)
               weights: Optional list of weights (validation F1 scores)
           """
           self.models = models
           self.model_names = model_names

           if weights is None:
               # Equal weights
               self.weights = [1.0 / len(models)] * len(models)
           else:
               # Normalize
               total = sum(weights)
               self.weights = [w / total for w in weights]

       def predict(self, tokens_list, method='span_voting', threshold=0.5):
           """
           Ensemble prediction

           Args:
               tokens_list: List of token sequences
               method: 'span_voting', 'weighted', 'token_voting', 'confidence'
               threshold: Voting threshold
           """
           all_predictions = []

           # Get predictions from each model
           for model, name in zip(self.models, self.model_names):
               print(f"Getting predictions from {name}...")
               preds = model.predict(tokens_list)
               all_predictions.append(preds)

           # Ensemble
           final_predictions = []

           for i in range(len(tokens_list)):
               tokens = tokens_list[i]
               pred_tags = [preds[i] for preds in all_predictions]

               if method == 'span_voting':
                   final = ensemble_span_voting(tokens, pred_tags, threshold)
               elif method == 'weighted':
                   final = ensemble_weighted_voting(tokens, pred_tags, self.weights)
               elif method == 'token_voting':
                   final = ensemble_token_voting(pred_tags)
               else:
                   raise ValueError(f"Unknown method: {method}")

               final_predictions.append(final)

           return final_predictions
   ```

8. **Usage Example**
   ```python
   # Load trained models
   model_bert = load_model('bert_model.pt')
   model_bilstm = load_model('bilstm_crf_model.pt')
   model_llm = LLMPredictor()  # Wrapper for LLM

   # Create ensemble
   ensemble = NERPEnsemble(
       models=[model_bert, model_bilstm, model_llm],
       model_names=['BERT', 'BiLSTM-CRF', 'LLM-FewShot'],
       weights=[0.92, 0.91, 0.88]  # Validation F1 scores
   )

   # Predict
   predictions = ensemble.predict(
       val_tokens,
       method='weighted',
       threshold=0.5
   )

   # Evaluate
   from utils import print_evaluation_report
   print_evaluation_report(val_true_tags, predictions, val_tokens,
                          model_name="Ensemble (Weighted)")
   ```

9. **Hyperparameters**
   - **Number of models:** 3-5 (more isn't always better)
   - **Voting threshold:** 0.3-0.6 (experiment)
     - Lower: more entities (higher recall, lower precision)
     - Higher: fewer entities (higher precision, lower recall)
   - **Weights:** Use validation F1 scores

10. **Best Practices**
    - **Diverse models:** Mix different architectures
    - **Quality over quantity:** 3 good models > 5 mediocre models
    - **Validate ensemble:** Test on validation set
    - **Consider speed:** Ensemble is slower (multiple models)
    - **Error analysis:** Check which combinations work best

11. **Expected Results**
    | Best Individual Model | Ensemble | Improvement |
    |-----------------------|----------|-------------|
    | 90% F1 | 91-92% F1 | +1-2% |
    | 92% F1 | 93-94% F1 | +1-2% |
    | 88% F1 | 90-91% F1 | +2-3% |

**Notebook:** `13_Ensemble.ipynb`

**References:**
- Ensemble Methods for NER: https://aclanthology.org/W04-2401.pdf
- Voting Strategies: https://arxiv.org/abs/1909.08053
- Model Combination: https://www.aclweb.org/anthology/P18-1139.pdf

---

## Final Deliverables

### Task 15: Compare All Models

**Objective:** Comprehensive comparison of all 11 models

#### Implementation Steps

1. **Create Comparison Table**
   ```python
   import pandas as pd

   results = {
       'Model': [],
       'Precision': [],
       'Recall': [],
       'F1': [],
       'Training Time': [],
       'Inference Time': [],
       'Complexity': []
   }

   # Add results for each model
   for model_name, metrics in all_model_results.items():
       results['Model'].append(model_name)
       results['Precision'].append(metrics['precision'])
       results['Recall'].append(metrics['recall'])
       results['F1'].append(metrics['f1'])
       results['Training Time'].append(metrics['train_time'])
       results['Inference Time'].append(metrics['inference_time'])
       results['Complexity'].append(metrics['complexity'])

   df = pd.DataFrame(results)
   df = df.sort_values('F1', ascending=False)
   print(df.to_string(index=False))
   ```

2. **Visualizations**

   **F1 Score Comparison (Bar Chart):**
   ```python
   import matplotlib.pyplot as plt
   import seaborn as sns

   plt.figure(figsize=(12, 6))
   sns.barplot(data=df, x='Model', y='F1', palette='viridis')
   plt.xticks(rotation=45, ha='right')
   plt.ylabel('F1 Score')
   plt.title('Model Comparison - F1 Scores')
   plt.axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
   plt.legend()
   plt.tight_layout()
   plt.savefig('model_comparison_f1.png', dpi=300)
   plt.show()
   ```

   **Per-Entity-Type Heatmap:**
   ```python
   # Get per-type F1 for each model
   per_type_results = {}

   for model_name, predictions in all_predictions.items():
       by_type = evaluate_entity_spans_by_type(val_true_tags, predictions, val_tokens)
       per_type_results[model_name] = {entity_type: metrics['f1']
                                       for entity_type, metrics in by_type.items()}

   # Create DataFrame
   df_heatmap = pd.DataFrame(per_type_results).T

   # Heatmap
   plt.figure(figsize=(10, 8))
   sns.heatmap(df_heatmap, annot=True, fmt='.3f', cmap='RdYlGn',
               vmin=0.5, vmax=1.0, cbar_kws={'label': 'F1 Score'})
   plt.title('Per-Entity-Type F1 Scores by Model')
   plt.xlabel('Entity Type')
   plt.ylabel('Model')
   plt.tight_layout()
   plt.savefig('per_entity_type_heatmap.png', dpi=300)
   plt.show()
   ```

   **Precision-Recall Trade-off:**
   ```python
   plt.figure(figsize=(10, 8))

   for model_name in df['Model']:
       metrics = all_model_results[model_name]
       plt.scatter(metrics['recall'], metrics['precision'],
                  s=metrics['f1']*500, alpha=0.6, label=model_name)

   plt.xlabel('Recall')
   plt.ylabel('Precision')
   plt.title('Precision-Recall Trade-off (size = F1 score)')
   plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.savefig('precision_recall_tradeoff.png', dpi=300)
   plt.show()
   ```

3. **Analysis Sections**
   - Which models perform best overall?
   - Which entity types are hardest to predict?
   - Trade-offs: accuracy vs speed vs complexity
   - Error analysis: common failure modes
   - Computational cost comparison

4. **Select Final Model**
   ```python
   # Decision criteria
   criteria = {
       'f1_score': 0.5,       # 50% weight
       'speed': 0.2,          # 20% weight
       'robustness': 0.3      # 30% weight (per-type consistency)
   }

   # Calculate scores
   for model_name in df['Model']:
       score = (
           metrics['f1'] * criteria['f1_score'] +
           normalize_speed(metrics['inference_time']) * criteria['speed'] +
           calculate_robustness(per_type_f1s) * criteria['robustness']
       )
       print(f"{model_name}: {score:.3f}")
   ```

**Notebook:** `14_Model_Comparison.ipynb`

---

### Task 16: Generate Test Predictions

**Objective:** Create final predictions for test set

#### Implementation Steps

1. **Load Best Model(s)**
   ```python
   # Load the selected final model
   best_model = load_model('best_model.pt')

   # Or load ensemble
   ensemble = NERPEnsemble(
       models=[model1, model2, model3],
       weights=[0.92, 0.91, 0.88]
   )
   ```

2. **Load Test Data**
   ```python
   import json

   def load_jsonl(file_path):
       data = []
       with open(file_path, 'r', encoding='utf-8') as f:
           for line in f:
               data.append(json.loads(line.strip()))
       return data

   test_data = load_jsonl('test_data.jsonl')
   print(f"Loaded {len(test_data)} test samples")

   # Extract tokens
   test_tokens = [sample['tokens'] for sample in test_data]
   ```

3. **Generate Predictions**
   ```python
   print("Generating predictions...")

   # Predict
   test_predictions = best_model.predict(test_tokens)

   # Or ensemble
   # test_predictions = ensemble.predict(test_tokens, method='weighted')

   print(f"Generated {len(test_predictions)} predictions")
   ```

4. **Add Predictions to Test Data**
   ```python
   # Add ner_tags field
   for sample, pred_tags in zip(test_data, test_predictions):
       sample['ner_tags'] = pred_tags

   # Verify format
   print("Sample prediction:")
   print(json.dumps(test_data[0], indent=2))
   ```

5. **Validation Checks**
   ```python
   # Check 1: All samples have predictions
   assert len(test_data) == 5000, f"Expected 5000 samples, got {len(test_data)}"

   # Check 2: All have ner_tags field
   for sample in test_data:
       assert 'ner_tags' in sample, f"Sample {sample['id']} missing ner_tags"

   # Check 3: Length matches
   for sample in test_data:
       assert len(sample['tokens']) == len(sample['ner_tags']), \
           f"Sample {sample['id']}: {len(sample['tokens'])} tokens vs {len(sample['ner_tags'])} tags"

   # Check 4: Valid tags
   valid_tags = set(['O'] +
                    [f'B-{t}' for t in ['Politician', 'Artist', 'Facility', 'HumanSettlement',
                                        'OtherPER', 'PublicCorp', 'ORG']] +
                    [f'I-{t}' for t in ['Politician', 'Artist', 'Facility', 'HumanSettlement',
                                        'OtherPER', 'PublicCorp', 'ORG']])

   for sample in test_data:
       for tag in sample['ner_tags']:
           assert tag in valid_tags, f"Invalid tag: {tag}"

   # Check 5: No invalid BIO sequences
   invalid_count = 0
   for sample in test_data:
       prev_tag = 'O'
       for tag in sample['ner_tags']:
           if tag.startswith('I-'):
               entity_type = tag[2:]
               if not (prev_tag == f'B-{entity_type}' or prev_tag == f'I-{entity_type}'):
                   invalid_count += 1
                   break
           prev_tag = tag

   print(f"Validation complete!")
   print(f"Invalid BIO sequences: {invalid_count}")

   if invalid_count > 0:
       print("Fixing invalid sequences...")
       for sample in test_data:
           sample['ner_tags'] = fix_bio_sequences(sample['ner_tags'])
       print("Fixed!")
   ```

6. **Save Predictions**
   ```python
   output_file = 'test_data_predictions.jsonl'

   with open(output_file, 'w', encoding='utf-8') as f:
       for sample in test_data:
           f.write(json.dumps(sample) + '\n')

   print(f"Saved predictions to {output_file}")

   # Verify saved file
   loaded = load_jsonl(output_file)
   assert len(loaded) == len(test_data), "File save/load mismatch"
   print("Verification successful!")
   ```

7. **Statistics Summary**
   ```python
   from collections import Counter
   from utils import extract_entities

   # Tag distribution
   all_tags = []
   for sample in test_data:
       all_tags.extend(sample['ner_tags'])

   tag_counts = Counter(all_tags)

   print("\nPrediction Statistics:")
   print(f"Total tokens: {len(all_tags):,}")
   print(f"\nTag distribution:")
   for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
       pct = count / len(all_tags) * 100
       print(f"  {tag:20s}: {count:8,} ({pct:5.2f}%)")

   # Entity statistics
   all_entities = []
   for sample in test_data:
       entities = extract_entities(sample['tokens'], sample['ner_tags'])
       all_entities.extend(entities)

   entity_type_counts = Counter(e[1] for e in all_entities)

   print(f"\nTotal entities predicted: {len(all_entities):,}")
   print("\nEntity type distribution:")
   for entity_type, count in sorted(entity_type_counts.items(), key=lambda x: x[1], reverse=True):
       print(f"  {entity_type:20s}: {count:6,}")
   ```

**Notebook:** `15_Generate_Test_Predictions.ipynb`

---

### Task 17: Write Report (≤4 pages)

**Objective:** Document all experiments in a comprehensive report

#### Report Structure

**1. Problem and Data (0.5 pages)**
- Task description: BIO tagging for NER
- Dataset statistics from EDA:
  - Training samples: X
  - Validation samples: Y
  - Test samples: 5,000
  - Entity types: 7
  - Average sentence length
  - Entity distribution
- Data cleaning:
  - Removed N samples with invalid BIO sequences
  - Handled empty sentences
- Train/validation split:
  - 90/10 split
  - Stratified by entity presence
- Preprocessing:
  - Tokenization approach
  - Any special handling

**2. Methods (2 pages)**

For EACH model (11 total), describe:
- **Architecture:** Model structure, layers, dimensions
- **Features:** What inputs were used (word embeddings, POS, characters, etc.)
- **Training details:**
  - Optimizer, learning rate, batch size
  - Number of epochs
  - Early stopping criteria
  - Hardware used (GPU type)
  - Training time
- **BIO sequence enforcement:**
  - CRF layer (for applicable models)
  - Post-processing rules
  - How invalid sequences were prevented

**Model summary table:**
| Model | Architecture | Features | Training? | F1 |
|-------|--------------|----------|-----------|-----|
| HMM | ... | ... | Yes | 0.78 |
| CRF | ... | ... | Yes | 0.87 |
| ... | ... | ... | ... | ... |

**3. Results (1 page)**

- **Comparison table:**
  - All models with Precision, Recall, F1
  - Ranked by F1 score

- **Per-entity-type analysis:**
  - Which entity types are hardest?
  - Which models handle rare entities best?
  - Include heatmap or table

- **What worked and why:**
  - BERT: Pre-training provides strong baseline
  - Character embeddings: Handle OOV words
  - CRF: Enforces valid sequences, better boundaries
  - LLM few-shot: External knowledge helps
  - Ensemble: Complementary strengths

- **Failure analysis:**
  - Common error types
  - Confusion between entity types
  - Boundary detection issues
  - Examples of failures

**4. Final Model Choice (0.5 pages)**

- **Selected model:** [Name]
- **Justification:**
  - Best validation F1: X.XX
  - Good balance of accuracy vs speed
  - Robust across all entity types
  - Handles edge cases well
- **Trade-offs considered:**
  - Ensemble has highest F1 but slowest
  - BERT+CRF chosen for balance
  - Could deploy LLM for critical cases
- **Test predictions:**
  - Generated N entities
  - Tag distribution reasonable
  - No invalid BIO sequences

**Formatting:**
- Font: 11pt or 12pt
- Margins: 1 inch
- Line spacing: Single or 1.15
- Figures: Include 2-3 key visualizations
- References: Cite key papers/libraries

**File:** `NER_Contest_Report.pdf`

---

## Timeline Estimates

### Time per Model (Approximate)

| Model | Implementation | Training | Evaluation | Total |
|-------|---------------|----------|------------|-------|
| 0. HMM | 2h | 0.5h | 0.5h | 3h |
| 1. CRF | 3h | 1h | 0.5h | 4.5h |
| 2. Word+RNN | 4h | 2h | 0.5h | 6.5h |
| 3. Word+POS+RNN | 3h | 2h | 0.5h | 5.5h |
| 4. Char+BiLSTM+CRF | 6h | 4h | 0.5h | 10.5h |
| 5. BERT | 3h | 3h | 0.5h | 6.5h |
| 6. BERT+CRF | 4h | 3h | 0.5h | 7.5h |
| 7. Attention+BiLSTM+CRF | 8h | 5h | 0.5h | 13.5h |
| 8. LLM Basic | 2h | 0h | 1h | 3h |
| 9. LLM Few-Shot | 3h | 0h | 1h | 4h |
| 10. Ensemble | 2h | 0h | 1h | 3h |
| **Subtotal** | **40h** | **20.5h** | **6.5h** | **67h** |

### Final Tasks

| Task | Time |
|------|------|
| Model comparison | 3h |
| Test predictions | 2h |
| Report writing | 6h |
| **Subtotal** | **11h** |

### **Total Estimated Time: 78 hours**

**Realistic schedule:**
- Week 1: Setup + Models 0-3 (20h)
- Week 2: Models 4-7 (35h)
- Week 3: Models 8-10 + Final tasks (23h)

**Minimum viable (4 models only):**
- CRF (4.5h)
- Word+RNN (6.5h)
- BERT (6.5h)
- LLM Few-Shot (4h)
- Final tasks (11h)
- **Total: ~32 hours**

---

## Resources & References

### General NER Resources

1. **CoNLL-2003 Benchmark:** https://aclanthology.org/W03-0419.pdf
   - Standard NER dataset and evaluation

2. **Named Entity Recognition Survey (2024):** https://arxiv.org/abs/2401.10825
   - Comprehensive overview of recent advances

3. **Papers with Code - NER:** https://paperswithcode.com/task/named-entity-recognition-ner
   - Latest SOTA results and implementations

### Libraries & Tools

1. **HuggingFace Transformers:** https://huggingface.co/docs/transformers/
   - BERT and other transformer models

2. **spaCy:** https://spacy.io/
   - POS tagging and NLP preprocessing

3. **sklearn-crfsuite:** https://sklearn-crfsuite.readthedocs.io/
   - CRF implementation

4. **PyTorch:** https://pytorch.org/
   - Deep learning framework

5. **pytorch-crf:** https://pytorch-crf.readthedocs.io/
   - CRF layer for PyTorch

### Pre-trained Embeddings

1. **GloVe:** https://nlp.stanford.edu/projects/glove/
   - Pre-trained word embeddings

2. **FastText:** https://fasttext.cc/
   - Subword embeddings

### Research Papers by Model

**HMM:**
- Hidden Markov Models: https://www.nltk.org/_modules/nltk/tag/hmm.html

**CRF:**
- Conditional Random Fields (Original): https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers
- CRF for NER: https://www.chokkan.org/software/crfsuite/

**BiLSTM-CRF:**
- Neural Architectures for NER: https://arxiv.org/abs/1603.01360
- Character-level representations: https://arxiv.org/abs/1508.01991
- CoNLL 2003 Results (90.94% F1): https://domino.ai/blog/named-entity-recognition-ner-challenges-and-model

**BERT:**
- BERT Paper: https://arxiv.org/abs/1810.04805
- BERT for NER: https://arxiv.org/abs/1810.04805

**BERT + CRF:**
- BERT-CRF for NER: https://arxiv.org/abs/1902.03006

**Attention:**
- Attention is All You Need: https://arxiv.org/abs/1706.03762
- Attention for NER (92.57% F1): https://academic.oup.com/bioinformatics/article/34/8/1381/4657076

**LLM for NER:**
- PromptNER (+4-24% improvement): https://arxiv.org/abs/2305.15444
- GPT-NER: https://arxiv.org/abs/2304.10428
- Few-Shot NER Evaluation (73% F1 with CoT): https://arxiv.org/abs/2408.15796
- Medical NER with GPT-4 (95% F1): https://arxiv.org/abs/2505.08704

**Ensemble:**
- Ensemble Methods for NER: https://aclanthology.org/W04-2401.pdf

### Tutorials & Guides

1. **Token Classification Guide:** https://huggingface.co/course/chapter7/2
2. **PyTorch NER Tutorial:** https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
3. **CRF Tutorial:** https://www.chokkan.org/software/crfsuite/tutorial.html
4. **Attention Explained:** https://jalammar.github.io/illustrated-transformer/
5. **Prompt Engineering Guide:** https://www.promptingguide.ai/

---

## Notes

- This plan implements **11 models total** (exceeds requirement of 4+)
- **8 models involve training** (exceeds requirement of 2+)
- Models ordered from simplest → most advanced
- Total estimated time: **~78 hours** (or ~32 hours for minimum 4 models)
- All evaluation uses **entity-span level F1** (strict matching)
- BIO sequence validity enforced via CRF or post-processing
- Expected best F1: **92-95%** (ensemble or LLM few-shot)

**Key Success Factors:**
1. ✅ Thorough experimentation (11 models)
2. ✅ Proper evaluation (entity-span F1)
3. ✅ Good documentation (detailed report)
4. ✅ Valid BIO sequences (CRF or post-processing)
5. ✅ Diverse approaches (classical ML, deep learning, LLMs)

**Good luck with your NER contest!** 🚀
