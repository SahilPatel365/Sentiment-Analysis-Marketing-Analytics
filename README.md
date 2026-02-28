# Sentiment Analysis Using Text Mining -- Movie Reviews
> **Course:** BDA 622 -- Marketing Analytics | Mercer University | Spring 2024
> **Authors:** Sahil Patel | Drashti Khatra

---

## Table of Contents
- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Text Processing Pipeline](#text-processing-pipeline)
- [Models Built](#models-built)
- [Experimental Results](#experimental-results)
- [Key Findings & Analysis](#key-findings--analysis)
- [Marketing Implications](#marketing-implications)
- [Limitations & Future Work](#limitations--future-work)
- [Tools & Technologies](#tools--technologies)
- [References](#references)

---

## Project Overview
This project develops a **text mining and sentiment analysis pipeline** using
RapidMiner to classify movie reviews as positive or negative. Two k-Nearest
Neighbor (k-NN) classification models were built with different parameter
configurations and evaluated on accuracy, precision, recall, and F1 metrics
using 10-fold cross-validation.

The project demonstrates the full text analytics workflow -- from raw unstructured
text through preprocessing, feature extraction, model training, parameter
optimization, and performance evaluation -- with direct application to marketing
analytics use cases such as brand monitoring, customer feedback analysis, and
competitive benchmarking.

---

## Business Problem
Businesses generate and receive vast volumes of unstructured text data daily --
customer reviews, social media posts, survey responses, and support tickets.
Manual analysis of this data at scale is not feasible.

**The core question this project addresses:**
> Can a machine learning model accurately classify customer sentiment from raw
> text data, and how significantly does parameter tuning impact classification
> performance?

Understanding this has direct commercial value:
- Identify product quality issues before they escalate
- Monitor brand health across review platforms
- Detect emerging customer trends and risks in real time
- Personalize marketing messages based on sentiment signals

---

## Dataset
| Attribute | Detail |
|---|---|
| Source | Stanford Large Movie Review Dataset |
| Full dataset size | 50,000 labeled reviews |
| Working dataset | 1,000 reviews (500 positive, 500 negative) |
| Sampling method | Random stratified selection from training set |
| Label balance | Perfectly balanced -- 50% positive, 50% negative |
| Data format | Individual labeled text files |

**Dataset Citation:**
Maas, A. et al. (2011). Learning Word Vectors for Sentiment Analysis.
*Proceedings of the 49th Annual Meeting of the Association for Computational
Linguistics.*
Link: https://ai.stanford.edu/~amaas/data/sentiment/

---

## Text Processing Pipeline

All text preprocessing was performed in RapidMiner using the following operators:

### Step 1 -- Document Loading
- `Process Documents from Files` operator
- Settings: extract text only, use file extension as type, System encoding

### Step 2 -- Tokenization
| | Model 1 | Model 2 |
|---|---|---|
| Mode | Non-letters | Linguistic tokens (English) |

### Step 3 -- Case Normalization
- `Transform Cases` operator -- all tokens converted to lowercase

### Step 4 -- Stemming
- `Stem (Snowball)` operator -- English language
- Reduces words to their root form (e.g., "running" → "run")

### Step 5 -- Feature Extraction
- **TF-IDF (Term Frequency -- Inverse Document Frequency)** word vectors
- Weights terms by how frequently they appear in a document relative to
  the entire corpus, reducing the influence of common words

### Step 6 -- Pruning
| | Model 1 | Model 2 |
|---|---|---|
| Method | Percentual | Absolute |
| Lower bound | 3.0% | 3 documents |
| Upper bound | 30.0% | 30 documents |

### Step 7 -- Data Partitioning
- `Cross Validation` operator
- 10 folds, automatic sampling, parallel execution enabled for both models

---

## Models Built

### Model 1 -- Default Parameters
| Parameter | Value |
|---|---|
| Algorithm | k-Nearest Neighbors (k-NN) |
| k (number of neighbors) | 5 |
| Distance measure | Cosine Similarity |
| Voting | Weighted |
| Tokenization mode | Non-letters |
| Pruning | Percentual |

### Model 2 -- Optimized Parameters
| Parameter | Value |
|---|---|
| Algorithm | k-Nearest Neighbors (k-NN) |
| k (number of neighbors) | 10 |
| Distance measure | Euclidean Distance |
| Voting | Weighted |
| Tokenization mode | Linguistic tokens |
| Pruning | Absolute |

**Why k-NN for text classification?**
k-NN is a non-parametric, instance-based algorithm well-suited for
high-dimensional text feature spaces (TF-IDF vectors). It classifies a
document by majority vote among its k nearest neighbors in the feature space.
No model training is required -- the algorithm memorizes the training examples
and classifies new instances at prediction time.

---

## Experimental Results

### Confusion Matrix Summary

| Metric | Model 1 (Default) | Model 2 (Optimized) |
|---|---|---|
| **Overall Accuracy** | 77.90% | **83.50%** |
| True Positives (TP) | 406 | **473** |
| True Negatives (TN) | 373 | 362 |
| False Positives (FP) | 94 | 138 |
| False Negatives (FN) | 127 | **27** |

### Classification Metrics

| Metric | Model 1 | Model 2 | Change |
|---|---|---|---|
| Precision -- Positive | 76.17% | 77.41% | +1.24% |
| Recall -- Positive | 81.20% | **94.60%** | **+13.40%** |
| Precision -- Negative | 79.87% | **93.06%** | **+13.19%** |
| Recall -- Negative | 74.60% | 72.40% | -2.20% |

---

## Key Findings & Analysis

### 1. Parameter Tuning Delivers Significant Accuracy Gains
Model 2 achieved **83.50% overall accuracy**, a 5.6 percentage point improvement
over Model 1's 77.90%. This improvement was driven by three key parameter changes:
- Increasing k from 5 to 10 -- more neighbors reduces sensitivity to noise
- Switching to Euclidean distance -- better suited for normalized TF-IDF vectors
  in the specific feature space of this dataset
- Switching to linguistic tokenization -- more semantically meaningful token
  boundaries compared to non-letter splitting

### 2. False Negatives Reduced Dramatically
False negatives dropped from **127 to 27** -- a 78.7% reduction. In marketing
contexts, false negatives (missed negative sentiment) are often more costly than
false positives because undetected customer dissatisfaction can escalate
unaddressed. Model 2 is significantly better suited for real-world brand
monitoring applications.

### 3. Precision-Recall Trade-off
Model 2 improved positive recall from 81.20% to 94.60% at the cost of increasing
false positives from 94 to 138. This is an expected trade-off in classification --
the optimal balance depends on the business priority:
- **If missing negative sentiment is costly** (e.g., product safety, quality
  issues): prioritize recall -- Model 2 is the better choice
- **If false alarms are costly** (e.g., unnecessary escalations): balance
  precision and recall depending on operational context

### 4. TF-IDF Outperforms Simple Word Counts
Using TF-IDF rather than raw term frequency downweights common words that appear
across all documents (e.g., "the", "and") and upweights discriminative terms
specific to positive or negative sentiment, improving classifier performance.

---

## Marketing Implications

The techniques demonstrated in this project have direct applications in marketing:

| Application | How Sentiment Analysis Helps |
|---|---|
| Brand monitoring | Track positive/negative sentiment trends across review platforms over time |
| Product quality | Detect spikes in negative sentiment around specific product features |
| Customer experience | Identify recurring complaint themes before they escalate |
| Competitive benchmarking | Compare sentiment scores against competitor products |
| Campaign measurement | Measure sentiment shift before and after marketing campaigns |
| Personalization | Segment customers by sentiment profile for targeted messaging |

---

## Limitations & Future Work

### Current Limitations
- **Dataset scope:** Only movie reviews -- sentiment patterns may differ across
  domains (e.g., automotive, financial services, consumer products)
- **Binary classification only:** Positive/negative labels; no neutral category
  or intensity scoring
- **Tool dependency:** RapidMiner workflow is not easily reproducible without
  the software license
- **Static model:** No retraining mechanism for concept drift over time

### Future Enhancements
- Rebuild pipeline in Python using `scikit-learn`, `NLTK`, or `spaCy` for full
  reproducibility
- Extend to multi-class sentiment (positive, neutral, negative, mixed)
- Experiment with more advanced models: Naive Bayes, SVM, BERT
- Apply domain-specific sentiment lexicons for non-movie-review text
- Integrate real-time social media data via API for live brand monitoring
- Add topic modeling (LDA) to identify themes within sentiment clusters

---

## Tools & Technologies
| Tool / Technology | Purpose |
|---|---|
| RapidMiner | End-to-end text mining and ML workflow |
| TF-IDF Vectorization | Text feature extraction |
| Snowball Stemmer | NLP preprocessing -- stemming |
| k-NN Classification | Supervised sentiment classification |
| 10-fold Cross-Validation | Model evaluation and generalization testing |
| Stanford Movie Review Dataset | Benchmark NLP dataset |

---

## Files in This Repository
| File | Description |
|---|---|
| `Final-Report.docx` | Full written project report with methodology and results |
| `Marketing-Analytics-Presentation.pptx` | Class presentation slides |
| `BUS622-Final-Project.pdf` | Original project brief and requirements |
| `Project-Rubric.pdf` | Grading rubric (written report + presentation) |

---

## References
Maas, A., Daly, R., Pham, P., Huang, D., Ng, A., & Potts, C. (2011).
Learning Word Vectors for Sentiment Analysis. *Proceedings of the 49th
Annual Meeting of the Association for Computational Linguistics*, 142--150.
https://ai.stanford.edu/~amaas/data/sentiment/

Lilien, G. L., Rangaswamy, A., & de Bruyn, A. (2017). *Principles of
Marketing Engineering and Analytics* (3rd ed.). DecisionPro, Inc.

---

*Completed as part of BDA 622 -- Marketing Analytics, Mercer University,
Stetson-Hatcher School of Business, Spring 2024.*
