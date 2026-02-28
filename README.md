# Sentiment Analysis Using Text Mining -- Movie Reviews
**Course:** BDA 622 -- Marketing Analytics | Mercer University | Spring 2024
**Authors:** Sahil Patel | Drashti Khatra

---

## Project Overview
Developed a sentiment analysis model using RapidMiner to classify movie reviews
as positive or negative. Compared two k-NN models with different parameter
configurations to evaluate the impact of parameter tuning on classification accuracy.

---

## Dataset
- **Source:** Stanford Large Movie Review Dataset
  (https://ai.stanford.edu/~amaas/data/sentiment/)
- **Total dataset size:** 50,000 labeled movie reviews
- **Working dataset:** 1,000 reviews (500 positive, 500 negative) randomly
  sampled from the training set

---

## Methodology

### Text Processing Pipeline
- Loaded text files using RapidMiner's `Process Documents from Files` operator
- Applied **tokenization** (non-letters mode for Model 1; linguistic tokens for Model 2)
- Applied **case normalization** (Transform Cases -- lowercase)
- Applied **stemming** (Snowball Stemmer -- English)
- Generated **TF-IDF word vectors** for feature representation
- Applied **pruning** (percentual in Model 1; absolute in Model 2)

### Data Partitioning
- **10-fold cross-validation** with automatic sampling and parallel execution

### Models Built

| | Model 1 (Default) | Model 2 (Optimized) |
|---|---|---|
| Algorithm | k-NN | k-NN |
| k (neighbors) | 5 | 10 |
| Distance Measure | Cosine Similarity | Euclidean Distance |
| Pruning | Percentual (3% -- 30%) | Absolute (3 -- 30 docs) |
| Tokenization | Non-letters | Linguistic tokens |

---

## Results

| Metric | Model 1 | Model 2 |
|---|---|---|
| Overall Accuracy | 77.90% | **83.50%** |
| Precision (Positive) | 76.17% | 77.41% |
| Recall (Positive) | 81.20% | **94.60%** |
| Precision (Negative) | 79.87% | **93.06%** |
| Recall (Negative) | 74.60% | 72.40% |
| False Negatives | 127 | **27** |
| False Positives | 94 | 138 |

**Model 2 achieved 83.50% overall accuracy**, a 5.6 percentage point improvement
over Model 1, driven by optimized tokenization, neighbor count, and distance measure.

---

## Key Findings
- Parameter tuning has a significant impact on text classification accuracy
- Shifting from cosine similarity (k=5) to Euclidean distance (k=10) substantially
  reduced false negatives (127 to 27)
- Model 2 improved positive recall from 81.20% to 94.60% -- critical for
  brand sentiment monitoring applications
- Trade-off noted: Model 2 introduced more false positives (94 to 138),
  highlighting the precision-recall balance in classification tasks

---

## Marketing Implications
Text mining and sentiment analysis enable businesses to:
- Monitor brand sentiment trends over time
- Identify common product issues and areas for improvement
- Detect emerging customer trends and opportunities
- Benchmark performance against competitors
- Personalize marketing messages based on customer feedback signals

---

## Tools & Technologies
- **RapidMiner** -- text mining, model building, cross-validation
- **Stanford Large Movie Review Dataset** -- benchmark NLP dataset
- **TF-IDF** -- feature extraction
- **k-NN classification** -- supervised machine learning
- **Snowball Stemmer** -- NLP preprocessing

---

## Files in This Repository
| File | Description |
|---|---|
| `Final-Report.docx` | Full written project report |
| `Marketing-Analytics-Presentation.pptx` | Class presentation slides |
| `BUS622-Final-Project.pdf` | Original project brief and requirements |
| `Project-Rubric.pdf` | Grading rubric |
