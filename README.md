# TOPSIS-Based Selection of Best Pre-Trained Model for Text Summarization

## 📌 Problem Statement

The objective of this assignment is to apply the **TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)** method to select the best pre-trained model for **Text Summarization**.

Since my roll number ends with **0**, the assigned task category is **Text Summarization**.

The goal is not just to compare models using a single metric, but to evaluate them using **multiple performance criteria** and make a balanced, data-driven decision.

---

# 🧠 Methodology

## 1️⃣ Model Selection

The following widely-used pre-trained summarization models were selected for evaluation:

- `facebook/bart-large-cnn`
- `sshleifer/distilbart-cnn-12-6`
- `t5-base`
- `google/pegasus-xsum`

These models were chosen because they are popular, reliable, and commonly used for abstractive text summarization.

---

## 2️⃣ Dataset Used

- **Dataset:** XSum
- **Samples Evaluated:** 50 test samples
- XSum is a benchmark dataset for extreme summarization.
- Each sample contains:
  - A news article (input)
  - A single-sentence summary (reference)

A subset of 50 samples was used to ensure smooth execution in Google Colab.

---

## 3️⃣ Evaluation Metrics (Decision Criteria)

Each model was evaluated using the following five criteria:

| Criterion      | Type     | Description |
|--------------|----------|-------------|
| ROUGE-1      | Benefit (+) | Measures unigram overlap |
| ROUGE-2      | Benefit (+) | Measures bigram overlap |
| ROUGE-L      | Benefit (+) | Measures longest common subsequence |
| AvgTimeSec   | Cost (-) | Average inference time per sample |
| Params (M)   | Cost (-) | Model size (number of parameters in millions) |

### Why These Criteria?

- ROUGE metrics measure **summary quality**.
- Average inference time measures **efficiency**.
- Parameter count measures **model complexity and resource requirement**.

This ensures we consider both **accuracy and computational cost**.

---

## 4️⃣ Weight Assignment

Weights were assigned to reflect the importance of each criterion:

```
Weights = [3, 4, 3, 2, 1]
Impacts = ["+", "+", "+", "-", "-"]
```

- ROUGE-2 was given slightly higher weight because bigram overlap better captures summary coherence.
- Time and model size were treated as cost criteria.
- All weights were normalized before applying TOPSIS.

---

## 5️⃣ TOPSIS Procedure

The following steps were implemented:

### Step 1: Construct Decision Matrix  
Rows = Models  
Columns = Evaluation criteria  

### Step 2: Normalize Decision Matrix  
Vector normalization was used:

```
R_ij = X_ij / sqrt(sum(X_ij^2))
```

### Step 3: Weighted Normalized Matrix  

```
V_ij = W_j × R_ij
```

### Step 4: Determine Ideal Best and Ideal Worst

- For benefit criteria → max value is ideal best
- For cost criteria → min value is ideal best

### Step 5: Compute Separation Measures

Distance from Ideal Best:
```
S_i+ = sqrt(sum((V_ij - V_j+)^2))
```

Distance from Ideal Worst:
```
S_i- = sqrt(sum((V_ij - V_j-)^2))
```

### Step 6: Compute TOPSIS Score

```
Score_i = S_i- / (S_i+ + S_i-)
```

Higher score = Better model.

---

# 📊 Results

## 📋 Result Table

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | AvgTimeSec | Params(M) | TOPSIS Score | Rank |
|-------|---------|---------|---------|------------|------------|--------------|------|
| Model A | ... | ... | ... | ... | ... | ... | ... |

(*Actual values will be filled from generated CSV file*)

The final ranking is based on the TOPSIS score.

---

# 📈 Result Graphs

## 1️⃣ TOPSIS Score Comparison

- A bar graph was generated showing TOPSIS scores for each model.
- The tallest bar represents the best overall model.
- This graph makes it visually easy to compare model performance.

Saved as:
```
topsis_scores_bar.png
```

---

## 2️⃣ ROUGE Metrics Comparison

A line graph was generated comparing:
- ROUGE-1
- ROUGE-2
- ROUGE-L

This helps visualize:
- Which model performs best in terms of summary quality.
- Trade-offs between models.

Saved as:
```
rouge_metrics_line.png
```

---

# 🏆 Final Conclusion

- The model with **Rank = 1** is considered the best summarization model according to TOPSIS.
- This model achieves the best balance between:
  - High summary quality
  - Low inference time
  - Manageable model size

Rather than selecting the model with highest ROUGE alone, TOPSIS allows us to select a model that is **closest to the ideal solution** considering multiple criteria simultaneously.

---

# 📂 Repository Contents

```
├── topsis_summarization_results.csv
├── topsis_scores_bar.png
├── rouge_metrics_line.png
├── Colab_Notebook.ipynb
└── README.md
```

---

# 🚀 Tools & Libraries Used

- Python
- HuggingFace Transformers
- Datasets
- Evaluate (ROUGE)
- NumPy
- Pandas
- Matplotlib

---

# ✨ Learning Outcome

Through this project, I learned:

- How to evaluate NLP models using multiple performance metrics
- How to apply Multi-Criteria Decision Making (MCDM)
- Practical implementation of TOPSIS
- Real-world model comparison beyond single-metric evaluation

---

# 📌 Why TOPSIS is Suitable Here

In real-world deployment:

- A highly accurate model may be too slow.
- A fast model may produce poor summaries.

TOPSIS ensures we choose a **balanced and practical solution** instead of optimizing only one metric.

---

**Author:** Harshpreet  
**Course:** Predictive Analytics  
**Assignment:** TOPSIS-based Model Selection  
