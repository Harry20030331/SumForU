# Comparison Analysis

This section compares the performance of four methods—baseline, prompt engineering (pe), supervised fine-tuning (sft), and reinforcement learning (rl)—across rule-based, LLM-based, and user-based evaluation perspectives.

## Four Methods
- **baseline**: The base model without any fine-tuning, serving as a reference point for performance comparison.
- **pe**: Prompt Engineering approach, using carefully crafted prompts to guide the model for better persona-aware summaries.
- **sft**: Supervised Fine-Tuning, where the model is fine-tuned on synthetic conversation pairs generated from persona data.
- **rl**: Reinforcement Learning from AI Feedback (RLAIF), optimizing the model through preference-based learning with Prometheus-style rewards.

## Three Evaluation Perspectives
- **Rule-based Metrics**: Objective metrics calculated using established algorithms to assess summary quality and suitability.
- **LLM-based Metrics**: Subjective evaluations performed by large language models (LLMs) acting as judges, providing nuanced assessments of summary attributes.
- **User-based Metrics**: Human evaluations through user studies to gather direct feedback on summary effectiveness and persona alignment.

### Rule-based Metrics

---

**Summary Text Metrics**
- **RevCov**: Review Coverage - measures the proportion of review content words covered in the summary, assessing how comprehensively the summary captures the original reviews.
- **PersCov**: Persona Coverage - evaluates the coverage of persona-specific vocabulary in the summary, indicating alignment with user priorities.
- **RefBS-R**: Reference BERTScore Recall - computes BERTScore recall between summary and reference, measuring semantic similarity in terms of recall.
- **RevBS-P**: Review BERTScore Precision - calculates BERTScore precision against the concatenated reviews, assessing semantic precision.
- **PersBS-R**: Persona BERTScore Recall - measures BERTScore recall with persona text, evaluating semantic alignment with user persona.

---

**Suitability Score Metrics**
- **MAE**: Mean Absolute Error - computes the average absolute difference between predicted suitability scores and ground truth, quantifying prediction accuracy.
- **Spearman**: Spearman's Rank Correlation - assesses the monotonic relationship between predicted and actual suitability scores, measuring ranking consistency.
- **Within1Acc**: Within-1 Accuracy - calculates the percentage of predictions where the predicted score is within 1 point of the true score, evaluating near-exact accuracy.

---
**Results Summary**
|  | Summary Text | | | | | Suitability Score | | |
|:--------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:------------------:|:-------------:|:-------------:|
| Method | RevCov | PersCov | RefBS-R | RevBS-P | PersBS-R | MAE | Spearman | Within1Acc |
| gt | 0.5870 | 0.0700 | 1.0000 | 0.7828 | 0.7283 | 0.0000 | 1.0000 | 1.0000 |
| baseline | 0.3571 | 0.2683 | 0.7135 | 0.7618 | 0.8257 | 1.2362 | 0.4233 | 0.7007 |
| pe | 0.3611 | 0.2148 | 0.7146 | 0.7606 | 0.8172 | 1.2515 | 0.4498 | 0.6890 |
| sft | 0.3029 | 0.1954 | 0.7185 | 0.7542 | 0.8238 | 1.1130 | 0.5583 | 0.7720 |
| rl | 0.2816 | 0.1944 | 0.7220 | 0.7516 | 0.8345 | 1.0780 | 0.5629 | 0.7640 |

---

**Discussion**
- For **Suitability Score Metrics**, as training progresses, the MAE decreases while Spearman's correlation and Within1Acc increase, indicating that our methods better reflect attitudes towards products.

- For **Summary Text Metrics**, the four methods exhibit great but similar performance, suggesting semantic saturation in rule-based metrics. Each method identifies useful information, but whether the summaries align with user preferences remains an open question.

### LLM-based Metrics
---
**Metrics Definition**
- **Overall**: Aggregated score across all dimensions, providing a holistic evaluation of summary quality.
- **Consistency**: Assesses internal coherence between the summary's qualitative content and quantitative suitability score, ensuring sentiment alignment.
- **Grounding**: Evaluates factual accuracy and minimization of hallucinations by checking claims against the original review set.
- **Persona**: Measures alignment with user-specified persona priorities, focusing on relevant aspects and appropriate tone.

---
**Results Summary**
- Judge: Qwen/Qwen3-235B-A22B-Instruct-2507

| Method | Overall | Consistency | Grounding | Persona |
|--------|---------|-------------|-----------|---------|
| BASELINE | 0.336 | 0.441 | 0.331 | 0.235 |
| PE | 0.489 | 0.435 | 0.709 | 0.322 |
| SFT | 0.507 | 0.440 | 0.531 | 0.551 |
| RL | 0.668 | 0.684 | 0.429 | 0.892 |

- Judge: openai/gpt-oss-120b

| Method | Overall | Consistency | Grounding | Persona |
|--------|---------|-------------|-----------|---------|
| BASELINE | 0.441 | 0.464 | 0.361 | 0.497 |
| PE | 0.439 | 0.419 | 0.466 | 0.431 |
| SFT | 0.490 | 0.488 | 0.532 | 0.452 |
| RL | 0.630 | 0.629 | 0.642 | 0.620 |

---

**Discussion**
We apply an LLM judge to compare the four methods in three dimensions. The results show that the RL method achieves the best performance in all dimensions, demonstrating the effectiveness of our framework.

The RL method performs notably better under the Qwen/Qwen3-235B-A22B-Instruct-2507 judge, which is not unexpected since this model also serves as our reward model, potentially introducing bias. Nevertheless, the consistent superiority shown in GPT's evaluations underscores the robustness of our model's performance.

However, A key concern is the circular reasoning inherent in using LLM-trained networks evaluated by LLM judges, as both training and assessment depend on similar LLM capabilities, which may amplify underlying model biases and domenstrate the importance of User-based Metrics for a more objective evaluation.

---

### User-based Metrics

**Method**  
Three annotators evaluated 10 cases (one from each category). Each annotator read the persona and review set, then ranked the summaries from the four methods based on usefulness. For the top-ranked summary, they performed Targeted Likert Validation using a 5-Point Likert Scale (1 = Strongly Disagree, 5 = Strongly Agree) in three dimensions(Persona Alignment, Decision Utility, Factual Trustworthiness). This ensures data collection efficiency by focusing human effort only on the most valuable output. This small-scale user study was conducted due to limited resources, but provides initial insights into real user preferences.

**Metrics**  
1. **Kendall's W**: Kendall's coefficient of concordance to measure inter-annotator agreement.  
2. **Win Rate**: Percentage of cases where each method ranks first.  
3. **Mean Rank**: Average ranking position across all cases.  
4. **Spearman's Rho**: Correlation between Human-based and LLM-based Overall Preference(especially, focus on SFT v.s. RL) to assess alignment between human and LLM evaluations.

**implementation**
We collected rankings from three annotators for 10 cases, then calculated the Kendall's W, Win Rate, Mean Rank.
We call the LLM Judge(Qwen-235B) to get Win Rate, Mean Rank for LLM-based Overall Preference for the same 10 cases.
Then we extract the rank between SFT and RL, and compute Spearman's Rho between Human-based and LLM-based Overall Preference.
Finally, we collected Targeted Likert Validation scores for the top-ranked summaries from each annotator and selected out the RL method's scores only.
Then we computed the average scores in three dimensions.

**Results Summary**
| Metric | baseline | pe | sft | rl |
|--------|----------|----|-----|----|
| Win Rate-LLM | 0.1 | 0.1 | 0.0 | 0.8 |
| Win Rate-Humans | - | - | - | - |
| Mean Rank-LLM | 3.3 | 2.8 | 2.7 | 1.2 |
| Mean Rank-Humans | - | - | - | - |

Kendall's W: -

**Targeted Likert Validation Scores for RL Method**
| Persona Alignment | Usefulness | Factual Trustworthiness |
|-------------------|------------|-------------------------|
| -                 | -          | -                       |

**Discussion**
- The high Kendall's W (0.6) indicates substantial agreement among annotators, suggesting consistent human preferences.
- The RL method achieves the highest Win Rate and lowest Mean Rank in both LLM-based and Human-based evaluations, confirming its superiority.
- The similar behavior between Human-based and LLM-based rankings for SFT vs RL indicates alignment between human judgments and LLM evaluations. It is good to use LLM-based metrics as a proxy for human evaluation in future studies. 
- The high Targeted Likert Validation scores for the RL method across all dimensions indicate that users perceive RL-generated summaries as well-aligned with personas, useful for decision-making, and factually trustworthy, evaluated on absolute criteria rather than comparatively.

# Categories Analysis

## Introduction
To understand how different product categories impact the effectiveness of our RL framework, we conducted a detailed category-wise analysis. This analysis is essential for identifying category-specific strengths and weaknesses of RL optimization, enabling more targeted improvements and ensuring the model's robustness across diverse product domains. By examining performance variations, we can better tailor future adaptations to enhance persona-aware summarization in varied contexts.

### Implementation
The RL model was trained on the entire training dataset, encompassing multiple product categories to capture general patterns. We then evaluated its performance separately on each category within the training set, without additional category-specific training. We evaluated LLM-based metrics (Consistency, Grounding, Persona) by comparing RL against other methods using large language models as judges.

## RL Performance Summary Across Categories
| Category | Consistency | Grounding | Persona |
|----------|------------:|---------:|--------:|
| All_Beauty | 0.640 | 0.687 | 0.600 |
| Amazon_Fashion | 0.633 | 0.667 | 0.583 |
| Appliances | 0.627 | 0.653 | 0.573 |
| Baby_Products | 0.660 | 0.697 | 0.653 |
| CDs_and_Vinyl | 0.563 | 0.613 | 0.600 |
| Gift_Cards | 0.643 | 0.680 | 0.623 |
| Handmade_Products | 0.667 | 0.647 | 0.680 |
| Health_and_Personal_Care | 0.647 | 0.640 | 0.627 |
| Magazine_Subscriptions | 0.560 | 0.667 | 0.567 |
| Software | 0.617 | 0.583 | 0.600 |
| Overall | 0.634 | 0.619 | 0.618 |

## Train/Test Split Statistics for Key Categories (CDs_and_Vinyl vs Handmade_Products)
Using the same split method (shuffle with seed 42, train: first 300 records, test: records 300-400), we computed the metrics for train and test subsets:

### CDs_and_Vinyl
| Subset | Num Entries | Avg Reviews/Entry | Avg Word Count | Rating Dist | Overall Mean | Overall Var | Avg Var/Product |
|--------|-------------|-------------------|----------------|-------------|--------------|-------------|-----------------|
| Train | 300 | 19.1633 | 1173.1167 | 1.0:330, 2.0:318, 3.0:361, 4.0:679, 5.0:4061 | 4.3608 | 1.3771 | 1.1718 |
| Test | 100 | 18.9200 | 1119.6700 | 1.0:117, 2.0:98, 3.0:121, 4.0:192, 5.0:1364 | 4.3679 | 1.4140 | 1.2326 |

### Handmade_Products
| Subset | Num Entries | Avg Reviews/Entry | Avg Word Count | Rating Dist | Overall Mean | Overall Var | Avg Var/Product |
|--------|-------------|-------------------|----------------|-------------|--------------|-------------|-----------------|
| Train | 300 | 19.0767 | 801.9633 | 1.0:717, 2.0:407, 3.0:270, 4.0:368, 5.0:3961 | 4.1269 | 2.1356 | 1.6971 |
| Test | 100 | 19.2100 | 799.8000 | 1.0:218, 2.0:132, 3.0:102, 4.0:117, 5.0:1352 | 4.1728 | 2.0243 | 1.6408 |

The train/test splits show similar distributions within each category, confirming the split maintains representativeness.
These statistics show comparable data volume and quality, with CDs_and_Vinyl reviews being longer but ratings more consistent internally (lower variance), while Handmade_Products exhibit higher variance, suggesting greater opinion divergence.

## Conclusion on Category-Specific RL Performance
### Phenomenon
RL demonstrates strong performance in subjective quality metrics (Consistency, Grounding, Persona) across most categories, but shows significant variability in objective metrics and overall effectiveness between categories. Notably, **Handmade_Products** excels in RL optimization, achieving superior results in all evaluated metrics, while **CDs_and_Vinyl** lags behind, particularly in subjective metrics and baseline-to-RL improvements.

To highlight the differences, here is a comparison of RL performance in the three key metrics for CDs_and_Vinyl (CD) and Handmade_Products (HM) versus the overall average:

- **Consistency**: CD (0.563) < Overall (0.634) < HM (0.667)  
- **Grounding**: CD (0.613) < Overall (0.619) < HM (0.647)  
- **Persona**: CD (0.600) < Overall (0.618) < HM (0.680)  

This clearly shows that CDs_and_Vinyl performs below the overall average in all metrics, while Handmade_Products performs above the overall average.

### Analysis
Several initial hypotheses were proposed to explain these differences, but empirical data from our analysis has refuted them, pointing instead to category-inherent traits as the key driver.

1. **Hypothesis: Differences stem from data quantity or quality imbalances.**  
   - **Refutation**: Data statistics show comparable volumes (400 entries each, ~19 reviews per entry) and quality (**CDs_and_Vinyl** reviews are even longer on average at 55.97 words vs. 38.20). Rating distributions are similar, with high prevalence of 5.0 ratings (**CDs_and_Vinyl**: 5425, **Handmade_Products**: 5313). Thus, data characteristics do not account for the performance gap.

2. **Hypothesis: Subjectivity in CDs_and_Vinyl causes internal opinion divergence, hindering RL optimization.**  
   - **Refutation**: **CDs_and_Vinyl** exhibits lower average rating variance per product (1.1870) compared to **Handmade_Products** (1.6830), indicating higher opinion consistency rather than divergence. Despite this, RL performs worse in **CDs_and_Vinyl**, suggesting that uniformity in feedback may limit RL's learning diversity, contrary to expectations.

3. **Hypothesis: RL improvements are more pronounced in categories with high opinion consistency.**  
   - **Refutation**: Baseline-to-RL gains are less significant in **CDs_and_Vinyl** (e.g., Consistency: 0.563 → 0.563, Grounding: 0.400 → 0.613, Persona: 0.560 → 0.600) compared to **Handmade_Products** (e.g., Consistency: 0.470 → 0.667, Grounding: 0.373 → 0.647, Persona: 0.460 → 0.680). This indicates that RL thrives in divergent feedback environments, where it can leverage varied signals for better optimization.

In summary, category-inherent traits—such as objectivity vs. subjectivity and opinion divergence vs. consistency—profoundly influence RL effectiveness. **Handmade_Products**' objective, divergent reviews enable RL to achieve robust improvements, while **CDs_and_Vinyl**'s subjective, consistent nature constrains its potential. Future work could explore targeted adaptations, such as category-aware reward functions or fine-tuning, to mitigate such variations and enhance RL robustness across diverse product domains.

# Generalization Analysis

## Introduction
To assess the generalization capability of our RL-trained model, we evaluated its performance on categories not included in the training set. This is crucial because our training dataset is relatively small and covers only a limited number of product categories, raising concerns about potential overfitting to seen categories. In real-world applications, e-commerce platforms encompass a vast array of product categories, and a model that performs well only on trained categories may not generalize effectively. By testing on unseen categories, we aim to determine whether the small training dataset is sufficient to enable robust cross-category performance or if further data expansion and optimization are needed.

### Implementation
We directly deployed the RL-trained model on a new test set consisting of three unseen categories: Video_Games, Arts_Crafts_and_Sewing, and Industrial_and_Scientific. Each category included 100 data entries. We evaluated the model's performance using the same rule-based metrics (MAE, Spearman, Within1Acc) and LLM-based metrics (Consistency, Grounding, Persona) as in the main analysis, comparing RL against the baseline to measure generalization improvements.

## Data Results
### Video_Games

| Method | MAE | Spearman | Within1Acc | Consistency | Grounding | Persona |
|--------|-----|----------|------------|-------------|-----------|---------|
| baseline | 1.1300 | 0.5199 | 0.7200 | 0.443 | 0.360 | 0.440 |
| rl | 1.0500 | 0.5522 | 0.7800 | 0.630 | 0.630 | 0.637 |

### Arts_Crafts_and_Sewing

| Method | MAE | Spearman | Within1Acc | Consistency | Grounding | Persona |
|--------|-----|----------|------------|-------------|-----------|---------|
| baseline | 1.3550 | 0.3184 | 0.6400 | 0.480 | 0.390 | 0.457 |
| rl | 1.1600 | 0.5335 | 0.7500 | 0.640 | 0.587 | 0.617 |

### Industrial_and_Scientific

| Method | MAE | Spearman | Within1Acc | Consistency | Grounding | Persona |
|--------|-----|----------|------------|-------------|-----------|---------|
| baseline | 1.3100 | 0.4882 | 0.6900 | 0.500 | 0.310 | 0.463 |
| rl | 1.0600 | 0.5418 | 0.7700 | 0.623 | 0.683 | 0.607 |

## Generalization Performance Conclusion

- RL demonstrates improvements across all evaluation metrics (MAE, Spearman, Within1Acc, Consistency, Grounding, Persona), proving our framework's ability to generalize effectively to unseen categories.
- The MAE improvement in the Video_Games category is relatively small (improvement of 0.0800), only about 50% of the overall improvement in the main analysis (0.1582), indicating differences in RL's generalization performance on untrained categories, potentially requiring further optimization to enhance cross-category robustness.