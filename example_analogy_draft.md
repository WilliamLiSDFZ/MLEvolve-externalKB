# analogy agent — parent draft (invocation 1)

corpus 5cd1d5e7afd0 (38273 papers) | turns 3 | tokens in/out 38902/3039 | 61s | report

## Packet

# TASK (no solution has been written yet)

## Competition description (head and tail)
## Description

Can you help detect toxic comments ― *and* minimize unintended model bias? That's your challenge in this competition.

The Conversation AI team, a research initiative founded by Jigsaw and Google (both part of Alphabet), builds technology to protect voices in conversation. A main area of focus is machine learning models that can identify toxicity in online conversations, where toxicity is defined as anything *rude, disrespectful or otherwise likely to make someone leave a discussion*.

Last year, in the Toxic Comment Classification Challenge, you built multi-headed models to recognize toxicity and several subtypes of toxicity. This year's competition is a related challenge: building toxicity models that operate fairly across a diverse range of conversations.

Here's the background: When the Conversation AI team first built toxicity models, they found that the models incorrectly learned to associate the names of frequently attacked identities with toxicity. Models predicted a high likelihood of toxicity for comments containing those identities (e.g. "gay"), even when those comments were not actually toxic (such as "I am a gay woman"). This happens because training data was pulled from available sources where unfortunately, certain identities are overwhelmingly referred to in offensive ways. Training a model from data with these imbalances risks simply mirroring those biases back to users.

In this competition, you're challenged to build a model that recognizes toxicity and minimizes this type of unintended bias with respect to mentions of identities. You'll be using a dataset labeled for identity mentions and optimizing a metric designed to measure unintended bias. Develop strategies to reduce unintended bias in machine learning models, and you'll help the Conversation AI team, and the entire industry, build models that work well for a wide range of conversations.

*Disclaimer: The dataset for this competition contains text that may be considered profane, vulgar, or offensive.*

### Acknowledgments

The Conversation AI team would like to thank Civil Comments for making this dataset available publicly and the Online Hate Index Research Project at D-Lab, University of California, Berkeley, whose labeling survey/instrument informed the dataset labeling. We'd also like to thank everyone who has contributed to Conversation AI's research, especially those who took part in our last competition, the success of which led to the creation of this challenge.

## Evaluation

### Competition Evaluation

This competition will use a newly developed metric that combines several submetrics to balance overall performance with various aspects of unintended bias.

First, we'll define each submetric.

### Overall AUC

This is the ROC-AUC for the full evaluation set.

### Bias AUCs

To measure unintended bias, we again calculate the ROC-AUC, this time on three specific subsets of the test set for each identity, each capturing a different aspect of unintended bias. You can learn more about these metrics in Conversation AI's recent paper *Nuanced Metrics for Measuring Unintended Bias with Real Data in Text Classification*.

**Subgroup AUC**: Here, we restrict the data set to only the examples that mention the specific identity subgroup. *A low value in this metric means the model does a poor job of distinguishing between toxic and non-toxic comments that mention the identity*.

**BPSN (Background Positive, Subgroup Negative) AUC**: Here, we restrict the test set to the non-toxic examples that mention the identity and the toxic examples that do not. *A low value in this metric means that the model confuses non-toxic examples that mention the identity with toxic examples that do not*, likely meaning that the model predicts higher toxicity scores than it should for non-toxic examples mentioning the identity.

**BNSP (Background Negative, Subgroup Positive) AUC**: Here, we restrict the test set to the toxic examples that mention the identity and the non-toxic examples that do not. *A low value here means that the model confuses toxic examples that mention the identity with non-toxic examples that do not*, likely meaning that the model predicts lower toxicity scores than it should for toxic examples mentioning the identity.

### Generalized Mean of Bias AUCs

To combine the per-identity Bias AUCs into one overall measure, we calculate their generalized mean as defined below:

$$
M_p\left(m_s\right)=\left(\frac{1}{N} \sum_{s=1}^N m_

[... middle of the description omitted ...]

e proficient in English.

Raters were compensated 1.5 cents per judgment, a rate that was set based on two factors: what rates were competitive on the platform and targeting an hourly wage appropriate for raters' locales.

Figure Eight offers raters the option to select from multiple tasks with different pay, so rates need to be competitive to attract enough raters. At a rate of 1.5 cents per judgment, enough raters participated to complete this annotation in a few weeks. Following task completion, raters were given a satisfaction survey where the average score for "Pay" was 3.8 out of 5, which aligned with Figure Eight's recommendation to target pay to a satisfaction score greater than 3.5.

Hourly rates for workers depend on how fast judgements were completed. Most raters will have earned between $0.90/hour (at one comment per minute) to $5.40/hour (at 6 comments per minute), which aligns with typical hourly pay in the geographic regions where most raters are located.

Since this dataset was annotated in 2018, more tools have become available to help set rates. Consequently, typical pay for data annotation is increasing globally.

## File descriptions

- **train.csv** - the training set, which includes toxicity labels and subgroups
- **test.csv** - the test set, which does **not** include toxicity labels or subgroups
- **sample_submission.csv** - a sample submission file in the correct format

## Usage

This dataset is released under CC0, as is the underlying comment text.

## Available data (workspace listing / preview)
```
global_memory/

input/
    description.md (286 lines)
    sample_submission.csv (97321 lines)
    test.csv (205781 lines)
    train.csv (3820211 lines)
submission/

working/
```

-> input/sample_submission.csv has 97320 rows and 2 columns.
⚠️  IMPORTANT: This is the CORRECT submission format that must be followed!
The exact column names are: id, prediction
Any description.md format information should be IGNORED if it conflicts with this file.
Here is some information about the columns:
id (int64) has range: 7097320.00 - 7194639.00, 0 nan values
prediction (float64) has 1 unique values: [0.0]

-> input/test.csv has 97320 rows and 2 columns.
Here is some information about the columns:
comment_text (object) has 96862 unique values. Some example values: ['No.', 'Yes.', 'Well said.', 'Well said!']
id (int64) has range: 7097320.00 - 7194639.00, 0 nan values

-> input/train.csv has 1804874 rows and 45 columns.
Here is some information about the columns:
article_id (int64) has range: 2006.00 - 399541.00, 0 nan values
asian (float64) has range: 0.00 - 1.00, 1399744 nan values
atheist (float64) has range: 0.00 - 1.00, 1399744 nan values
bisexual (float64) has range: 0.00 - 1.00, 1399744 nan values
black (float64) has range: 0.00 - 1.00, 1399744 nan values
buddhist (float64) has range: 0.00 - 1.00, 1399744 nan values
christian (float64) has range: 0.00 - 1.00, 1399744 nan values
comment_text (object) has 1780822 unique values. Some example values: ['Well said.', 'No.', 'Exactly.', 'Yes.']
created_date (object) has 1804362 unique values. Some example values: ['2015-10-13 18:40:35.757707+00', '2015-09-29 20:48:48.201423+00', '2015-09-29 19:40:17.101981+00', '2015-09-29 17:54:15.971935+00']
disagree (int64) has range: 0.00 - 187.00, 0 nan values
female (float64) has range: 0.00 - 1.00, 1399744 nan values
funny (int64) has range: 0.00 - 102.00, 0 nan values
heterosexual (float64) has range: 0.00 - 1.00, 1399744 nan values
hindu (float64) has range: 0.00 - 1.00, 1399744 nan val…

## Resource budget
- search budget: 12.0 h wall clock for the whole search
- per-solution execution cap: 6.0 h
- CPU cores: 8
- GPU: Tesla V100-SXM2-32GB, 32 GiB

## Pretrained models available offline
Model1: ModernBERT

Description:ModernBERT is a modernized bidirectional encoder-only Transformer model pre-trained on 2 trillion tokens of English and code data. It significantly improves upon the original BERT with extended context length to 8k tokens (compared to just 512 for most encoders), making it faster and more accurate. The model features architectural innovations including improved efficiency enhancements and hardware-aware design. Unlike earlier BERT models, ModernBERT does not use token type IDs, simplifying its usage while maintaining compatibility as a drop-in replacement for BERT-like models. It excels at various NLP tasks including classification, token classification, question answering, and masked language modeling.

Code template (MUST copy exactly — do NOT change model variant names or file paths):
```python
from transformers import AutoTokenizer, ModernBertForSequenceClassification, ModernBertForMaskedLM, ModernBertForTokenClassification, ModernBertForQuestionAnswering, ModernBertForMultipleChoice
from torch.optim import AdamW

model_id = "answerdotai/ModernBERT-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Choose one model class according to your task:
# model = ModernBertForSequenceClassification.from_pretrained(model_id)  # for classification
# model = ModernBertForMaskedLM.from_pretrained(model_id)  # for masked language modeling
# model = ModernBertForTokenClassification.from_pretrained(model_id)  # for token classification/NER
# mode…
```

## Trace

[turn 1] assistant:
### Step 1 — structural properties

1. **The leaderboard objective is a tail-sensitive aggregate of conditional ranking problems, not the ordinary global classification objective.**  
   - **Objects:** continuous toxicity scorer; positive/negative comments; identity-conditioned subsets; overall ROC-AUC; subgroup, BPSN, and BNSP ROC-AUCs; generalized mean across identities.  
   - **Relations:** Every AUC depends on *relative score ordering* within a particular paired subset, and the generalized mean emphasizes weak subset scores. Standard pointwise BCE trained on the full, imbalanced population can improve global AUC while leaving the critical cross-group orderings (especially identity-negative versus background-positive) poor.  
   - **Evidence:** Evaluation defines three identity-specific AUCs and combines them with a generalized mean; it explicitly interprets BPSN as scoring non-toxic identity mentions too highly and BNSP as scoring toxic identity mentions too low.

2. **Identity mention is a training-time observed nuisance/proxy that is correlated with the target, but test-time predictions must remain discriminative conditional on that proxy.**  
   - **Objects:** comment text; identity-mention annotations in training; toxicity label; identity-bearing non-toxic and toxic comments; test text without identity columns.  
   - **Relations:** The model can exploit identity as a shortcut because the training distribution associates some identities with toxicity, yet evaluation penalizes exactly the resulting conditional ranking errors. A plain text classifier has no constraint separating legitimate toxicity evidence from proxy-identity evidence; subgroup annotations are available only to impose such a constraint or construct countervailing training comparisons, not as test inputs.  
   - **Evidence:** Description says identity names were incorrectly learned as toxicity signals; `train.csv` includes identity columns with many missing values, while `test.csv` contains only `comment_text` and `id`; BPSN/BNSP definitions operationalize the unwanted proxy association.

3. **The corrective cases that define the metric are sparse conditional strata inside a 1.8M-row corpus, under a single-V100 six-hour training cap.**  
   - **Objects:** majority training comments; identity-annotated comments; rare label×identity strata; minibatches; pretrained encoder; 6-hour cap.  
   - **Relations:** Uniform minibatches approximate population risk and can underexpose the BPSN/BNSP error pairs, whereas full exhaustive transformer training or many resampling experiments may exceed the cap. The first run needs a sampling/weighting mechanism that concentrates optimization on metric-relevant strata without multiplying models or passes.  
   - **Evidence:** `train.csv` has 1,804,874 rows, identity columns have 1,399,744 missing values, and the per-solution budget is 6 hours on one V100; the metric separately evaluates identity-conditioned strata.

### Step 2 — query translations

**Property 1**
- `partial AUC optimization imbalanced`
- `worst group AUC optimization`
- `generalized mean group fairness`
- `ranking loss subgroup robustness`

**Property 2**
- `spurious correlation invariant risk minimization`
- `group robust classification shortcut features`
- `counterfactual fairness proxy discrimination`
- `conditional demographic parity ranking`

**Property 3**
- `minority hard example sampling ranking`
- `stratified minibatch group robustness`
- `importance sampling worst group loss`

[turn 1] search_papers('partial AUC optimization imbalanced', k=10) ->
     22.56  [aaai-2025] Towards Macro-AUC Oriented Imbalanced Multi-Label Continual Learning  (aaai-2025/2e0b591f1c77306f4923e9ab2e615fff2317b795)
     22.29  [icml-2025] A Federated Stochastic Multi-level Compositional Minimax Algorithm for Deep AUC Maximization  (icml-2025/zhang24aw)
     22.22  [icml-2024] A Federated Stochastic Multi-level Compositional Minimax Algorithm for Deep AUC Maximization  (icml-2024/zhang24aw)
     21.04  [icml-2024] Interplay of ROC and Precision-Recall AUCs: Theoretical Limits and Practical Implications in Binary   (icml-2024/mihelich24a)
     20.97  [icml-2025] Interplay of ROC and Precision-Recall AUCs: Theoretical Limits and Practical Implications in Binary   (icml-2025/mihelich24a)
     20.20  [neurips-2024] AUC Maximization under Positive Distribution Shift  (neurips-2024/3f8c7eb848ffec848f3ed2b7ca44915d)
     19.65  [aaai-2025] Leveraging First and Zeroth-Order Gradient to Address Imbalanced Black-Box Prompt Tuning via Minimax  (aaai-2025/f805572b4f4ddec9b21f845f8eef12392a8b7556)
     18.07  [iclr-2024] P$^2$OT: Progressive Partial Optimal Transport for Deep Imbalanced Clustering  (iclr-2024/hd3sgvqpsr)
     16.47  [aaai-2025] GBRIP: Granular Ball Representation for Imbalanced Partial Label Learning  (aaai-2025/d1e30abfe61da356955ff5b226e013ec8c5f0de3)
     14.49  [neurips-2024] AUCSeg: AUC-oriented Pixel-level Long-tail Semantic Segmentation  (neurips-2024/e574db41163e700545ff4114f96b3d7a)

[turn 1] search_papers('worst group AUC optimization', k=10) ->
     17.87  [icml-2025] Differentially Private Worst-group Risk Minimization  (icml-2025/zhou24b)
     17.78  [icml-2024] Differentially Private Worst-group Risk Minimization  (icml-2024/zhou24b)
     16.11  [icml-2025] Controllable Prompt Tuning For Balancing Group Distributional Robustness  (icml-2025/phan24b)
     16.04  [icml-2024] Controllable Prompt Tuning For Balancing Group Distributional Robustness  (icml-2024/phan24b)
     15.65  [neurips-2024] Group Robust Preference Optimization in Reward-free RLHF  (neurips-2024/4147dfaa46cd7e20a2aecb91097ae8cc)
     15.45  [iclr-2025] MixMax: Distributional Robustness in Function Space via Optimal Data Mixtures  (iclr-2025/dikphooa2d)
     15.34  [neurips-2024] The Group Robustness is in the Details: Revisiting Finetuning under Spurious Correlations  (neurips-2024/dc4db2ff2c1aefce3b594f821ea82fe6)
     15.01  [aaai-2025] FROC: Building Fair ROC from a Trained Classifier  (aaai-2025/770f3d5748477fdb1a04856827eed0eb120449cb)
     14.54  [icml-2025] Improving Group Robustness on Spurious Correlation Requires Preciser Group Inference  (icml-2025/han24g)
     14.51  [icml-2024] Improving Group Robustness on Spurious Correlation Requires Preciser Group Inference  (icml-2024/han24g)

[turn 1] search_papers('generalized mean group fairness', k=10) ->
     22.14  [aaai-2024] Towards Fairer Centroids in K-means Clustering  (aaai-2024/3f9d284f2841cdc83484cb11cd9d425c707ca1a5)
     22.07  [aaai-2025] Bayes-Optimal Fair Classification with Multiple Sensitive Features  (aaai-2025/57f6efb4fe18c8f96a93acf5163435a0b250f0d1)
     21.77  [aaai-2025] Fair Bayesian Data Selection via Generalized Discrepancy Measures  (aaai-2025/d4fcd0ede5b01d10d79e4801974e13979e7a649a)
     19.40  [icml-2024] Fair Risk Control: A Generalized Framework for Calibrating Multi-group Fairness Risks  (icml-2024/zhang24be)
     19.40  [icml-2025] Fair Risk Control: A Generalized Framework for Calibrating Multi-group Fairness Risks  (icml-2025/zhang24be)
     19.24  [iclr-2025] Relax and Merge: A Simple Yet Effective Framework for Solving Fair $k$-Means and $k$-sparse Wasserst  (iclr-2025/n8h1z588eu)
     18.61  [aaai-2024] Long-Term Fair Decision Making through Deep Generative Models  (aaai-2024/6bf3a4745889aab27b288d8527602b2e9ed4c039)
     18.49  [neurips-2024] Fair Kernel K-Means: from Single Kernel to Multiple Kernel  (neurips-2024/b48bb4ad31718381838f68bf5f0d4ff1)
     18.11  [aaai-2025] Harnessing Diffusion-Generated Synthetic Images for Fair Image Classification  (aaai-2025/e934049906863cc3ca1b86d8afabd4f3b4551d56)
     17.91  [naacl-2024] Group Fairness in Multilingual Speech Recognition Models  (naacl-2024/2024findings-naacl143)

[turn 1] search_papers('ranking loss subgroup robustness', k=10) ->
     15.22  [iccv-2023] Identification of Systematic Errors of Image Classifiers on Rare Subgroups  (iccv-2023/metzenidentificationofsystematicerrorsofimageclassifiersonraresubgroupsiccv2023paper)
     15.14  [neurips-2024] Improving Subgroup Robustness via Data Selection  (neurips-2024/abbbb25cddb2c2cd08714e6bfa2f0634)
     14.80  [iclr-2025] Group Distributionally Robust Dataset Distillation with Risk Minimization  (iclr-2025/3jsu5qxnru)
     13.82  [iccv-2023] Better May Not Be Fairer: A Study on Subgroup Discrepancy in Image Classification  (iccv-2023/chiubettermaynotbefairerastudyonsubgroupdiscrepancyiccv2023paper)
     13.53  [icml-2025] Fine-Grained Causal Dynamics Learning with Quantization for Improving Robustness in Reinforcement Le  (icml-2025/hwang24b)
     13.53  [icml-2024] Fine-Grained Causal Dynamics Learning with Quantization for Improving Robustness in Reinforcement Le  (icml-2024/hwang24b)
     13.46  [neurips-2024] Dynamic Subgroup Identification in Covariate-adjusted Response-adaptive Randomization Experiments  (neurips-2024/9af09f8aa60f81dad5dcdbafff605b0d)
     13.21  [icml-2025] Stability and Multigroup Fairness in Ranking with Uncertain Predictions  (icml-2025/devic24a)
     13.16  [icml-2024] Stability and Multigroup Fairness in Ranking with Uncertain Predictions  (icml-2024/devic24a)
     12.96  [aaai-2025] From Pairwise to Ranking: Climbing the Ladder to Ideal Collaborative Filtering with Pseudo-Ranking  (aaai-2025/27d5e36ead44eeec5a48911c5affcc8d3ab0fe01)

[turn 1] search_papers('spurious correlation invariant risk minimization', k=10) ->
     25.17  [iclr-2025] Learning Graph Invariance by Harnessing Spuriosity  (iclr-2025/usvjlgd1f7)
     25.08  [icml-2025] Improving Group Robustness on Spurious Correlation Requires Preciser Group Inference  (icml-2025/han24g)
     24.93  [icml-2024] Improving Group Robustness on Spurious Correlation Requires Preciser Group Inference  (icml-2024/han24g)
     23.18  [neurips-2024] Mitigating Spurious Correlations via Disagreement Probability  (neurips-2024/879c5890a9d2ecdcb590c9674cda4a59)
     22.09  [icml-2024] Empowering Graph Invariance Learning with Deep Spurious Infomax  (icml-2024/yao24a)
     22.03  [icml-2025] Empowering Graph Invariance Learning with Deep Spurious Infomax  (icml-2025/yao24a)
     20.88  [iclr-2025] Severing Spurious Correlations with Data Pruning  (iclr-2025/bk13qfu8ru)
     20.75  [neurips-2024] Dissecting the Failure of Invariant Learning on Graphs  (neurips-2024/932147114c48f8b04d41aebc0c631158)
     20.49  [aaai-2025] Exploring Rationale Learning for Continual Graph Learning  (aaai-2025/6b2eeab029cfacd0ecd8ad582558f846da80cbbf)
     20.36  [iclr-2024] Views Can Be Deceiving: Improved SSL Through Feature Space Augmentation  (iclr-2024/mutjbk3ilg)

[turn 1] search_papers('group robust classification shortcut features', k=10) ->
     18.72  [neurips-2025] Rectifying Shortcut Behaviors in Preference-based Reward Learning  (neurips-2025/5d97b7e62022c859347397f6c1e8d0f9)
     17.34  [icml-2025] Improving Robustness to Multiple Spurious Correlations by Multi-Objective Optimization  (icml-2025/kim24l)
     17.34  [icml-2024] Improving Robustness to Multiple Spurious Correlations by Multi-Objective Optimization  (icml-2024/kim24l)
     17.14  [iclr-2025] Generative Classifiers Avoid Shortcut Solutions  (iclr-2025/ocuyc7bzxq)
     16.94  [iclr-2024] On the Foundations of Shortcut Learning  (iclr-2024/tj3xlvue9f)
     16.54  [iccv-2023] What do neural networks learn in image classification? A frequency shortcut perspective  (iccv-2023/wangwhatdoneuralnetworkslearninimageclassificationafrequencyiccv2023paper)
     16.02  [aaai-2025] Bot Meets Shortcut: How Can LLMs Aid in Handling Unknown Invariance OOD Scenarios?  (aaai-2025/f930d2cbb89f26ff54093ebcf01376b2e8332495)
     15.70  [naacl-2025] Fighting Spurious Correlations in Text Classification via a Causal Learning Perspective  (naacl-2025/2025naacl-long215)
     15.50  [aaai-2024] Spurious Feature Eraser: Stabilizing Test-Time Adaptation for Vision-Language Foundation Model  (aaai-2024/233265df9246d846bec87b546354b1993188b551)
     15.42  [icml-2025] Federated Self-Explaining GNNs with Anti-shortcut Augmentations  (icml-2025/yue24b)

[turn 1] search_papers('counterfactual fairness proxy discrimination', k=10) ->
     18.99  [neurips-2024] FairJob: A Real-World Dataset for Fairness in Online Systems  (neurips-2024/142bff4f4c01dd55c4309860ff3a59f1-abstract-datasetsandbenchmarkstrackhtml)
     18.78  [iclr-2025] Towards counterfactual fairness through auxiliary variables  (iclr-2025/gpuv1fvzi1)
     18.14  [aaai-2024] Sequential Conditional Transport on Probabilistic Graphs for Interpretable Counterfactual Fairness  (aaai-2024/28bf9340a47d4a29ad9cd81428b1ccfa4a2376ce)
     18.02  [neurips-2024] Counterfactual Fairness by Combining Factual and Counterfactual Predictions  (neurips-2024/558da25ef295121e2d79209a95e45436)
     17.65  [neurips-2024] Do Counterfactually Fair Image Classifiers Satisfy Group Fairness? -- A Theoretical and Empirical St  (neurips-2024/65b4660c1fee9842472e1557b582c940-abstract-datasetsandbenchmarkstrackhtml)
     17.12  [iclr-2024] Empirical Likelihood for Fair Classification  (iclr-2024/gacjmj1ms1)
     17.07  [aaai-2024] All Should Be Equal in the Eyes of LMs: Counterfactually Aware Fair Text Generation  (aaai-2024/2c10fefef87073416bef7976a9caed544f0762ad)
     16.48  [aaai-2024] Fairness-Accuracy Trade-Offs: A Causal Perspective  (aaai-2024/3cad85c59140aad4ab1af29a332da5685f4f72a3)
     16.01  [naacl-2024] Addressing Both Statistical and Causal Gender Fairness in NLP Models  (naacl-2024/2024findings-naacl38)
     15.42  [neurips-2024] Exogenous Matching: Learning Good Proposals for Tractable Counterfactual Estimation  (neurips-2024/ee94bf235482e4c1f689c04c81656dbf)

[turn 1] search_papers('conditional demographic parity ranking', k=10) ->
     27.70  [aaai-2024] Auditing and Enforcing Conditional Fairness via Optimal Transport  (aaai-2024/2aed817fa2fa8523329ae5df4f5e2b542e8a8ab1)
     25.73  [aaai-2025] Breaking the Dyadic Barrier: Rethinking Fairness in Link Prediction Beyond Demographic Parity  (aaai-2025/4c7d2f8db37f3491280c660a9d49a41a7f96cfaa)
     22.70  [neurips-2024] Regression under demographic parity constraints via unlabeled post-processing  (neurips-2024/d5c3ecf397fff63419bb5f5f2d8afe33)
     18.32  [naacl-2025] Watching the AI Watchdogs: A Fairness and Robustness Analysis of AI Safety Moderation Classifiers  (naacl-2025/2025naacl-short22)
     16.75  [icml-2024] On the Maximal Local Disparity of Fairness-Aware Classifiers  (icml-2024/jin24c)
     16.24  [iclr-2024] Post-hoc bias scoring is optimal for fair classification  (iclr-2024/fm5xfcar2y)
     15.22  [aaai-2025] Fairness Aware Reinforcement Learning via Proximal Policy Optimization  (aaai-2025/9045a5dec9a8983b307c45fd0323aaf9b744d7e9)
     14.84  [acl-2025] Has Machine Translation Evaluation Achieved Human Parity? The Human Reference and the Limits of Prog  (acl-2025/2025acl-short63)
     14.69  [acl-2025] Fairness Beyond Performance: Revealing Reliability Disparities Across Groups in Legal NLP  (acl-2025/2025acl-long1188)
     14.54  [acl-2025] Which Demographics do LLMs Default to During Annotation?  (acl-2025/2025acl-long848)

[turn 1] search_papers('minority hard example sampling ranking', k=10) ->
     15.82  [aaai-2025] A Unified Loss for Handling Inter-Class and Intra-Class Imbalance in Medical Image Segmentation  (aaai-2025/81dbec1b95c66ad888253be955ca4dcd0f2182cd)
     14.66  [acl-2025] Mitigating Shortcut Learning with InterpoLated Learning  (acl-2025/2025acl-long450)
     14.06  [iclr-2024] Don't Play Favorites: Minority Guidance for Diffusion Models  (iclr-2024/3nmo9ly4jn)
     13.97  [naacl-2025] MiLoRA: Harnessing Minor Singular Components for Parameter-Efficient LLM Finetuning  (naacl-2025/2025naacl-long248)
     13.63  [neurips-2025] Concentration and excess risk bounds for imbalanced classification with synthetic oversampling  (neurips-2025/e715ad3d527e246af3ab286c3f3d367c)
     13.52  [iccv-2023] Late Stopping: Avoiding Confidently Learning from Mislabeled Examples  (iccv-2023/yuanlatestoppingavoidingconfidentlylearningfrommislabeledexamplesiccv2023paper)
     13.40  [neurips-2024] The Surprising Effectiveness of SP Voting with Partial Preferences  (neurips-2024/054e9f9a286671ababa3213d6e59c1c2)
     13.38  [acl-2025] FocalPO: Enhancing Preference Optimizing by Focusing on Correct Preference Rankings  (acl-2025/2025acl-short21)
     13.18  [acl-2024] Instruction Tuning with Retrieval-based Examples Ranking for Aspect-based Sentiment Analysis  (acl-2024/2024findings-acl284)
     13.15  [aaai-2025] SoLA: Leveraging Soft Activation Sparsity and Low-Rank Decomposition for Large Language Model Compre  (aaai-2025/bad655f4e3830680576c4894edc29af7b25a1b7b)

[turn 1] search_papers('stratified minibatch group robustness', k=10) ->
     16.49  [aaai-2025] Stratified Knowledge-Density Super-Network for Scalable Vision Transformers  (aaai-2025/2db9b499c2b22da2d1cef3bcbee66f1cd2486185)
     14.37  [aaai-2024] Using Stratified Sampling to Improve LIME Image Explanations  (aaai-2024/0572aa74eb83ffb591d2d6310dd1ad662e889476)
     13.96  [iclr-2025] Aioli: A Unified Optimization Framework for Language Model Data Mixing  (iclr-2025/szgzjhanse)
     13.91  [aaai-2024] Towards Stability and Generalization Bounds in Decentralized Minibatch Stochastic Gradient Descent  (aaai-2024/f111d7302652acc103bcb1f305e33363031a6c60)
     13.89  [iclr-2025] Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling  (iclr-2025/rsgprjdihh)
     13.28  [aaai-2024] Stratified GNN Explanations through Sufficient Expansion  (aaai-2024/24a1674e332118a0322784060fc3f3ca0c4f10be)
     13.08  [neurips-2025] Beyond the Average: Distributional Causal Inference under Imperfect Compliance  (neurips-2025/62fe03e97f70b2d004d684d707d990eb)
     12.92  [neurips-2024] Stratified Prediction-Powered Inference for Effective Hybrid Evaluation of Language Models  (neurips-2024/c9fcd02e6445c7dfbad6986abee53d0d)
     12.51  [neurips-2025] Stratify or Die: Rethinking Data Splits in Image Segmentation  (neurips-2025/69a2ebb622b999be1cb932aceea5a236)
     11.93  [iccv-2023] Panoramas from Photons  (iccv-2023/jungermanpanoramasfromphotonsiccv2023paper)

[turn 1] search_papers('importance sampling worst group loss', k=10) ->
     19.04  [neurips-2024] Group Robust Preference Optimization in Reward-free RLHF  (neurips-2024/4147dfaa46cd7e20a2aecb91097ae8cc)
     18.92  [icml-2024] Differentially Private Worst-group Risk Minimization  (icml-2024/zhou24b)
     18.84  [icml-2025] Differentially Private Worst-group Risk Minimization  (icml-2025/zhou24b)
     18.09  [naacl-2024] Group Fairness in Multilingual Speech Recognition Models  (naacl-2024/2024findings-naacl143)
     17.51  [neurips-2024] The Group Robustness is in the Details: Revisiting Finetuning under Spurious Correlations  (neurips-2024/dc4db2ff2c1aefce3b594f821ea82fe6)
     16.72  [iclr-2025] Group-robust Sample Reweighting for Subpopulation Shifts via Influence Functions  (iclr-2025/aqj9ifxrl6)
     16.57  [aaai-2025] Boosting Test Performance with Importance Sampling-a Subpopulation Perspective  (aaai-2025/150b28ab6b1d5a47f8126f1073afdf668e5bb9f6)
     16.10  [neurips-2024] CODA: A Correlation-Oriented Disentanglement and Augmentation Modeling Scheme for Better Resisting S  (neurips-2024/ba06eb7c3ff7cb731ba04c906d212445)
     15.47  [iclr-2025] MixMax: Distributional Robustness in Function Space via Optimal Data Mixtures  (iclr-2025/dikphooa2d)
     15.18  [iclr-2025] Group Distributionally Robust Dataset Distillation with Risk Minimization  (iclr-2025/3jsu5qxnru)

[turn 2] read_abstract(['icml-2024/zhang24aw', 'icml-2024/han24g', 'neurips-2024/dc4db2ff2c1aefce3b594f821ea82fe6', 'iclr-2025/aqj9ifxrl6', 'aaai-2025/150b28ab6b1d5a47f8126f1073afdf668e5bb9f6', 'neurips-2024/abbbb25cddb2c2cd08714e6bfa2f0634', 'naacl-2024/2024findings-naacl38', 'acl-2025/2025acl-long450']) -> 8 abstracts

[turn 2] search_papers('pairwise AUC surrogate loss', k=10) ->
     20.38  [neurips-2024] AUCSeg: AUC-oriented Pixel-level Long-tail Semantic Segmentation  (neurips-2024/e574db41163e700545ff4114f96b3d7a)
     17.95  [neurips-2024] PSL: Rethinking and Improving Softmax Loss from Pairwise Perspective for Recommendation  (neurips-2024/db1d5c63576587fc1d40d33a75190c71)
     17.87  [aaai-2024] Limited Memory Online Gradient Descent for Kernelized Pairwise Learning with Dynamic Averaging  (aaai-2024/7e62417c144a59928ad55e275e7c99cd608de11a)
     15.04  [aaai-2025] Towards Macro-AUC Oriented Imbalanced Multi-Label Continual Learning  (aaai-2025/2e0b591f1c77306f4923e9ab2e615fff2317b795)
     14.71  [aaai-2025] Multi-Label Ranking Loss Minimization for Matrix Completion  (aaai-2025/d9bcb53d01f997df3ef0312058bce7c7617cae9d)
     14.69  [neurips-2025] Establishing Linear Surrogate Regret Bounds for Convex Smooth Losses via Convolutional Fenchel–Young  (neurips-2025/2fbeed1dd7162f91804e7b9246e0c1a8)
     14.66  [acl-2025] Influences on LLM Calibration: A Study of Response Agreement, Loss Functions, and Prompt Styles  (acl-2025/2025acl-long188)
     14.38  [neurips-2025] Consistency Conditions for Differentiable Surrogate Losses  (neurips-2025/c52d0836624a1f7c55cfc5aed8c501f4)
     13.84  [neurips-2025] ComRank: Ranking Loss for Multi-Label Complementary Label Learning  (neurips-2025/adedc33539eafdf3d2c14a5e0032db6e)
     13.82  [neurips-2024] A Universal Growth Rate for Learning with Smooth Surrogate Losses  (neurips-2024/4964cd3f89ee5b57f07f4bc1824d2beb)

[turn 2] search_papers('group DRO reweighting spurious correlation', k=10) ->
     26.45  [icml-2025] Prompting is a Double-Edged Sword: Improving Worst-Group Robustness of Foundation Models  (icml-2025/setlur24a)
     26.40  [icml-2024] Prompting is a Double-Edged Sword: Improving Worst-Group Robustness of Foundation Models  (icml-2024/setlur24a)
     24.96  [neurips-2024] CODA: A Correlation-Oriented Disentanglement and Augmentation Modeling Scheme for Better Resisting S  (neurips-2024/ba06eb7c3ff7cb731ba04c906d212445)
     23.95  [icml-2025] Improving Group Robustness on Spurious Correlation Requires Preciser Group Inference  (icml-2025/han24g)
     23.92  [icml-2024] Improving Group Robustness on Spurious Correlation Requires Preciser Group Inference  (icml-2024/han24g)
     22.17  [icml-2025] Unsupervised Concept Discovery Mitigates Spurious Correlations  (icml-2025/arefin24a)
     22.17  [icml-2024] Unsupervised Concept Discovery Mitigates Spurious Correlations  (icml-2024/arefin24a)
     21.19  [naacl-2025] Fighting Spurious Correlations in Text Classification via a Causal Learning Perspective  (naacl-2025/2025naacl-long215)
     21.19  [neurips-2024] The Group Robustness is in the Details: Revisiting Finetuning under Spurious Correlations  (neurips-2024/dc4db2ff2c1aefce3b594f821ea82fe6)
     20.59  [acl-2024] ASPIRE: Language-Guided Data Augmentation for Improving Robustness Against Spurious Correlations  (acl-2024/2024findings-acl22)

[turn 2] search_papers('balanced group sampling spurious correlation', k=10) ->
     27.04  [icml-2024] Unsupervised Concept Discovery Mitigates Spurious Correlations  (icml-2024/arefin24a)
     27.04  [icml-2025] Unsupervised Concept Discovery Mitigates Spurious Correlations  (icml-2025/arefin24a)
     26.51  [neurips-2024] The Group Robustness is in the Details: Revisiting Finetuning under Spurious Correlations  (neurips-2024/dc4db2ff2c1aefce3b594f821ea82fe6)
     23.95  [icml-2024] Prompting is a Double-Edged Sword: Improving Worst-Group Robustness of Foundation Models  (icml-2024/setlur24a)
     23.95  [icml-2025] Improving Group Robustness on Spurious Correlation Requires Preciser Group Inference  (icml-2025/han24g)
     23.92  [icml-2024] Improving Group Robustness on Spurious Correlation Requires Preciser Group Inference  (icml-2024/han24g)
     22.87  [neurips-2025] The Impact of Coreset Selection on Spurious Correlations and Group Robustness  (neurips-2025/8ae3fd80c07b1c5fdc0ce59b55396812-abstract-datasetsandbenchmarkstrackhtml)
     22.81  [neurips-2024] Mitigating Spurious Correlations via Disagreement Probability  (neurips-2024/879c5890a9d2ecdcb590c9674cda4a59)
     22.73  [icml-2025] Prompting is a Double-Edged Sword: Improving Worst-Group Robustness of Foundation Models  (icml-2025/setlur24a)
     22.57  [aaai-2025] Boosting Test Performance with Importance Sampling-a Subpopulation Perspective  (aaai-2025/150b28ab6b1d5a47f8126f1073afdf668e5bb9f6)

[turn 2] search_papers('minority group oversampling worst group', k=10) ->
     29.46  [neurips-2024] The Group Robustness is in the Details: Revisiting Finetuning under Spurious Correlations  (neurips-2024/dc4db2ff2c1aefce3b594f821ea82fe6)
     25.77  [iclr-2024] Language-Interfaced Tabular Oversampling via Progressive Imputation and Self-Authentication  (iclr-2024/8f6bws5jby)
     24.68  [neurips-2025] Concentration and excess risk bounds for imbalanced classification with synthetic oversampling  (neurips-2025/e715ad3d527e246af3ab286c3f3d367c)
     23.24  [icml-2025] Differentially Private Worst-group Risk Minimization  (icml-2025/zhou24b)
     23.14  [icml-2024] Differentially Private Worst-group Risk Minimization  (icml-2024/zhou24b)
     22.85  [iclr-2024] Like Oil and Water: Group Robustness Methods and Poisoning Defenses May Be at Odds  (iclr-2024/rm9vjpb20f)
     22.17  [neurips-2025] FairNet: Dynamic Fairness Correction without Performance Loss via Contrastive Conditional LoRA  (neurips-2025/81f2d59479a96afd8056db9468254515)
     21.93  [icml-2025] Improving Group Robustness on Spurious Correlation Requires Preciser Group Inference  (icml-2025/han24g)
     21.91  [icml-2024] Improving Group Robustness on Spurious Correlation Requires Preciser Group Inference  (icml-2024/han24g)
     21.44  [aaai-2024] Who's the (Multi-)Fairest of Them All: Rethinking Interpolation-Based Data Augmentation Through the   (aaai-2024/565377f83dd38dd080b675078e181cf34e5b13c9)

[turn 3] submit_report -> 1 mechanism(s)

## Report (as injected)

## Cross-domain mechanism suggestions (analogy search on this task's structure)

Structural properties of this task that the suggestions address:
0. The evaluation score is a lower-tail-sensitive aggregate of conditional ROC ranking problems rather than the population pointwise classification objective. — evidence: The evaluation defines subgroup, BPSN, and BNSP AUC per identity, then aggregates them with a generalized mean; it explicitly identifies the cross-subset confusions represented by BPSN and BNSP.
1. Identity mention is an observed training-time shortcut attribute correlated with the target, while the desired score must remain discriminative within and across identity-bearing strata and cannot consume identity columns at inference. — evidence: The description states that models learned identity names as toxicity signals and gives a non-toxic identity-mention example; train.csv has identity columns, whereas test.csv has only id and comment_text.
2. Metric-defining identity-by-label strata are comparatively sparse inside a 1.8M-example corpus, so uniform minibatches underrepresent them while exhaustive expensive corrective optimization is constrained by the single-V100 six-hour cap. — evidence: train.csv contains 1,804,874 rows and the displayed identity columns each have 1,399,744 missing values; the per-solution budget is six hours on one 32-GiB V100.

### Metric-stratum importance sampling
*Addresses property 0. Source: Boosting Test Performance with Importance Sampling-a Subpopulation Perspective (aaai-2025, `aaai-2025/150b28ab6b1d5a47f8126f1073afdf668e5bb9f6`); The Group Robustness is in the Details: Revisiting Finetuning under Spurious Correlations (neurips-2024, `neurips-2024/dc4db2ff2c1aefce3b594f821ea82fe6`)*

**Shared problem structure**: Population ERM overweights the prevalent shortcut-consistent strata; changing the training sample mixture gives greater optimization exposure to the rare shortcut-breaking strata that determine the low conditional ranking scores.
**Object mappings (this pipeline ↔ source)**: subpopulations defined by a hidden/spurious attribute ↔ training comments partitioned by identity mention and toxicity status (The relevant minority strata are precisely those in which the identity--target correlation differs from the population shortcut.); importance-sampled examples / subpopulation estimator ↔ weighted draws in the fine-tuning dataloader (Sampling controls the contribution of each conditional stratum without requiring a second model or inference-time group input.); worst-subpopulation performance under spurious correlation ↔ the generalized mean of subgroup, BPSN, and BNSP ROC-AUCs (Both objectives are dominated by failures on conditionally defined minority groups rather than mean population performance.)
**Mechanism in the source**: The importance-sampling paper formulates subpopulation/spurious-correlation failure as a sampling problem and reports that a single importance-sampling estimator can improve subgroup performance in both attribute-known and attribute-unknown settings. The finetuning study independently cautions that naive class-balancing by either minibatch upsampling or loss upweighting can reduce worst-group performance as training proceeds, so the mixture must be evaluated against the actual subgroup criterion rather than assumed beneficial.
**Proposed intervention here**: Add one non-standard component: a weighted sampler for the single ModernBERT BCE fine-tune. On the training rows with identity annotations, assign sampling mass by the four coarse cells {identity present/absent} × {toxic/non-toxic}, explicitly increasing the two cross-cells that supply BPSN/BNSP comparisons; retain a substantial uniform/background component rather than fully balancing every identity and label. Keep the model, prediction head, and BCE unchanged, but select the checkpoint using a local reproduction of the competition generalized-mean AUC; if the property is real, BPSN and BNSP AUCs should improve materially even if global AUC is flat or slightly lower.
**Feasibility with the available data**: Feasible. It uses only existing train-time identity annotations and alters dataloader probabilities, so no test-time identity feature, new modality, extra model, or additional full training pass is required. It is deliberately a conservative mixture rather than exhaustive per-identity oversampling because the cited finetuning results show naive balancing can harm worst-group performance; the validation metric decides whether the mixture is retained.


## Report (raw JSON)

```json
{
  "bottlenecks": [
    {
      "statement": "The evaluation score is a lower-tail-sensitive aggregate of conditional ROC ranking problems rather than the population pointwise classification objective.",
      "objects": [
        "continuous toxicity score",
        "toxic and non-toxic examples",
        "identity-conditioned subgroup/BPSN/BNSP subsets",
        "overall ROC-AUC",
        "generalized mean of bias AUCs"
      ],
      "relations": [
        "Each AUC is determined by score orderings between positive and negative examples inside a particular conditional subset, and the generalized mean penalizes weak subsets disproportionately. Uniform pointwise BCE over the population can raise overall AUC while retaining poor BPSN or BNSP ordering."
      ],
      "evidence": "The evaluation defines subgroup, BPSN, and BNSP AUC per identity, then aggregates them with a generalized mean; it explicitly identifies the cross-subset confusions represented by BPSN and BNSP."
    },
    {
      "statement": "Identity mention is an observed training-time shortcut attribute correlated with the target, while the desired score must remain discriminative within and across identity-bearing strata and cannot consume identity columns at inference.",
      "objects": [
        "comment text",
        "identity-mention annotations",
        "toxicity target",
        "identity-bearing toxic/non-toxic comments",
        "test-time text-only input"
      ],
      "relations": [
        "The training distribution makes identity mention predictive of the label, so ordinary ERM can use it as a shortcut. The metric penalizes the resulting conditional ranking failures, while test-time rows have no identity annotation to supply as a feature."
      ],
      "evidence": "The description states that models learned identity names as toxicity signals and gives a non-toxic identity-mention example; train.csv has identity columns, whereas test.csv has only id and comment_text."
    },
    {
      "statement": "Metric-defining identity-by-label strata are comparatively sparse inside a 1.8M-example corpus, so uniform minibatches underrepresent them while exhaustive expensive corrective optimization is constrained by the single-V100 six-hour cap.",
      "objects": [
        "majority comments",
        "identity-annotated examples",
        "identity-by-label strata",
        "minibatches",
        "pretrained encoder",
        "training-time budget"
      ],
      "relations": [
        "Uniform sampling targets population risk and may yield too few subgroup-negative/background-positive and subgroup-positive/background-negative comparisons per update. The first run needs to redirect existing updates toward these strata, rather than multiply models, passes, or tuning trials."
      ],
      "evidence": "train.csv contains 1,804,874 rows and the displayed identity columns each have 1,399,744 missing values; the per-solution budget is six hours on one 32-GiB V100."
    }
  ],
  "mechanisms": [
    {
      "bottleneck_idx": 0,
      "title": "Metric-stratum importance sampling",
      "paper_ids": [
        "aaai-2025/150b28ab6b1d5a47f8126f1073afdf668e5bb9f6",
        "neurips-2024/dc4db2ff2c1aefce3b594f821ea82fe6"
      ],
      "object_mappings": [
        {
          "source": "subpopulations defined by a hidden/spurious attribute",
          "target": "training comments partitioned by identity mention and toxicity status",
          "rationale": "The relevant minority strata are precisely those in which the identity--target correlation differs from the population shortcut."
        },
        {
          "source": "importance-sampled examples / subpopulation estimator",
          "target": "weighted draws in the fine-tuning dataloader",
          "rationale": "Sampling controls the contribution of each conditional stratum without requiring a second model or inference-time group input."
        },
        {
          "source": "worst-subpopulation performance under spurious correlation",
          "target": "the generalized mean of subgroup, BPSN, and BNSP ROC-AUCs",
          "rationale": "Both objectives are dominated by failures on conditionally defined minority groups rather than mean population performance."
        }
      ],
      "shared_relations": "Population ERM overweights the prevalent shortcut-consistent strata; changing the training sample mixture gives greater optimization exposure to the rare shortcut-breaking strata that determine the low conditional ranking scores.",
      "mechanism": "The importance-sampling paper formulates subpopulation/spurious-correlation failure as a sampling problem and reports that a single importance-sampling estimator can improve subgroup performance in both attribute-known and attribute-unknown settings. The finetuning study independently cautions that naive class-balancing by either minibatch upsampling or loss upweighting can reduce worst-group performance as training proceeds, so the mixture must be evaluated against the actual subgroup criterion rather than assumed beneficial.",
      "intervention": "Add one non-standard component: a weighted sampler for the single ModernBERT BCE fine-tune. On the training rows with identity annotations, assign sampling mass by the four coarse cells {identity present/absent} × {toxic/non-toxic}, explicitly increasing the two cross-cells that supply BPSN/BNSP comparisons; retain a substantial uniform/background component rather than fully balancing every identity and label. Keep the model, prediction head, and BCE unchanged, but select the checkpoint using a local reproduction of the competition generalized-mean AUC; if the property is real, BPSN and BNSP AUCs should improve materially even if global AUC is flat or slightly lower.",
      "feasibility": "Feasible. It uses only existing train-time identity annotations and alters dataloader probabilities, so no test-time identity feature, new modality, extra model, or additional full training pass is required. It is deliberately a conservative mixture rather than exhaustive per-identity oversampling because the cited finetuning results show naive balancing can harm worst-group performance; the validation metric decides whether the mixture is retained."
    }
  ]
}
```
