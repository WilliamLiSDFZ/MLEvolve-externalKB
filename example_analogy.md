# analogy agent — parent 8fcded93369a45c9b1c8052aa532e434 (invocation 1)

corpus 5cd1d5e7afd0 (38273 papers) | turns 3 | tokens in/out 36111/3454 | 72s | report

## Packet

# SEARCH STATE

## Task (competition description; head and tail)
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

## Current solution — node 8fcded93369a45c9b1c8052aa532e434 (stage: debug)
Design / plan:
Parent error: RuntimeError | Parent analysis: Execution failed before training, validation, checkpointing, or submission generation. Moving ModernBERT-base to CUDA raised torch.OutOfMemoryError because the 22.06 GiB GPU had only 3.44 MiB free, with other processes occupying most GPU memory.

Code summary:
The code loads train/test data, applies normalized text cleaning and text-style feature engineering, standardizes features using training rows, and uses a chronological 12% validation tail while purging normalized-text overlap from training. It trains either a ModernBERT sequence classifier plus residual style head when sufficient CUDA memory is available or a compact hashed-token pooling model otherwise, with an identity-aware pair sampler and combined fractional BCE/differentiable subgroup ranking loss. Validation uses the official Jigsaw metric (overall ROC-AUC plus power means of subgroup, BPSN, and BNSP AUCs), checkpoints the best model, and writes test predictions to submission.csv.

## Validation behaviour
Metric of this node: 0.76995 (higher is better)
Best metric on this branch so far: 0.76995 (higher is better)

Execution summary:
The pipeline completed successfully in one epoch and produced a submission without execution errors or crashes. It achieved an official Jigsaw validation score of 0.7699496982293698, with overall ROC-AUC of 0.809149; the logged warnings were deprecation/regex warnings only.

Tail of the run output:
```
…sion, and has match groups. To actually get the groups, use str.extract.
  repeated_punctuation = clean_text.str.contains(
/workspace/MLEvolve/runs/20260905_011231_jubias-ana-s45/workspace/runfile_2.py:792: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  grad_scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
/workspace/MLEvolve/runs/20260905_011231_jubias-ana-s45/workspace/runfile_2.py:1274: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=amp_enabled):
/workspace/MLEvolve/runs/20260905_011231_jubias-ana-s45/workspace/runfile_2.py:1162: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=amp_enabled):
/workspace/MLEvolve/runs/20260905_011231_jubias-ana-s45/workspace/runfile_2.py:1162: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=amp_enabled):
/workspace/MLEvolve/runs/20260905_011231_jubias-ana-s45/workspace/runfile_2.py:1162: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=amp_enabled):
Execution time: 57 minutes seconds (time limit is 6 hours).
```

## Improvement attempts already made from this node
(none yet)

## Branch trajectory (oldest -> newest)
- draft: metric n/a (buggy) — This merge uses a chronology-based, duplicate-purged validation split because repeated comments and identity-conditioned ranking make ordinary random validation optimistic. ModernBERT is trained with …
- debug: metric n/a (buggy) — Parent error: RuntimeError | Parent analysis: Execution reached the first training backward pass but failed with a CUDA out-of-memory error while running ModernBERT-large. No epoch completed, no valid…
- debug: metric 0.7699496982293698 — Parent error: RuntimeError | Parent analysis: Execution failed before training, validation, checkpointing, or submission generation. Moving ModernBERT-base to CUDA raised torch.OutOfMemoryError becaus…


## Trace

[turn 1] assistant:
### Step 1 — Diagnose

1. **Compute availability forces a representation bottleneck.**  
   - **Objects:** contextual sequence encoder; compact hashed-token pooling fallback; text inputs; GPU memory.  
   - **Relations:** the intended contextual encoder cannot be instantiated under the available memory, so prediction is made with a pooled hashed-token representation that compresses order-sensitive and compositional evidence into a lower-capacity summary. This is especially limiting where the same lexical marker must be interpreted differently from its surrounding context.  
   - **Evidence:** “ModernBERT-base … raised `torch.OutOfMemoryError`” with only “3.44 MiB free”; the implemented fallback is a “compact hashed-token pooling model”; the successful run completed in one epoch.

2. **The optimization is only partially aligned with a tail-sensitive, subset-defined ranking metric.**  
   - **Objects:** fractional BCE term; differentiable subgroup-ranking term; per-subset ROC-AUCs; generalized/power mean aggregator; subgroup-conditioned pair types.  
   - **Relations:** BCE optimizes individual-label likelihood, while the evaluation rewards *relative ordering* in several specifically constructed conditional subsets and penalizes weak subsets through a power mean. A generic subgroup-ranking component need not allocate gradient according to the worst conditional ranking deficits or all three cross-subset pair constructions.  
   - **Evidence:** the pipeline uses “combined fractional BCE/differentiable subgroup ranking loss,” whereas validation is “overall ROC-AUC plus power means of subgroup, BPSN, and BNSP AUCs.”

3. **The learner has incomplete group observability during training.**  
   - **Objects:** group-membership fields; rows with missing group fields; identity-aware sampler/ranking loss; group-conditioned metric.  
   - **Relations:** group-aware sampling and losses require reliable membership, but group fields are present only for part of the training population. Treating absence as non-membership creates corrupted conditional sets; dropping such examples sacrifices most ordinary supervision. This is a partially observed-group learning problem, not simply class imbalance.  
   - **Evidence:** each displayed membership field has `1399744 nan values` among `1804874` rows, while the current design uses an “identity-aware pair sampler.”

### Step 2 — Queries

- Bottleneck 1: `contextual distillation compact sequence model`; `token pooling order sensitive classification`; `memory efficient transformer distillation`; `late interaction lightweight text encoder`
- Bottleneck 2: `worst group risk optimization ranking`; `conditional AUC surrogate optimization`; `tail risk group robust learning`; `pairwise ranking hard group mining`
- Bottleneck 3: `learning with missing sensitive attributes`; `partially observed group robust optimization`; `latent group distributionally robust learning`; `fairness missing group labels`

[turn 1] search_papers('contextual distillation compact sequence model', k=8) ->
     17.57  [acl-2025] Enhancing Cross-Tokenizer Knowledge Distillation with Contextual Dynamical Mapping  (acl-2025/2025findings-acl419)
     16.95  [acl-2025] Just KIDDIN’ : Knowledge Infusion and Distillation for Detection of INdecent Memes  (acl-2025/2025findings-acl1184)
     15.47  [acl-2024] PLaD: Preference-based Large Language Model Distillation with Pseudo-Preference Pairs  (acl-2024/2024findings-acl923)
     14.68  [iccv-2023] ICD-Face: Intra-class Compactness Distillation for Face Recognition  (iccv-2023/yuicd-faceintra-classcompactnessdistillationforfacerecognitioniccv2023paper)
     14.12  [aaai-2025] A Compact Model for Mathematics Problem Representations Distilled from BERT  (aaai-2025/6a073f2017bbd0a983bdb63b19bc63d5cf97f2a2)
     13.20  [iclr-2025] Building, Reusing, and Generalizing Abstract Representations from Concrete Sequences  (iclr-2025/xiuunzrutd)
     13.13  [iccv-2023] Representation Disparity-aware Distillation for 3D Object Detection  (iccv-2023/lirepresentationdisparity-awaredistillationfor3dobjectdetectioniccv2023paper)
     12.87  [neurips-2024] Compact Language Models via Pruning and Knowledge Distillation  (neurips-2024/4822991365c962105b1b95b1107d30e5)

[turn 1] search_papers('token pooling order sensitive classification', k=8) ->
     19.77  [acl-2024] Addressing Order Sensitivity of In-Context Demonstration Examples in Causal Language Models  (acl-2024/2024findings-acl386)
     16.90  [acl-2024] Unveiling Selection Biases: Exploring Order and Token Sensitivity in Large Language Models  (acl-2024/2024findings-acl333)
     16.67  [neurips-2025] Navigating the MIL Trade-Off: Flexible Pooling for Whole Slide Image Classification  (neurips-2025/8bacb23a4f6b6f52dd5c81c15b4e4ac5)
     15.94  [aaai-2025] Maximizing the Position Embedding for Vision Transformers with Global Average Pooling  (aaai-2025/029dcfa6621f2a797c6b25dd29c373393fac0717)
     15.76  [neurips-2025] Pool Me Wisely: On the Effect of Pooling in Transformer-Based Models  (neurips-2025/3f218e63e5aace7043eb9e7c39cd20c3)
     15.56  [icml-2024] TimeMIL: Advancing Multivariate Time Series Classification via a Time-aware Multiple Instance Learni  (icml-2024/chen24af)
     15.52  [icml-2025] TimeMIL: Advancing Multivariate Time Series Classification via a Time-aware Multiple Instance Learni  (icml-2025/chen24af)
     15.52  [aaai-2024] ViCToR: Improving Visual Comprehension via Token Reconstruction for Pretraining LMMs  (aaai-2024/5f4fa6d9c4b97b8e1650195e643a0b4f8c71829e)

[turn 1] search_papers('memory efficient transformer distillation', k=8) ->
     13.93  [neurips-2025] SpectraLDS: Provable Distillation for Linear Dynamical Systems  (neurips-2025/9a3942c235daa9c5f62a8598ae81a946)
     13.00  [icml-2024] Distilling Morphology-Conditioned Hypernetworks for Efficient Universal Morphology Control  (icml-2024/xiong24c)
     12.95  [icml-2025] Distilling Morphology-Conditioned Hypernetworks for Efficient Universal Morphology Control  (icml-2025/xiong24c)
     12.90  [naacl-2025] Efficient One-shot Compression via Low-Rank Local Feature Distillation  (naacl-2025/2025naacl-long291)
     12.16  [aaai-2025] Distillation-Guided Structural Transfer for Continual Learning Beyond Sparse Distributed Memory  (aaai-2025/f1feb244cdb04ca23f2d1f9939154c358ea482a3)
     12.13  [iccv-2023] Fcaformer: Forward Cross Attention in Hybrid Vision Transformer  (iccv-2023/zhangfcaformerforwardcrossattentioninhybridvisiontransformericcv2023paper)
     12.07  [iccv-2023] Remembering Normality: Memory-guided Knowledge Distillation for Unsupervised Anomaly Detection  (iccv-2023/gurememberingnormalitymemory-guidedknowledgedistillationforunsupervisedanomalydetectioniccv2023paper)
     12.02  [aaai-2025] Efficient Attention-Sharing Information Distillation Transformer for Lightweight Single Image Super-  (aaai-2025/eedebe6aeff70fdcc4c38f4a2d5936b5c9affe28)

[turn 1] search_papers('late interaction lightweight text encoder', k=8) ->
     21.34  [naacl-2024] CELI: Simple yet Effective Approach to Enhance Out-of-Domain Generalization of Cross-Encoders.  (naacl-2024/2024naacl-short16)
     17.33  [acl-2025] GRAM: Generative Recommendation via Semantic-aware Multi-granular Late Fusion  (acl-2025/2025acl-long1596)
     16.57  [aaai-2025] Incorporating Token Importance in Multi-Vector Retrieval  (aaai-2025/910200f157300fc5f0d12a92bca3d13feb16d094)
     15.23  [acl-2024] PreFLMR: Scaling Up Fine-Grained Late-Interaction Multi-modal Retrievers  (acl-2024/2024acl-long289)
     14.08  [acl-2025] HELIOS: Harmonizing Early Fusion, Late Fusion, and LLM Reasoning for Multi-Granular Table-Text Retri  (acl-2025/2025acl-long1559)
     13.51  [acl-2024] See It All: Contextualized Late Aggregation for 3D Dense Captioning  (acl-2024/2024findings-acl202)
     13.25  [neurips-2025] LightFair: Towards an Efficient Alternative for Fair T2I Diffusion via Debiasing Pre-trained Text En  (neurips-2025/20bd42d82998bc61732c00452228e814)
     13.02  [neurips-2025] One Stone with Two Birds: A Null-Text-Null Frequency-Aware Diffusion Models for Text-Guided Image In  (neurips-2025/10046f4ea131db47db65ee667b23c079)

[turn 1] search_papers('worst group risk optimization ranking', k=8) ->
     24.61  [icml-2025] Differentially Private Worst-group Risk Minimization  (icml-2025/zhou24b)
     24.58  [icml-2024] Differentially Private Worst-group Risk Minimization  (icml-2024/zhou24b)
     18.60  [iclr-2025] MixMax: Distributional Robustness in Function Space via Optimal Data Mixtures  (iclr-2025/dikphooa2d)
     17.39  [icml-2025] Improving Group Robustness on Spurious Correlation Requires Preciser Group Inference  (icml-2025/han24g)
     17.33  [icml-2024] Improving Group Robustness on Spurious Correlation Requires Preciser Group Inference  (icml-2024/han24g)
     17.29  [iclr-2025] Group Distributionally Robust Dataset Distillation with Risk Minimization  (iclr-2025/3jsu5qxnru)
     17.19  [neurips-2025] Risk Bounds For Distributional Regression  (neurips-2025/3333f5a87d626f8e46af7c4b27363066)
     16.27  [neurips-2024] Soft-Label Integration for Robust Toxicity Classification  (neurips-2024/ac4c13a5c6ec1bba68c9ce1d908321d2)

[turn 1] search_papers('conditional AUC surrogate optimization', k=8) ->
     15.67  [neurips-2024] AUC Maximization under Positive Distribution Shift  (neurips-2024/3f8c7eb848ffec848f3ed2b7ca44915d)
     14.69  [icml-2024] Boosting Offline Optimizers with Surrogate Sensitivity  (icml-2024/dao24b)
     14.58  [neurips-2024] Incorporating Surrogate Gradient Norm to Improve Offline Optimization Techniques  (neurips-2024/0f588c9d9fa34762362697c1dd463294)
     14.52  [icml-2025] Boosting Offline Optimizers with Surrogate Sensitivity  (icml-2025/dao24b)
     14.49  [neurips-2024] AUCSeg: AUC-oriented Pixel-level Long-tail Semantic Segmentation  (neurips-2024/e574db41163e700545ff4114f96b3d7a)
     14.40  [icml-2025] A Federated Stochastic Multi-level Compositional Minimax Algorithm for Deep AUC Maximization  (icml-2025/zhang24aw)
     14.37  [icml-2024] A Federated Stochastic Multi-level Compositional Minimax Algorithm for Deep AUC Maximization  (icml-2024/zhang24aw)
     14.14  [neurips-2025] Consistency Conditions for Differentiable Surrogate Losses  (neurips-2025/c52d0836624a1f7c55cfc5aed8c501f4)

[turn 1] search_papers('tail risk group robust learning', k=8) ->
     18.91  [neurips-2024] Theoretical Investigations and Practical Enhancements on Tail Task Risk Minimization in Meta Learnin  (neurips-2024/9709b85c06f72243cc2ea335c0110a94)
     17.56  [neurips-2024] Robust group and simultaneous inferences for high-dimensional single index model  (neurips-2024/77f2d0c271e508278ea13e24cd8773d5)
     17.20  [iclr-2024] Safe Collaborative Filtering  (iclr-2024/yaruvgexq3)
     16.68  [aaai-2025] Revisiting Graph Contrastive Learning on Anomaly Detection: A Structural Imbalance Perspective  (aaai-2025/261777d3cd6ed165672b9bfeded2277240fc47ed)
     15.44  [aaai-2025] Differentiable Semantic Meta-Learning Framework for Long-Tail Motion Forecasting in Autonomous Drivi  (aaai-2025/77c57748ad891aee2ce3d11811d6acb9863cd1cc)
     15.40  [neurips-2025] Pareto Optimal Risk-Agnostic Distributional Bandits with Heavy-Tail Rewards  (neurips-2025/8c4de96b9169aa869cc102afe31055e8)
     15.32  [aaai-2024] Decoupled Optimisation for Long-Tailed Visual Recognition  (aaai-2024/9578e815851bad56c71bc45c256c45307e0895e8)
     15.18  [iclr-2025] Group Distributionally Robust Dataset Distillation with Risk Minimization  (iclr-2025/3jsu5qxnru)

[turn 1] search_papers('pairwise ranking hard group mining', k=8) ->
     19.10  [acl-2025] ConFit v2: Improving Resume-Job Matching using Hypothetical Resume Embedding and Runner-Up Hard-Nega  (acl-2025/2025findings-acl661)
     18.44  [aaai-2025] BiCA: Effective Biomedical Dense Retrieval with Citation-Aware Hard Negatives  (aaai-2025/3a119e58d53c013780d32ef750a5559e3fbbcc5d)
     18.21  [acl-2025] Hard Negative Mining for Domain-Specific Retrieval in Enterprise Systems  (acl-2025/2025acl-industry72)
     17.92  [aaai-2025] UniME-V2: MLLM-as-a-Judge for Universal Multimodal Embedding Learning  (aaai-2025/23ab3ad922220f0e48a8c5ba7bb92f893a834a8d)
     17.35  [acl-2025] Confidence and Stability of Global and Pairwise Scores in NLP Evaluation  (acl-2025/2025acl-srw3)
     17.22  [iclr-2025] A Statistical Framework for Ranking LLM-based Chatbots  (iclr-2025/raoeub6nw2)
     16.50  [iclr-2024] Mathematical Justification of Hard Negative Mining via Isometric Approximation Theorem  (iclr-2024/udo3k28bew)
     16.25  [neurips-2024] Prediction-Powered Ranking of Large Language Models  (neurips-2024/cd47cd67caa87f5b1944e00f6781598f)

[turn 1] search_papers('learning with missing sensitive attributes', k=8) ->
     14.29  [aaai-2024] Attribute-Missing Graph Clustering Network  (aaai-2024/4eac0846a1c9433eee0d2808bf3be848d0d974e2)
     13.27  [icml-2025] Intersectional Unfairness Discovery  (icml-2025/xu24d)
     13.11  [icml-2024] Intersectional Unfairness Discovery  (icml-2024/xu24d)
     12.72  [neurips-2025] Taming Hyperparameter Sensitivity in Data Attribution: Practical Selection Without Costly Retraining  (neurips-2025/f7dee2af577ab06037891f264157d718)
     12.42  [acl-2025] ADO: Automatic Data Optimization for Inputs in LLM Prompts  (acl-2025/2025findings-acl1340)
     12.42  [neurips-2025] Revisiting LRP: Positional Attribution as the Missing Ingredient for Transformer Explainability  (neurips-2025/801750bc49fdc3d498e9ee63479f315e)
     12.39  [acl-2025] Evaluation of Attribution Bias in Generator-Aware Retrieval-Augmented Large Language Models  (acl-2025/2025findings-acl1087)
     11.98  [iclr-2025] Towards counterfactual fairness through auxiliary variables  (iclr-2025/gpuv1fvzi1)

[turn 1] search_papers('partially observed group robust optimization', k=8) ->
     15.58  [icml-2024] Variational Partial Group Convolutions for Input-Aware Partial Equivariance of Rotations and Color-S  (icml-2024/kim24q)
     15.58  [icml-2025] Variational Partial Group Convolutions for Input-Aware Partial Equivariance of Rotations and Color-S  (icml-2025/kim24q)
     14.59  [icml-2025] Exploration by Optimization with Hybrid Regularizers: Logarithmic Regret with Adversarial Robustness  (icml-2025/tsuchiya24a)
     14.59  [icml-2024] Exploration by Optimization with Hybrid Regularizers: Logarithmic Regret with Adversarial Robustness  (icml-2024/tsuchiya24a)
     14.09  [neurips-2024] Robust Mixture Learning when Outliers Overwhelm Small Groups  (neurips-2024/b577c062bd4f894b7e05fab6440373ed)
     14.02  [iccv-2023] P2C: Self-Supervised Point Cloud Completion from Single Partial Clouds  (iccv-2023/cuip2cself-supervisedpointcloudcompletionfromsinglepartialcloudsiccv2023paper)
     13.99  [iclr-2025] Partially Observed Trajectory Inference using Optimal Transport and a Dynamics Prior  (iclr-2025/h8ho3t3dye)
     13.48  [icml-2024] Fair Classification with Partial Feedback: An Exploration-Based Data Collection Approach  (icml-2024/keswani24a)

[turn 1] search_papers('latent group distributionally robust learning', k=8) ->
     13.25  [neurips-2024] Learning Group Actions on Latent Representations  (neurips-2024/e63309e532688c722177f81e99f94f32)
     13.21  [icml-2025] Controllable Prompt Tuning For Balancing Group Distributional Robustness  (icml-2025/phan24b)
     12.95  [aaai-2024] Providing Fair Recourse over Plausible Groups  (aaai-2024/38db00dfb66b0b1f8181b713573eb243ce2c9e78)
     12.91  [neurips-2024] Communication-Efficient Federated Group Distributionally Robust Optimization  (neurips-2024/29021b06afa4c648ee438584f7ef3e7e)
     12.75  [icml-2024] Controllable Prompt Tuning For Balancing Group Distributional Robustness  (icml-2024/phan24b)
     12.67  [icml-2024] Efficient Algorithms for Empirical Group Distributionally Robust Optimization and Beyond  (icml-2024/yu24a)
     12.64  [icml-2025] Efficient Algorithms for Empirical Group Distributionally Robust Optimization and Beyond  (icml-2025/yu24a)
     12.57  [iclr-2025] MixMax: Distributional Robustness in Function Space via Optimal Data Mixtures  (iclr-2025/dikphooa2d)

[turn 1] search_papers('fairness missing group labels', k=8) ->
     19.50  [neurips-2025] On Group Sufficiency Under Label Bias  (neurips-2025/b78adc1c1558bf344809275854d40fd6)
     18.17  [neurips-2024] Enhancing Robustness of Last Layer Two-Stage Fair Model Corrections  (neurips-2024/fba9133d7fa896ab9414ddd1a6b1ecbf)
     17.85  [icml-2024] Standardized Interpretable Fairness Measures for Continuous Risk Scores  (icml-2024/becker24a)
     17.85  [icml-2025] Standardized Interpretable Fairness Measures for Continuous Risk Scores  (icml-2025/becker24a)
     17.42  [neurips-2025] Fair Minimum Labeling: Efficient Temporal Network Activations for Reachability and Equity  (neurips-2025/01d5b44366f50df931fc14cac12bee60)
     17.18  [icml-2025] FRAPPÉ: A Group Fairness Framework for Post-Processing Everything  (icml-2025/tifrea24a)
     17.14  [icml-2024] FRAPPÉ: A Group Fairness Framework for Post-Processing Everything  (icml-2024/tifrea24a)
     16.37  [aaai-2025] FROC: Building Fair ROC from a Trained Classifier  (aaai-2025/770f3d5748477fdb1a04856827eed0eb120449cb)

[turn 2] read_abstract(['neurips-2024/fba9133d7fa896ab9414ddd1a6b1ecbf', 'icml-2024/han24g', 'icml-2024/yu24a', 'neurips-2024/3f8c7eb848ffec848f3ed2b7ca44915d', 'acl-2025/2025findings-acl419', 'neurips-2024/4822991365c962105b1b95b1107d30e5']) -> 6 abstracts

[turn 2] search_papers('multi objective AUC worst group optimization', k=10) ->
     18.12  [icml-2025] Controllable Prompt Tuning For Balancing Group Distributional Robustness  (icml-2025/phan24b)
     18.05  [icml-2024] Controllable Prompt Tuning For Balancing Group Distributional Robustness  (icml-2024/phan24b)
     17.87  [icml-2025] Differentially Private Worst-group Risk Minimization  (icml-2025/zhou24b)
     17.78  [icml-2024] Differentially Private Worst-group Risk Minimization  (icml-2024/zhou24b)
     17.42  [icml-2025] A Federated Stochastic Multi-level Compositional Minimax Algorithm for Deep AUC Maximization  (icml-2025/zhang24aw)
     17.38  [icml-2024] A Federated Stochastic Multi-level Compositional Minimax Algorithm for Deep AUC Maximization  (icml-2024/zhang24aw)
     17.20  [iclr-2025] MixMax: Distributional Robustness in Function Space via Optimal Data Mixtures  (iclr-2025/dikphooa2d)
     16.98  [neurips-2025] Multi-Expert Distributionally Robust Optimization for Out-of-Distribution Generalization  (neurips-2025/929814ae1e76ae131a473e3f30609fef)
     16.18  [aaai-2025] Towards Macro-AUC Oriented Imbalanced Multi-Label Continual Learning  (aaai-2025/2e0b591f1c77306f4923e9ab2e615fff2317b795)
     16.13  [aaai-2024] Towards Fairer Centroids in K-means Clustering  (aaai-2024/3f9d284f2841cdc83484cb11cd9d425c707ca1a5)

[turn 2] search_papers('adaptive reweighting subgroup AUC fairness', k=10) ->
     19.98  [aaai-2024] FedGCR: Achieving Performance and Fairness for Federated Learning with Distinct Client Types via Gro  (aaai-2024/089b447a1a9d9cf3cd2d9b23faf4280ecc76a6b4)
     19.49  [neurips-2025] Understanding challenges to the interpretation of disaggregated evaluations of algorithmic fairness  (neurips-2025/3bc4f613b77f59c3998ac83b9a1b5f8a)
     18.03  [neurips-2025] Size-adaptive Hypothesis Testing for Fairness  (neurips-2025/8cdc8e07038dcd423d382048b7063d07)
     17.62  [iccv-2023] Identification of Systematic Errors of Image Classifiers on Rare Subgroups  (iccv-2023/metzenidentificationofsystematicerrorsofimageclassifiersonraresubgroupsiccv2023paper)
     17.50  [neurips-2024] Enhancing Robustness of Last Layer Two-Stage Fair Model Corrections  (neurips-2024/fba9133d7fa896ab9414ddd1a6b1ecbf)
     17.30  [aaai-2025] Towards Macro-AUC Oriented Imbalanced Multi-Label Continual Learning  (aaai-2025/2e0b591f1c77306f4923e9ab2e615fff2317b795)
     16.36  [neurips-2024] Dynamic Subgroup Identification in Covariate-adjusted Response-adaptive Randomization Experiments  (neurips-2024/9af09f8aa60f81dad5dcdbafff605b0d)
     16.17  [iclr-2025] No Location Left Behind: Measuring and Improving the Fairness of Implicit Representations for Earth   (iclr-2025/hszaciznb2)
     15.89  [icml-2024] Feature Importance Disparities for Data Bias Investigations  (icml-2024/chang24a)
     15.87  [aaai-2025] FROC: Building Fair ROC from a Trained Classifier  (aaai-2025/770f3d5748477fdb1a04856827eed0eb120449cb)

[turn 2] search_papers('teacher student contextual knowledge distillation classifier', k=10) ->
     28.26  [aaai-2024] Hybrid Data-Free Knowledge Distillation  (aaai-2024/58ee100badd39cbfa4e12aa56d365e8838f4d096)
     27.65  [iclr-2024] Bayes Conditional Distribution Estimation for Knowledge Distillation Based on Conditional Mutual Inf  (iclr-2024/yv6wwebtkr)
     27.02  [iclr-2025] Speculative Knowledge Distillation: Bridging the Teacher-Student Gap Through Interleaved Sampling  (iclr-2025/egjhwyr2tb)
     26.94  [iclr-2024] Toward Student-oriented Teacher Network Training for Knowledge Distillation  (iclr-2024/wswgcw6qkd)
     26.93  [aaai-2025] Adaptive Dual Guidance Knowledge Distillation  (aaai-2025/48988376a62d184be51bca2eab7f34287a7e8966)
     26.88  [iclr-2024] A Good Learner can Teach Better: Teacher-Student Collaborative Knowledge Distillation  (iclr-2024/ixi4j6ltdx)
     26.71  [aaai-2025] Can Students Beyond the Teacher? Distilling Knowledge from Teacher's Bias  (aaai-2025/0720cf8bc135c049f8ee0c7ce5acbd4eba9e7f2e)
     26.69  [aaai-2024] Cooperative Knowledge Distillation: A Learner Agnostic Approach  (aaai-2024/90c07aa7c9379d61b56a348dcc1138d6d5dcd92d)
     26.63  [iccv-2023] ORC: Network Group-based Knowledge Distillation using Online Role Change  (iccv-2023/choiorcnetworkgroup-basedknowledgedistillationusingonlinerolechangeiccv2023paper)
     26.61  [aaai-2025] Multi-Teacher Knowledge Distillation with Reinforcement Learning for Visual Recognition  (aaai-2025/405c8e99df02826a9d1e7f88566bd28f87616566)

[turn 3] submit_report -> 3 mechanism(s)

## Report (as injected)

## Cross-domain mechanism suggestions (analogy search on this node's bottleneck)

Diagnosed bottlenecks of the current solution:
0. GPU-memory contention forces the pipeline to replace the intended contextual sequence encoder with compact hashed-token pooling, losing context-dependent evidence needed to distinguish the same lexical cue under different surrounding language. — evidence: ModernBERT-base failed with CUDA OOM when only 3.44 MiB was free; the successful design explicitly falls back to a compact hashed-token pooling model.
1. The training objective only partially represents the evaluation's conditional, tail-sensitive ranking structure. — evidence: The current pipeline uses combined fractional BCE/differentiable subgroup ranking, but validation is the official overall AUC plus power means of subgroup, BPSN, and BNSP AUCs.
2. Group-conditioned supervision is partially observed: membership fields needed by the identity-aware sampler/loss are missing for most training rows. — evidence: Displayed membership columns each have 1,399,744 NaNs among 1,804,874 training rows, while the pipeline uses an identity-aware pair sampler.

### Offline contextual-teacher distillation into the deployable student
*Addresses bottleneck 0. Source: Enhancing Cross-Tokenizer Knowledge Distillation with Contextual Dynamical Mapping (acl-2025, `acl-2025/2025findings-acl419`); Compact Language Models via Pruning and Knowledge Distillation (neurips-2024, `neurips-2024/4822991365c962105b1b95b1107d30e5`)*

**Shared problem structure**: A resource-limited student cannot retain the teacher's full contextual representation, so it is trained to approximate the teacher's output behavior using an alignment loss.
**Object mappings (this pipeline ↔ source)**: large teacher model in knowledge distillation ↔ the intended contextual ModernBERT-style encoder (It supplies context-sensitive soft predictions that the fallback cannot obtain directly.); compressed/pruned student model ↔ memory-fit hashed-token or small sequence student (It is the model that must train and infer under the constrained accelerator budget.); teacher-student alignment/distillation objective ↔ soft-score supervision over all training comments (It transfers graded contextual decision structure in addition to hard toxicity labels.)
**Mechanism in the source**: The cited distillation work transfers a stronger model's behavior into a smaller model; the cross-tokenizer work specifically addresses sequence/vocabulary mismatch with contextual alignment, while pruning plus distillation reduces model cost with limited retraining data. The common mechanism is to retain expensive contextual behavior in a compact learner rather than substitute a fundamentally different pooled representation without transfer.
**Proposed intervention here**: First obtain out-of-fold soft logits from a contextual teacher only when an uncontended GPU/CPU-offload environment is available. Train the current memory-fit student on a mixture of fractional-label BCE, temperature-scaled teacher-logit KL, and the existing conditional ranking term; oversample the current worst conditional slices when computing KL so transferred behavior does not merely reproduce dominant-population scores. Keep the student for test inference and compare the three conditional validation AUCs against the present fallback.
**Feasibility with the available data**: Conditionally feasible but not feasible on the currently observed GPU state: the available files provide ample unlabeled/labeled text for distillation, but a teacher-forward pass cannot run with 3.44 MiB free. It requires freeing the GPU, CPU/offload teacher inference, or precomputed teacher logits; student training remains feasible.

### Adaptive minimax weighting of metric-defined pair groups
*Addresses bottleneck 1. Source: Efficient Algorithms for Empirical Group Distributionally Robust Optimization and Beyond (icml-2024, `icml-2024/yu24a`); A Federated Stochastic Multi-level Compositional Minimax Algorithm for Deep AUC Maximization (icml-2024, `icml-2024/zhang24aw`); Controllable Prompt Tuning For Balancing Group Distributional Robustness (icml-2024, `icml-2024/phan24b`)*

**Shared problem structure**: Multiple heterogeneous conditional populations share one predictor, and the objective gives disproportionate importance to poorly performing populations; minimax/adaptive weighting reallocates optimization toward the active weak groups while preserving average performance.
**Object mappings (this pipeline ↔ source)**: groups in empirical group-DRO ↔ subgroup, BPSN, BNSP conditional sets for every annotated membership field (Each is a separately evaluated population whose poor performance depresses the aggregate.); group adversary/primal-dual weights ↔ per-slice loss weights updated from held-out conditional AUC deficits (The weights focus optimization on currently weak metric components instead of fixed generic sampler proportions.); deep AUC pairwise objective ↔ within-slice and cross-slice toxic-versus-nontoxic score pairs (AUC is determined by these relative score orderings, not by isolated classification decisions.)
**Mechanism in the source**: Empirical GDRO formulates worst-group learning as a two-level minimax problem with group sampling and primal-dual optimization. Deep AUC work directly optimizes pairwise ranking under imbalance, and controllable group robustness explicitly trades average performance against group robustness rather than committing to fixed weighting.
**Proposed intervention here**: Replace the fixed identity-aware pair-sampling proportions with an epoch-level adversarial weight vector over every metric-defined slice: membership-within-slice, membership-negative/background-positive, and membership-positive/background-negative. In each batch, form only the corresponding positive-negative pairs and apply a pairwise AUC surrogate; after each validation pass, update and normalize slice weights toward conditional AUCs furthest below the current power-mean target, with a capped weight and an overall-AUC BCE/pairwise anchor. Expect the generalized bias component to rise first, particularly the currently weakest BPSN/BNSP slices, while monitoring overall AUC so the robustness weight is not excessive.
**Feasibility with the available data**: Feasible with the available training annotations and current compact model. It adds bookkeeping and pair sampling but no new modalities or test-time group fields; the 57-minute one-epoch run leaves substantial time within the six-hour limit.


## Report (raw JSON)

```json
{
  "bottlenecks": [
    {
      "statement": "GPU-memory contention forces the pipeline to replace the intended contextual sequence encoder with compact hashed-token pooling, losing context-dependent evidence needed to distinguish the same lexical cue under different surrounding language.",
      "objects": [
        "contextual sequence encoder",
        "hashed-token pooling fallback",
        "token sequence and local context",
        "available GPU memory"
      ],
      "relations": [
        "The memory constraint prevents use of the high-capacity context encoder; pooling compresses a sequence into a summary with substantially weaker order- and context-sensitive interactions."
      ],
      "evidence": "ModernBERT-base failed with CUDA OOM when only 3.44 MiB was free; the successful design explicitly falls back to a compact hashed-token pooling model."
    },
    {
      "statement": "The training objective only partially represents the evaluation's conditional, tail-sensitive ranking structure.",
      "objects": [
        "fractional BCE",
        "generic differentiable subgroup-ranking loss",
        "overall and conditional AUCs",
        "power-mean aggregation",
        "conditional positive-negative pairs"
      ],
      "relations": [
        "Pointwise BCE estimates label likelihood, while each reported AUC depends on ordering positive-negative pairs inside a particular conditional population; the power mean concentrates value on weak conditional AUCs, rather than on average pairwise behavior."
      ],
      "evidence": "The current pipeline uses combined fractional BCE/differentiable subgroup ranking, but validation is the official overall AUC plus power means of subgroup, BPSN, and BNSP AUCs."
    },
    {
      "statement": "Group-conditioned supervision is partially observed: membership fields needed by the identity-aware sampler/loss are missing for most training rows.",
      "objects": [
        "group-membership annotations",
        "unannotated rows",
        "identity-aware pair sampler",
        "conditional ranking groups",
        "toxicity labels"
      ],
      "relations": [
        "Conditional loss construction requires membership, but annotation absence is neither verified non-membership nor a usable group label; discarding missing rows loses broad label supervision, whereas treating them as background contaminates conditional comparisons."
      ],
      "evidence": "Displayed membership columns each have 1,399,744 NaNs among 1,804,874 training rows, while the pipeline uses an identity-aware pair sampler."
    }
  ],
  "mechanisms": [
    {
      "bottleneck_idx": 0,
      "title": "Offline contextual-teacher distillation into the deployable student",
      "paper_ids": [
        "acl-2025/2025findings-acl419",
        "neurips-2024/4822991365c962105b1b95b1107d30e5"
      ],
      "object_mappings": [
        {
          "source": "large teacher model in knowledge distillation",
          "target": "the intended contextual ModernBERT-style encoder",
          "rationale": "It supplies context-sensitive soft predictions that the fallback cannot obtain directly."
        },
        {
          "source": "compressed/pruned student model",
          "target": "memory-fit hashed-token or small sequence student",
          "rationale": "It is the model that must train and infer under the constrained accelerator budget."
        },
        {
          "source": "teacher-student alignment/distillation objective",
          "target": "soft-score supervision over all training comments",
          "rationale": "It transfers graded contextual decision structure in addition to hard toxicity labels."
        }
      ],
      "shared_relations": "A resource-limited student cannot retain the teacher's full contextual representation, so it is trained to approximate the teacher's output behavior using an alignment loss.",
      "mechanism": "The cited distillation work transfers a stronger model's behavior into a smaller model; the cross-tokenizer work specifically addresses sequence/vocabulary mismatch with contextual alignment, while pruning plus distillation reduces model cost with limited retraining data. The common mechanism is to retain expensive contextual behavior in a compact learner rather than substitute a fundamentally different pooled representation without transfer.",
      "intervention": "First obtain out-of-fold soft logits from a contextual teacher only when an uncontended GPU/CPU-offload environment is available. Train the current memory-fit student on a mixture of fractional-label BCE, temperature-scaled teacher-logit KL, and the existing conditional ranking term; oversample the current worst conditional slices when computing KL so transferred behavior does not merely reproduce dominant-population scores. Keep the student for test inference and compare the three conditional validation AUCs against the present fallback.",
      "feasibility": "Conditionally feasible but not feasible on the currently observed GPU state: the available files provide ample unlabeled/labeled text for distillation, but a teacher-forward pass cannot run with 3.44 MiB free. It requires freeing the GPU, CPU/offload teacher inference, or precomputed teacher logits; student training remains feasible."
    },
    {
      "bottleneck_idx": 1,
      "title": "Adaptive minimax weighting of metric-defined pair groups",
      "paper_ids": [
        "icml-2024/yu24a",
        "icml-2024/zhang24aw",
        "icml-2024/phan24b"
      ],
      "object_mappings": [
        {
          "source": "groups in empirical group-DRO",
          "target": "subgroup, BPSN, BNSP conditional sets for every annotated membership field",
          "rationale": "Each is a separately evaluated population whose poor performance depresses the aggregate."
        },
        {
          "source": "group adversary/primal-dual weights",
          "target": "per-slice loss weights updated from held-out conditional AUC deficits",
          "rationale": "The weights focus optimization on currently weak metric components instead of fixed generic sampler proportions."
        },
        {
          "source": "deep AUC pairwise objective",
          "target": "within-slice and cross-slice toxic-versus-nontoxic score pairs",
          "rationale": "AUC is determined by these relative score orderings, not by isolated classification decisions."
        }
      ],
      "shared_relations": "Multiple heterogeneous conditional populations share one predictor, and the objective gives disproportionate importance to poorly performing populations; minimax/adaptive weighting reallocates optimization toward the active weak groups while preserving average performance.",
      "mechanism": "Empirical GDRO formulates worst-group learning as a two-level minimax problem with group sampling and primal-dual optimization. Deep AUC work directly optimizes pairwise ranking under imbalance, and controllable group robustness explicitly trades average performance against group robustness rather than committing to fixed weighting.",
      "intervention": "Replace the fixed identity-aware pair-sampling proportions with an epoch-level adversarial weight vector over every metric-defined slice: membership-within-slice, membership-negative/background-positive, and membership-positive/background-negative. In each batch, form only the corresponding positive-negative pairs and apply a pairwise AUC surrogate; after each validation pass, update and normalize slice weights toward conditional AUCs furthest below the current power-mean target, with a capped weight and an overall-AUC BCE/pairwise anchor. Expect the generalized bias component to rise first, particularly the currently weakest BPSN/BNSP slices, while monitoring overall AUC so the robustness weight is not excessive.",
      "feasibility": "Feasible with the available training annotations and current compact model. It adds bookkeeping and pair sampling but no new modalities or test-time group fields; the 57-minute one-epoch run leaves substantial time within the six-hour limit."
    },
    {
      "bottleneck_idx": 2,
      "title": "Two-stage latent-neighbor correction for incomplete group labels",
      "paper_ids": [
        "neurips-2024/fba9133d7fa896ab9414ddd1a6b1ecbf",
        "icml-2024/han24g"
      ],
      "object_mappings": [
        {
          "source": "pretrained base model and its latent nearest-neighbor graph",
          "target": "the trained compact classifier and penultimate text/style embedding graph",
          "rationale": "The embedding defines semantic neighborhoods through which reliability and latent group structure can be propagated."
        },
        {
          "source": "partially unavailable subgroup/spurious-attribute information",
          "target": "membership columns with extensive missingness",
          "rationale": "Both settings lack complete group labels needed for robust correction."
        },
        {
          "source": "two-stage last-layer fairness correction",
          "target": "frozen encoder plus retrained residual/style prediction head",
          "rationale": "A small correction layer can change group-sensitive score geometry without retraining the resource-limited representation."
        }
      ],
      "shared_relations": "Group-aware correction is desirable but membership is incomplete; infer or smooth reliable group-related structure from learned representations, then modify only a lightweight final decision layer.",
      "mechanism": "The two-stage fair-correction paper uses label spreading on a latent nearest-neighbor graph as a low-overhead drop-in step before last-layer retraining, and explicitly evaluates settings with and without group information. GIC likewise argues that robust learning requires more precise group inference rather than blindly relying on coarse pseudo-groups.",
      "intervention": "After fitting the base compact model, freeze its encoder and construct an approximate k-nearest-neighbor graph from its penultimate embeddings for all training comments. Propagate membership confidence only from observed membership labels, retain uncertainty rather than converting missing entries to background, then retrain just the residual/style head with confidence-weighted conditional pair losses and a regularizer that keeps scores close to the base model on low-confidence rows. Evaluate this correction on the chronological tail using only observed memberships and separately report performance by membership-confidence bin; use no group input at test inference.",
      "feasibility": "Feasible: all required text, partial membership fields, and an existing residual head are available; sparse/approximate neighbors are needed at this dataset scale. It needs no test membership labels because correction changes shared model parameters, not per-example group-specific test thresholds."
    }
  ]
}
```
