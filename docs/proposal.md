# Knowledge or Noise? Understanding When Research Literature Helps ML Engineering Agents

*Research proposal, working title. It continues the draft `MLE_agent_paper_KB.pdf` in this folder.*

## Summary

We built a knowledge base of 38,273 research papers and two methods that give this knowledge to ML engineering agents. In the first runs, the analogical reasoning method made the solutions of the agents more diverse, but it did not make their scores higher. This proposal makes this gap the research question. We ask what a research paper does to an agent. Does it give useful content, or does it only move the search to a different place? We also ask when the content helps. The study is mainly an analysis, and it needs CPU time, LLM calls and less than 200 GPU hours.

## 1. What we have now

The knowledge base holds the metadata, the abstracts and the PDF links of 38,273 conference papers: NeurIPS (10,312), AAAI (7,114), ICML (5,219), ACL (5,261) and NAACL (2,248).

Two methods give this knowledge to an agent:

- **Traditional RAG method.** The agent retrieves papers one time, before the first draft. BM25 and dense retrieval find up to 40 papers for a short summary of the task, and an LLM keeps 5 to 15 of them. An LLM then extracts up to 20 techniques with the effect that each paper reports. The agent gets the techniques that have a positive effect.
- **Analogical reasoning method.** At each improvement step, an LLM reads the current code, the training log and the task description, and it finds up to three bottlenecks. It writes each bottleneck as objects and relations, searches the papers with BM25 and reads the papers that it selects. It returns up to three proposals. Each proposal joins a bottleneck to a mechanism from a paper and gives the code change. The agent uses one proposal at most. Before the first draft, the method does the same steps on the structure of the task.

We connected both methods to MLEvolve, which searches with Monte Carlo Graph Search. We also connected the analogical reasoning method to autoresearch, which keeps or discards one change at a time.

We tested four Kaggle text tasks:

- essay scoring, with quadratic weighted kappa;
- toxic comment classification, with mean ROC-AUC;
- toxicity with identity bias (the bias task below), with bias-aware AUC;
- chatbot preference prediction, with log loss.

## 2. What the first results show

**Traditional RAG method (Table 1 of the draft, unpaired means).** Injection at the draft stage only gave a worse score than the baseline on two of the four tasks. Injection at the draft and improvement stages gave a better score on all four tasks. The largest gain is +0.061 on the bias task, from 8, 6 and 5 runs.

**Analogical reasoning method (Tables 2 and 3 of the draft, paired runs).** We computed these values again from the two tables:

| Measure | MLEvolve, bias task (9 pairs) | autoresearch (6 pairs) |
|---|---|---|
| Mean change of the final score | +0.0025; higher in 5 of 9 pairs; Wilcoxon p = 0.73 | -0.0098; higher in 1 of 6 pairs; Wilcoxon p = 0.16 |
| Mean change of the Vendi score | +0.078; higher in 7 of 9 pairs; Wilcoxon p = 0.039 | +0.242; higher in 6 of 6 pairs; Wilcoxon p = 0.031 |

In autoresearch, the standard deviation of the final score between runs is 0.0105 with the method and 0.0031 without it.

The result that repeats in both frameworks is diversity. The analogical reasoning method makes the solutions more diverse, but the score does not increase. In autoresearch, the score is lower, and it changes more from run to run. The draft does not explain this gap, and this proposal starts from it.

**Checks before new runs.** Four points in the draft can change this reading:

- The Vendi score in Table 2 includes failed and pending candidates. Compute it again on valid candidates only. If the diversity gain stops, the gap is not real.
- For the bias task in MLEvolve, the baseline mean is 0.840 in Table 1 and 0.908 in Table 2.
- Table 3 names the toxic comment task, but its composite AUC values (near 0.93) look like the bias task. Table 3 also uses validation scores, and Table 2 uses private-test scores.
- The original autoresearch instructions tell the agent to read the papers that the code refers to when it has no more ideas [17]. If our baseline keeps this instruction, Table 3 compares "read when stuck" with "read at each step".

## 3. Why this question matters

Many ML engineering agents now use external knowledge: web search for models [4], cases from Kaggle [5] and an expert knowledge base [6]. These systems report gains. In them, we did not find a control that separates the content of the knowledge from the other effects of the added text.

Diversity can help these agents. A controlled study used 22 MLE-bench Lite tasks and 10 seeds [2]. It removed the diversity instructions from the prompt, and the medal rate fell by 6.9 and 8.4 points for two agents. That diversity came from the prompt, not from papers, and it was the diversity of the first drafts. Other work shows that the operators of an agent, such as its draft and improvement steps, decide much of its result [3]. Knowledge injection changes these operators.

Analogical reasoning across domains increased solution diversity by 90% to 173% and gave better results on four biomedical problems [1]. Our first data show the diversity gain in ML engineering, but not the score gain.

Added text changes model behavior even when it has no useful content. Filler text can increase accuracy [13], random documents can increase the accuracy of RAG [11], and one distractor can decrease accuracy [12]. Much text that looks irrelevant is not irrelevant when two judges check it [14]. In multi-step agentic search, the static relevance of a document does not predict its causal effect on the agent (Spearman ρ = -0.026) [7].

Research practice also disagrees on when to read. Some guides read the literature before they find a bottleneck [15]. Others read only after their own attempts stop [16]. Mobile agents that retrieve only when an uncertainty score is high gained about 6 points [9]. We found no study that compares these timings for agents with an executable score.

**The gap.** We found no study that measures, at the same agent state, how much of the effect of a paper comes from its content, and when this content effect occurs.

## 4. What we study next

**Main question.** When an ML engineering agent reads a research paper, does it get useful content, or does the text only move its search? When does the content help?

### 4.1 What does a paper do to the agent?

Use saved agent states from earlier runs. An agent state is a parent solution with its code, its log and the context that the agent saw. From each state, run the same improvement step four times. Each time, give one of the four inputs below. Each input adds one thing to the input before it. This replay design follows recent work on counterfactual replay for LLM agents [7, 8].

| Input | What the agent gets |
|---|---|
| No text | The usual prompt only |
| Filler text | Neutral text with the length of a proposal |
| Unrelated paper | A proposal from a paper that two LLM judges mark as unrelated to the task |
| Relevant paper | The proposal from the analogical reasoning method |

Let y be the result of the new solution: its score, its validity, or its distance from the parent and the other new solutions. Then:

- length effect = y(filler text) - y(no text)
- perturbation effect = y(unrelated paper) - y(filler text)
- content effect = y(relevant paper) - y(unrelated paper)

The three effects add up to the total effect that the draft measures. Some new directions start with a lower score and improve later. For some states, continue each new solution for a few steps without injection, and record the best score.

### 4.2 When does a paper help?

For each state, record these features:

- the local improvement rate: the gain of the best score per step over the last few steps;
- the stage: draft or improvement;
- the part of the budget that the agent used;
- the type of bottleneck.

Then examine how the three effects change with these features. Two hypotheses guide this analysis:

- **Stuck states.** The content effect is largest when the local improvement rate is low. Optimal foraging theory gives this prediction: a forager leaves a patch when its rate of gain falls below the average rate of the environment [18, 19].
- **Stage.** Diversity helps first drafts [2]. This means that the perturbation effect can be positive at the draft stage and negative at the improvement stage.

Also compare the retrieval score of each paper with its content effect. In agentic search, relevance does not predict usefulness [7]. This comparison shows if the same is true for ML engineering agents.

### 4.3 A simple timing rule

From the results of 4.2, write one timing rule. An example: read papers only when the local improvement rate falls below a threshold. Set the threshold before the test. Compare this rule with "never read" and "read at each improvement step" in a small number of full runs. Add one more run that gives unrelated papers at the same moments. This run shows if the gain comes from the content or only from the timing.

## 5. Compute plan

The main cost moves from GPU time to LLM calls and CPU time.

- Start with the analyses that need no new runs: the Vendi score on valid candidates, the best score per step, and how often the agent used a proposal.
- Use tasks whose candidates train on CPU in minutes, for example tabular and small text tasks from MLE-bench Lite. These tasks also make the result less specific to text.
- Sections 4.1 and 4.2 need one improvement step and one evaluation for each input, not full runs.
- Use the GPU budget (less than 200 hours on RTX 3090 cards) for one check on a task from the draft. This check shows if the main result also holds there.

## 6. Expected results

We expect most of the diversity gain to come from the perturbation effect. We expect the content effect to occur mainly in stuck states. If this is correct, a rule that reads papers only in stuck states keeps the score of "read at each step" at a lower cost.

Each possible result gives a clear claim:

| If we find | The paper says |
|---|---|
| Content effect near zero, perturbation effect positive | For strong models, papers help mainly because they move the search. Their content adds little. |
| Content effect positive only in stuck states | Papers help when the agent is stuck. The agent must read them at these points. |
| Content effect positive in all states | The content always helps. A timing rule only decreases the cost. |
| No diversity gain on valid candidates | The first gap came from failed candidates. Stop, or change the question. |

The first three results each give an analysis paper. The fourth result shows in the first check, before new runs. In each case, the system of the draft becomes the tool of the study, and its first results become the motivation.

The main figures are:

- the three effects for each task;
- the content effect and the perturbation effect as functions of the local improvement rate;
- the score and the cost of each timing rule.

## 7. Risks

- **Small score differences.** Compare the inputs at the same state, and repeat each input. Use validity and diversity as main measures too.
- **Unrelated papers that are related.** Use two LLM judges, and report their agreement.
- **Strong models already know many of the papers.** With adaptive retrieval, larger models need retrieval for fewer questions [10]. If possible, add a weaker model, or papers that are newer than the training data of the model.
- **Restored states that do not repeat.** First, check that the same input at the same state gives similar results.

## References

1. Andrew Shen, Shaul Druckmann, James Zou. *Unlocking LLM Creativity in Science through Analogical Reasoning.* arXiv:2605.11258, 2026. https://arxiv.org/abs/2605.11258
2. Alexis Audran-Reiss, Jordi Armengol-Estapé, Karen Hambardzumyan, et al. *What Does It Take to Be a Good AI Research Agent? Studying the Role of Ideation Diversity.* arXiv:2511.15593, 2025. https://arxiv.org/abs/2511.15593
3. Edan Toledo, Karen Hambardzumyan, Martin Josifoski, et al. *AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench.* arXiv:2507.02554, 2025. https://arxiv.org/abs/2507.02554
4. Jaehyun Nam, et al. *MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement.* NeurIPS 2025. arXiv:2506.15692. https://arxiv.org/abs/2506.15692
5. Siyuan Guo, Cheng Deng, Ying Wen, Hechang Chen, Yi Chang, Jun Wang. *DS-Agent: Automated Data Science by Empowering Large Language Models with Case-Based Reasoning.* ICML 2024, PMLR 235. https://proceedings.mlr.press/v235/guo24b.html
6. Yixin Ou, Yujie Luo, et al. *AutoMind: Adaptive Knowledgeable Agent for Automated Data Science.* arXiv:2506.10974, 2025. https://arxiv.org/abs/2506.10974
7. Debayan Mukhopadhyay, Utshab Kumar Ghosh, Shubham Chatterjee. *Bridge Evidence: Static Retrieval Utility Does Not Predict Causal Utility in Multi-Step Agentic Search.* arXiv:2607.15253, 2026. https://arxiv.org/abs/2607.15253
8. Jaineet Shah. *Causal Agent Replay: Counterfactual Attribution for LLM-Agent Failures.* arXiv:2606.08275, 2026. https://arxiv.org/abs/2606.08275
9. Sijia Li, Xiaoyu Tan, Shahir Ali, Niels Schmidt, Gengchen Ma, Xihe Qiu. *Curiosity Driven Knowledge Retrieval for Mobile Agents.* arXiv:2601.19306, 2026. https://arxiv.org/abs/2601.19306
10. Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, Hannaneh Hajishirzi. *When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories.* ACL 2023. arXiv:2212.10511. https://arxiv.org/abs/2212.10511
11. Florin Cuconasu, et al. *The Power of Noise: Redefining Retrieval for RAG Systems.* SIGIR 2024. arXiv:2401.14887. https://arxiv.org/abs/2401.14887
12. Kelly Hong, Anton Troynikov, Jeff Huber. *Context Rot: How Increasing Input Tokens Impacts LLM Performance.* Chroma technical report, 2025. https://www.trychroma.com/research/context-rot
13. Ryan Greenblatt. *Recent LLMs can use filler tokens or problem repeats to improve (no-CoT) math performance.* LessWrong, 2025. https://www.lesswrong.com/posts/NYzYJ2WoB74E6uj9L/recent-llms-can-use-filler-tokens-or-problem-repeats-to
14. Sturb. *Revisiting GSM-Symbolic: models seem to reason okay, actually.* LessWrong, 2026. https://www.lesswrong.com/posts/Ze4C99Dasj74YKCFh/revisiting-gsm-symbolic-do-2026-frontier-models-still-fail
15. Qihao Zhao, et al. *ResearchStudio-Idea: An Evidence-Grounded Research-Ideation Skill Suite from ML Conference Outcomes.* arXiv:2607.04439, 2026. https://arxiv.org/abs/2607.04439
16. Jack Morris. *Zen and the Art of AI Research.* Blog post, 2026. https://blog.jxmo.io/p/zen-and-the-art-of-machine-learning
17. Andrej Karpathy. *autoresearch*, `program.md`. https://github.com/karpathy/autoresearch
18. Eric L. Charnov. *Optimal foraging, the marginal value theorem.* Theoretical Population Biology, 9(2):129-136, 1976.
19. Peter Pirolli, Stuart Card. *Information foraging.* Psychological Review, 1999.
