Hi Professor,

Quick update for this week.

I found two problems with how the system was retrieving papers. First, I was embedding the
whole competition description, which produced a vague query. Second, the paper embeddings
all pointed in nearly the same direction, so the part of each vector that actually carried
topic information was very small. I fixed the first by having an LLM condense the
description into a short task statement, and the second by subtracting the average vector
from every embedding. Together these raised the number of on-topic papers in the top 10
from 3 to 9.

With retrieval fixed, I ran the first proper A/B test on spooky-author-identification. The
knowledge-base run scored 0.288 log loss under official mle-bench grading. The baseline run
just finished and I am scoring it now.

Earlier, on an ADMET prediction task, the knowledge base made results worse. I think the
reason is coverage — only 3 of the 423 topic categories in the corpus were relevant to that
task. So I am now testing on jigsaw toxic-comment classification, where about 390 papers
across four categories are directly on topic. In parallel I am expanding the corpus.
