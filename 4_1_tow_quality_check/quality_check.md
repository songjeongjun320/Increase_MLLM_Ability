3 TOW: Thoughts Of Words
3.1 Overview
TOWs are word-level fine-grained thoughts describing what the next word should be, given the current
observed contexts. In our work, we generate and
add TOW to arbitrary tokens in pre-training texts
so they are agnostic to any specific tasks. Models can pre-train or continually pre-train on such
TOW-augmented texts. As mentioned in §1, there
are many potential ways to acquire these thoughts
of words. In our work, however, we only discuss
and use distillation as the first step in exploring this
direction. The distillation generation pipeline is
overviewed in Fig. 2. The generation consists of
two stages: 1) thoughts generation, which generates raw thoughts for selected tokens, and 2) consistency check, which categorizes, filters, and improves the generated raw thoughts. We now describe these two components in detail.
3.2 Thoughts Generation
Our experiments are based on two pre-training corpora, OpenWebMath (Paster et al., 2024) and C4
(Dodge et al., 2021), as they are known to have a
great number of reasoning tokens in mathematics
and common sense domains. We randomly select
words from raw documents of these pre-training
corpora and give GPT-4o the contexts before the
selected words. Given the context before each selected word, we ask GPT-4o to elaborate on what
it believes the next word should be, followed by its
prediction. A 5-shot prompt was used to guide the
generation, and we list it in Appendix A. We use
the one-word-at-a-time annotation method instead
of the more efficient method of providing the entire document to create an information bottleneck
that prevents the model from seeing the actual next
word. This way, we can collect the highest-quality
possible thoughts of words by forcing the model
to reason and close the artificial information gap
instead of providing superficial paraphrases.
3.3 Consistency Check
However, as there are inconsistencies between generated thoughts and actual observed next words,
we propose a consistency check step to reduce the
noises in the generated thoughts and provide finegrained categorizations as described in §1, primarily done by using GPT-4o-mini to compare the
generated content with the actual observed next
word. The words are first classified as trivial and
non-trivial by the stopwords list in spaCy.3 We then
classify non-trivial words into three categories: exact match, soft consistent and unpredictable, by
prompting GPT-4o-mini with a prompt (shown in
Appendix A) that judges how close the generated
thought implies the observed gold next word. The
categorization process is also illustrated in Fig. 2.
Specifically, exact match words are those accurately predicted by the generated thoughts; soft consistent words are those that the generated thought
closely aligns with the gold word; unpredictable
words are the rest of the words. Such categorization
is inspired by Kadavath et al. (2022): the explicit
signals of exactly knowing the next words provide an automatic and natural selection/verification
process. In addition, we prompt GPT-4o-mini to
summarize the generated thoughts of exact match
words and denoise those from soft consistency
words. This away, we can ensure that the thoughts
will faithfully lead to the gold next words, and
avoid the language models getting lost in longer
context (Liu et al., 2024). The corresponding
prompts are listed in Appendix A.
3.4 Manual Analysis
To investigate the biases of our LLM-as-judge-style
(Ye et al., 2024) consistency checker, we sample
200 examples from the generated data and manually annotate the consistency between generated
thoughts and gold next words, i.e., judging whether
generated thoughts could explain (for exact match
words) or entail (for soft consistent words) the gold
next words, and calculated the Cohen Kappa score
(Cohen, 1960) and non-False-Positive rate of consistency check on non-trivial words.
non-False-Positive Rate = 1 −
false positive
all examples
Table 1 shows that GPT-4o-mini only reaches the
fair agreement (> 40) with humans on consistency
check, but the noisy data, i.e., which are considered
as consistent by model but not human annotators,
are approximately less than 25%. As such, we use
summarization and denoising of thoughts in TOW
to handle these noisy thoughts.

Prompt for Consistency Check
Task Instruction: Given the following certain text, thought for its
next word and the gold next word, you need to judge whether the
thought for generating the next word is consistent based on the
reasoning process and the given text. For consistency, we mean that
the thought only needs to generally entail the gold next word in
reasoning and does NOT need to be specific on the gold next words.
Context: {<context>}
Thought: {<thought>}
Gold Next Word: {<next_word>}
Now please give me your reasoning and judgement, i.e. True or
False, for the consistency of thought and gold next word based on
the above information.
Reasoning: Let’s think step by step.
Judgement:
Prompt for Summarization Prompt (exact match
words)
Task Instruction: Please modify the following thought into a shorter
one within 15 words without changing much of the meaning. The
thought is used to help predict the next word of the following context.
Context: {<context>}
Thought: {<thought>}
Shorter Thought:
Prompt for Summarization and Denoising Prompt
(soft consistent words)
Task Instruction: Please modify the following thought into a shorter
one within 15 words without changing much of the meaning. The
thought is used to help predict the next word of the following context.
Besides, the gold next word is also given. You should try to shorten
the thought based on it.
Context: {<context>}
Thought: {<thought>}
Gold Next Word: {<gold_next>}
Shorter Thought:
