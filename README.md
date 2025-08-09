# Research Overview: Enhancing Non-English Accuracy in LLMs through Thoughts of Words (ToW)

## 1. Introduction
This research addresses the challenges posed by **English-centric bias** in large language models (LLMs). The core idea is to improve the **accuracy of non-English language models** through a novel approach called **Thoughts of Words (ToW)**. This method uses English as an intermediary to enhance reasoning and improve the quality of output in languages other than English.

## 2. Current Challenges in LLMs

### English-Centric Bias
- LLMs are predominantly trained on English datasets, creating an **English language bias** that impacts the performance in other languages. This leads to significant gaps in understanding and generating text in non-English languages.
- Users interacting with LLMs in languages other than English often receive **less accurate and relevant outputs**.

### Accuracy Disparities
- There is a notable **performance gap** between English and non-English outputs, which exacerbates the existing **AI divide**. The difference in language capabilities leads to inequities, especially for linguistically diverse communities. 
- Addressing this issue is critical for ensuring **equitable access to AI benefits** globally.

## 3. The ToW Approach: Leveraging English for Multilingual Accuracy

### Using English as a Cognitive Intermediary
- The approach proposes using **English-based cognitive processing** during the reasoning phase, referred to as **Thoughts of Words (ToW)**.
- This intermediary step allows the model to harness its **strong English capabilities** to **enhance output generation in non-English languages**.

### How ToW Works
- **ToW mechanism**: During reasoning, the model generates intermediate "thought tokens" in English that simplify complex reasoning processes. These tokens serve as a **cognitive bridge**, refining the processing of information before generating output in the target language.
- This mechanism aims to clarify and improve **comprehension and generation in non-English languages** by creating a clearer cognitive pathway.

## 4. Data Collection and Model Training Strategy

### Data Augmentation with ToW
- **Data collection** involves leveraging existing parallel corpora (e.g., English ↔ Korean) and augmenting them with **ToW**. This ensures that language models can effectively bridge English reasoning to other languages.
- **ToW generation** will involve creating new datasets through **self-generated tokens**, ensuring efficient learning and language transition.

### Using Open-Source Models
- Open-source models like **DeepSeek, Llama, and Qwen (70B)** will be leveraged for **cost-efficient training**. This eliminates the need for expensive API integrations, making it accessible for widespread use.
- These models will be trained and fine-tuned using **large-scale parallel datasets** and ToW-enhanced data.

## 5. Implementation and Fine-Tuning

- The **training methodology** will focus on **fine-tuning** large-scale models using augmented datasets generated with ToW.
- By implementing **continuous learning** and **active data augmentation**, the goal is to reduce the performance gap between languages without incurring significant costs.

## 6. Benchmarks for Evaluation

### Zero-Shot Testing
- The models will undergo **zero-shot testing** on multiple **multilingual tasks**, focusing on Korean, Chinese, and other target languages.
- Evaluation metrics will include **BLEU scores** for translation and **BERT Scores** for linguistic accuracy.

### Human Evaluation
- To ensure real-world effectiveness, **human evaluation** will be conducted to assess the contextual accuracy and relevance of the generated outputs.

## 7. Conclusion and Hypothesis
By applying **ToW** for multilingual AI tasks, this research aims to **enhance the inclusivity and accuracy** of LLMs in non-English languages. This model can **bridge the AI divide**, ensuring more **equitable access** to high-quality AI-powered services worldwide.
