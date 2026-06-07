
# Emotion Text Classification Using Llama3-8b and LoRA

## Table of Contents

- [Introduction](#introduction)
- [Background](#background)
- [Key Features](#key-features)
- [Methods](#methods)
- [Experimentation](#experimentation)
- [Setup](#setup)
- [Usage](#usage)
- [Dataset](#dataset)
- [Conclusion](#conclusion)
- [Repository Notes](#repository-notes)
- [License](#license)
- [Contact](#contact)
## Introduction

This project explores emotion text classification using the Llama3-8b model, enhanced with LoRA and FlashAttention techniques. The model is optimized for identifying six emotion categories: joy, sadness, anger, fear, love, and surprise. The Llama3-8b model demonstrates superior performance with an accuracy of 0.9262, surpassing other transformer models such as Bert-Base, Bert-Large, Roberta-Base, and Roberta-Large.

## Background

Natural Language Processing (NLP) has become a key focus area for sentiment analysis, also known as sentiment classification or sentiment detection. This technology helps businesses understand consumer emotions and opinions, enhancing customer satisfaction and product development. The vast amount of data in large companies makes manual analysis impractical, leading to the adoption of AI and NLP algorithms.

## Key Features

- **Model**: Llama3-8b, fine-tuned using supervised learning.
- **Techniques**: Utilized LoRA for efficient parameter tuning and FlashAttention for optimized attention computation.
- **Dataset**: Emotion text dataset with six categories.
- **Performance**: Achieved an accuracy of 0.9262, surpassing other NLP models.

## Methods

<div align="center">
    <img src="fig1.png" alt="Architecture of Llama3-8b" width="250">
    <br>
    <b>Figure 1: Architecture of Llama3-8b</b>
</div>

### Llama3-8b Model

The Llama3-8b model, developed by Meta AI, is a large language model optimized for dialogue use cases. It contains 8 billion parameters and features significant improvements over previous models. The Llama3 series incorporates a multi-phase training process that includes pretraining, supervised fine-tuning, and iterative refinement using reinforcement learning with human feedback (RLHF). This process ensures that the model aligns closely with human preferences for helpfulness and safety.

The architectural advancements in Llama3 include the implementation of Grouped-Query Attention (GQA). GQA clusters queries to share key-value pairs, thus reducing memory and computational costs while maintaining high performance. This method significantly enhances the efficiency of attention calculations, particularly in large-scale models.

Llama3-8b is pretrained on a diverse dataset comprising more than 15 trillion tokens from publicly available data, with the model's knowledge cutoff set at March 2023. The fine-tuning phase utilized publicly available instruction datasets and over 10 million human-annotated examples, ensuring a robust understanding of various language tasks.

<div align="center">
    <table>
        <caption><b>Table 1: Llama3-8b Model Details</b></caption>
        <tr>
            <th>Feature</th>
            <th>Specification</th>
        </tr>
        <tr>
            <td>Training Data</td>
            <td>Publicly available data</td>
        </tr>
        <tr>
            <td>Parameters</td>
            <td>8B</td>
        </tr>
        <tr>
            <td>Context Length</td>
            <td>8k</td>
        </tr>
        <tr>
            <td>GQA</td>
            <td>Yes</td>
        </tr>
        <tr>
            <td>Token Count</td>
            <td>15T+</td>
        </tr>
        <tr>
            <td>Knowledge Cutoff</td>
            <td>March 2023</td>
        </tr>
    </table>
</div>

### Instruction Fine-Tuning

Instruction fine-tuning enhances the model's zero-shot learning capabilities across diverse tasks. This technique involves training the model on datasets specifically designed to improve its ability to follow instructions. For example, models trained on datasets like Alpaca-7B can exhibit behaviors similar to OpenAI's text-davinci-003 in understanding and executing instructions.

### LoRA Method for Training

LoRA (Low-Rank Adaptation) is a technique used to integrate trainable rank decomposition matrices into each layer of the Transformer architecture. This method significantly reduces the number of trainable parameters while adapting large language models to specific tasks or domains. Unlike full fine-tuning, LoRA keeps the pretrained model weights unchanged, updating only the low-rank matrices during the adaptation process. This approach enhances training efficiency, reduces storage needs, and does not increase inference latency compared to fully fine-tuned models.

<div align="center">
    <img src="fig2.png" alt="LoRA Training Method" width="350">
    <br>
    <b>Figure 2: LoRA Training Method</b>
</div>

### Flash Attention V2

FlashAttention V2 is an optimization technique designed to accelerate the attention mechanism in Transformer models. It focuses on improving computational efficiency and reducing memory usage during training. FlashAttention achieves this by breaking down attention computation into smaller, more manageable chunks, thereby enhancing cache utilization and reducing memory access. Additionally, it employs sparse matrix operations to leverage the sparsity in attention mechanisms, which helps bypass unnecessary computations. Pipelined operations enable parallel execution of different computation stages, further minimizing processing time.

## Experimentation

<div align="center">
    <img src="fig3.png" alt="Emotion Text Label Distribution" width="450">
    <br>
    <b>Figure 3: Emotion Text Label Distribution</b>
</div>

### Data Analysis

The dataset used for training the model consists of text labeled with six emotions: joy, sadness, anger, fear, love, and surprise. The distribution of the dataset is relatively balanced, with "Joy" being the most common emotion and "Surprise" the least. This balanced distribution provides a strong foundation for the model to accurately classify emotions without bias towards any particular category.

### Experiment Settings

The Llama3-8b model's hyperparameters are set as follows:

<div align="center">
    <table>
        <caption><b>Table 2: Experiment Settings for Llama3-8b</b></caption>
        <tr>
            <th>Parameter</th>
            <th>Setting</th>
        </tr>
        <tr>
            <td>Optimizer</td>
            <td>Adam</td>
        </tr>
        <tr>
            <td>Learning Rate</td>
            <td>5e-5</td>
        </tr>
        <tr>
            <td>Batch Size</td>
            <td>5</td>
        </tr>
        <tr>
            <td>Epochs</td>
            <td>3</td>
        </tr>
        <tr>
            <td>LoRA Rank</td>
            <td>8</td>
        </tr>
        <tr>
            <td>Gradient Accumulation Steps</td>
            <td>4</td>
        </tr>
        <tr>
            <td>Max Length</td>
            <td>512</td>
        </tr>
    </table>
</div>

The model is trained using the Adam optimizer, known for its adaptive learning rate capabilities. A cosine learning rate schedule is employed to adjust the learning rate during training. The batch size is set to 5, with gradient accumulation over 4 steps to optimize memory usage. The model is trained for 3 epochs, with the FP16 precision format used to save GPU memory while maintaining performance. The LoRA rank of 8 indicates the order of the low-rank matrix used in the adaptation process.

### Evaluation Metrics

The primary metric used to evaluate the model's performance is accuracy. This metric measures the proportion of correct predictions made by the model out of all predictions. The formula for accuracy is:

$$
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{FP} + \text{FN} + \text{TN}}
$$

Where:
- TP = True Positive
- FP = False Positive
- FN = False Negative
- TN = True Negative

### Experiment Analysis

The model's performance is compared against other popular NLP models, such as Bert-Base, Bert-Large, Roberta-Base, and Roberta-Large. The Llama3-8b model achieves the highest accuracy of 0.9262, demonstrating the effectiveness of instruction fine-tuning and the model's large parameter set. The superior performance of Llama3-8b in this task underscores the advantages of large language models in achieving high accuracy across diverse and challenging text classification tasks.

<div align="center">
    <table>
        <caption><b>Table 3: Accuracy Results for Different Models</b></caption>
        <tr>
            <th>Model</th>
            <th>Accuracy</th>
        </tr>
        <tr>
            <td>Bert-Base</td>
            <td>0.9063</td>
        </tr>
        <tr>
            <td>Bert-Large</td>
            <td>0.9086</td>
        </tr>
        <tr>
            <td>Roberta-Base</td>
            <td>0.9125</td>
        </tr>
        <tr>
            <td>Roberta-Large</td>
            <td>0.9189</td>
        </tr>
        <tr>
            <td>Llama3-8b</td>
            <td>0.9262</td>
        </tr>
    </table>
</div>

## Setup

This project builds on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). To set up the environment:

```bash
git clone https://github.com/DaoyuanLi2816/Emotion_Text_Classification_Llama3-8B_LoRA.git
cd Emotion_Text_Classification_Llama3-8B_LoRA
pip install -e ".[torch,metrics]"
```
## Conclusion

This project demonstrates the potential of large language models, such as Llama3-8b, in domain-specific tasks like emotion text classification. The model's performance, boosted by specialized techniques like LoRA and FlashAttention, underscores the effectiveness of large models in achieving high accuracy in NLP applications.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

This project is based on modifications to the original work available under [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), which is licensed under the Apache License 2.0.

## Contact

For any questions or issues, please contact Daoyuan Li at lidaoyuan2816@gmail.com.
