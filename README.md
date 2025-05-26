# emotion-detection-nlp-multilabel
Multi-label Emotion Detection using classical neural networks, fine-tuned Transformer models (RoBERTa, DistilBERT, LLaMA, Gemma, Mistral), and QLoRA. Includes comparative analysis, model trade-offs, and zero-shot inference.
# 🧠 Emotion Detection with Deep Learning (Multi-label Classification)

This repository contains the complete workflow for a multi-label **emotion detection** task using tweets as input. The project explores a wide range of models—from simple feed-forward neural networks to transformer-based models, including efficient fine-tuning using **QLoRA** and **zero-shot classification**.

---

## 📂 Project Overview

We trained and evaluated different models across multiple homework assignments to detect multiple emotions from text. Models ranged from scratch-built neural networks to state-of-the-art transformers, all evaluated on efficiency, performance, and training resource constraints.

---

## 📊 Models Summary

### 🔹 HW5: Custom Feed-Forward Neural Network (FFNN)
- Built from scratch for multi-label classification.
- No pre-trained models used.
- Trained using Hugging Face’s `Trainer` API.
- Served as the baseline model.

---

### 🔹 HW6: Encoder-Only Transformer Models

#### 1. `roberta-base`
- Fine-tuned for emotion detection.
- Best performance among encoder-only models.
- Excellent contextual understanding.

#### 2. `distilbert-base-uncased`
- Lighter and faster than RoBERTa.
- Good performance, suitable when resources are limited.

#### 3. `distilroberta-base`
- Balanced speed and performance.
- Similar performance to DistilBERT.

---

### 🔹 HW7: Decoder-Only Models with QLoRA

#### 1. `google/gemma-2-2b`
- High performance and accuracy.
- Memory-efficient fine-tuning using QLoRA.
- High resource usage.

#### 2. `meta-llama/Llama-3.2-1B`
- Balanced model for speed and performance.
- Efficient QLoRA fine-tuning on limited hardware.

#### 3. `MTEB (Mistral)`
- Lightweight and optimized for benchmark tasks.
- Fastest to train, but slightly lower performance.

---

### 🔹 HW8: Advanced Decoder Models & Zero-Shot Inference

#### 1. `Qwen/Qwen2.5-0.5B`
- Small and efficient.
- Struggled with complex patterns.

#### 2. `meta-llama/Llama-3.2-1B-Instruct`
- Instruction-tuned and efficient.
- Solid results with moderate training needs.

#### 3. `meta-llama/Meta-Llama-3.1-8B-Instruct`
- Zero-shot classification.
- Best accuracy and F1 scores without fine-tuning.
- High resource consumption.

---

## 📈 Performance Analysis

### HW5:
- FFNN performed decently on basic inputs.
- Struggled with overlapping and contextual emotion patterns.

### HW6:
- **RoBERTa**: Best overall results.
- **DistilBERT & DistilRoBERTa**: Fast, efficient alternatives.

### HW7:
- **Gemma**: Top performance, high cost.
- **LLama**: Balanced approach.
- **Mistral**: Fastest and most resource-friendly.

### HW8:
- **Meta-LLaMA 8B**: Top performance via zero-shot.
- **Qwen**: Lightweight, but lower accuracy.
- **LLama-Instruct**: Solid middle ground.

---

## 🔮 Future Approaches & Lessons Learned

### HW5
- Use LSTMs, Transformers, or pre-trained embeddings (BERT, GloVe).
- Ensemble methods could improve accuracy.
- Simple FFNNs are a good baseline but limited in handling context.

### HW6
- Try weighted loss or oversampling for class imbalance.
- Smaller models are effective when optimized.
- Model choice significantly impacts outcomes.

### HW7
- Explore mixed precision or more aggressive hyperparameter tuning.
- Combine small models (ensemble) for efficiency + accuracy.
- QLoRA enables large model fine-tuning on limited hardware.

### HW8
- Combine large and small models for trade-offs.
- Use prompt engineering for improved zero-shot classification.
- Instruction-tuned models are great when compute is constrained.

---

## 📊 Experiment Tracking

- All experiments tracked using [Weights & Biases](https://wandb.ai/)
- Includes training logs, F1 scores, and learning curves.

> HW8 Workspace: *Weights & Biases* (link can be added here)

---

## 🧩 Tech Stack

- Python, PyTorch, Hugging Face Transformers
- QLoRA for memory-efficient fine-tuning
- Weights & Biases for experiment tracking
- Zero-shot classification for LLMs

---

## 📬 Contact

Feel free to reach out if you'd like to collaborate or ask questions!  
📧 *mowvagangagovardhan14@gmail.com*  
🔗 *https://www.linkedin.com/in/govardhanmowva/*

