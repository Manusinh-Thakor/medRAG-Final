# 🧠 MedRAG: Medical Image Analysis and Retrieval-Augmented Generation

**MedRAG** is a modular, multimodal AI system developed for **learning and research purposes** in the field of medical imaging and explainable artificial intelligence. It provides a coordinated pipeline that combines medical image interpretation, diagnostic reasoning, and retrieval of similar cases from both local databases and online sources.

---

## 🩺 Project Vision

The goal of MedRAG is to explore how AI can support medical decision-making by:

* Generating structured diagnostic reasoning from medical images
* Retrieving visually and semantically similar clinical cases
* Expanding context through medically relevant web search

This project is built strictly for **educational exploration**, with a strong emphasis on transparency, explainability, and controlled tool orchestration.

---

## ⚙️ System Pipeline

MedRAG operates in a staged sequence:

### 1. **Image Understanding**

Processes radiological images (e.g., chest X-rays) and outputs a step-wise explanation of the observed findings, along with a suggested diagnosis. The reasoning aims to emulate how a radiologist might explain the decision path.

### 2. **Similar Case Retrieval**

Searches a local medical image database to retrieve clinically similar examples based on the diagnostic content. These examples help ground the diagnosis in prior visual cases.

### 3. **Web-based Evidence Retrieval**

Queries trusted online medical sources to fetch related images or articles, offering broader medical context around the diagnosis.

### System Architecture

![MedRAG Architecture](images/medrag-architecture2.drawio.svg)

---

## 🧠 AI Agent Orchestration

The system is governed by a coordination agent that:

* Determines the necessary tools and order of execution
* Streams each result to the user in real time, step by step
* Enforces domain restrictions (medical-only)
* Avoids final conclusions—only provides evidence and reasoning flow

---

### Live Demo (GIF)

![MedRAG Demo](images/medrag_gif.gif)

## 🎯 Intended Use

MedRAG is intended **only for non-commercial educational and research use**. It is not designed, validated, or authorized for clinical deployment or medical advice. The system exists as a demonstration of how retrieval-augmented AI pipelines can be structured and studied within a controlled setting.

---

## 💡 Core Principles

* **Learning-first**: Developed to study multimodal reasoning and retrieval
* **Transparency**: Every output is labeled and streamed clearly
* **Modularity**: Each component is independently pluggable
* **No hallucination**: Reasoning is supported by retrieval and evidence
* **Medical domain only**: All queries outside medicine are rejected

---

## 📚 References

The MedRAG project draws inspiration and methodology from the following research and tools:

* [LLaVA-Med](https://github.com/microsoft/LLaVA-Med) – Vision-language instruction tuning for medical imaging
* [MedGEMMA](https://huggingface.co/google/medgemma-4b-it) – Open vision-language model fine-tuned for medical image understanding
* [MedCLIP](https://github.com/UCSD-AI4H/MedCLIP) – Medical image-text embedding and retrieval
* [LangChain](https://github.com/langchain-ai/langchain) – Framework for agent-based reasoning
* [LangGraph](https://github.com/langchain-ai/langgraph) – Graph-based control flow for multi-step AI systems
* [Tavily API](https://www.tavily.com/) / [SerpAPI](https://serpapi.com/) – Web search interfaces for medical context retrieval
* NIH Chest X-ray Dataset – Public dataset for medical image research

---

> ⚠️ **Disclaimer**: MedRAG is a research prototype built for academic exploration only. It is **not a diagnostic tool** and must not be used for clinical decision-making.

---
