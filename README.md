<p align="center">
  <img src="images/medrag_logo-2.png" alt="MedRAG Logo" width="200"/>
</p>

<h1 align="center">🧠 MedRAG: Medical Image Analysis & Retrieval-Augmented Generation</h1>

---

## 🔍 Overview

**MedRAG** is a modular AI system developed for **educational and research use** in medical imaging, focused on **chest X-rays**. It performs three primary functions:

* Chest X-ray interpretation using fine-tuned models
* Generation of structured diagnostic reasoning
* Retrieval of similar medical images from local and web sources

The system is trained on a **custom-generated dataset created and published by the author**:

> [CXR-10k Reasoning Dataset (Hugging Face)](https://huggingface.co/datasets/Manusinhh/cxr-10k-reasoning-dataset)

This dataset was curated by aggregating over 10,000 chest X-ray images and their associated metadata and report texts from **MIMIC-CXR**. The author generated step-by-step diagnostic reasoning and summary outputs from these findings to build the training corpus.

Details of the dataset preparation, formatting, and annotation guidelines are available on the dataset’s [Hugging Face README](https://huggingface.co/datasets/Manusinhh/cxr-10k-reasoning-dataset).

The reasoning model design and fine-tuning pipeline are inspired by the **Insight-V** methodology:

> [Insight-V GitHub](https://github.com/JoshuaChou2018/Insight-V)
> [Insight-V Paper](https://arxiv.org/abs/2411.14432)

---

## ⚡ Project Motivation

MedRAG was built to minimize the time and effort spent by clinicians when manually searching for similar medical cases. It automates reasoning and retrieval to assist in medical education and early-stage diagnosis support.

---

## 🩺 Project Vision

MedRAG explores the integration of multimodal AI in medicine by:

* Producing interpretable, step-wise diagnostic reasoning from chest X-rays
* Locating visually and semantically similar cases from a local database
* Fetching relevant medical images and content from trusted online sources

---

## 🧱 Flow Diagram

![System Architecture](images/medrag-interface.drawio.svg)

---

## 🧱 System Architecture

![System Architecture](images/medrag-architecture2.drawio.svg)

---

## 🔹 Live Demo

![Live Demo](images/medrag_gif.gif)

---

## 🔄 MedRAG System Pipeline

### 🖥️ Frontend Layer: MedRAG Web UI

* User inputs query or image
* Request forwarded to backend agent

---

### 🧠 AI Coordination Layer: MedRAG Agent

* Runs on AWS EC2
* Orchestrates all reasoning and retrieval steps

---

### 🧠 Stage 1 – Chest X-ray Analysis

#### 🔹 1. MedGEMMA Reasoner ([Link](https://huggingface.co/Manusinhh/medgemma-finetuned-cxr-reasoning))

* **Input**: Chest X-ray image
* **Output**: Structured, step-wise diagnostic reasoning
* **Note**: Fine-tuned by the author using the custom reasoning dataset derived from MIMIC-CXR, following Insight-V methodology

#### 🔹 2. MedGEMMA Summariser ([Link](https://huggingface.co/Manusinhh/medgemma-finetuned-cxr-summerizer))

* **Input**: Reasoning output
* **Output**: Concise diagnostic label or summary
* **Note**: Also fine-tuned by the author using the same dataset

---

### 🔍 Stage 2 – Similar Case Retrieval

* **Tool**: MedCLIP + FAISS
* **Input**: Disease label
* **Output**: Top-matching chest X-rays from local database

---

### 🌐 Stage 3 – Web-Based Context Retrieval

* **Tool**: serpAPI or Tavily API
* **Input**: Disease label
* **Output**: Online medical images and reference content

---

### 📤 Streaming Results to UI

* Real-time updates returned to the interface:

  1. Diagnostic reasoning
  2. Disease summary
  3. Similar local images
  4. External web results

---

## 🧠 AI Agent Orchestration

### ⟳ Flow:

1. **Input Handling**: Accepts medical image or text query
2. **Reasoning**: Calls `medgemma-finetuned-cxr-reasoning`
3. **Summarization**: Calls `medgemma-finetuned-cxr-summerizer`
4. **Retrieval**: Triggers MedCLIP and serpAPI tools
5. **Streaming**: Step-wise output to frontend
6. **Domain Control**: Filters non-medical prompts

---

## 🌟 Intended Use

This project is **strictly for research and academic purposes**. It is **not intended for clinical diagnosis or treatment** and has not been validated for real-world healthcare use.

---

## 🛠️ Technologies & Tools

### 💻 Core Frameworks

| Technology    | Purpose                                            |
| ------------- | -------------------------------------------------- |
| **Python**    | Core programming                                   |
| **FastAPI**   | Web backend                                        |
| **LangChain** | Tool-based orchestration                           |
| **LangGraph** | Agent state management                             |
| **AWS EC2**   | Agent hosting                                      |
| **SageMaker** | Model deployment (Reasoner + Summariser + MedCLIP) |
| **FAISS**     | Local image similarity search                      |

### 🧠 AI Models

| Model Name                            | Role                                                                                |
| ------------------------------------- | ----------------------------------------------------------------------------------- |
| **medgemma-finetuned-cxr-reasoning**  | Generates diagnostic reasoning from CXR images (trained and uploaded by the author) |
| **medgemma-finetuned-cxr-summerizer** | Converts reasoning into concise summaries (trained and uploaded by the author)      |
| **medCLIP**                           | Retrieves similar chest X-rays via vector search                                    |

### 📆 Dataset

| Dataset Name                  | Description                                                                                                                                                |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **cxr-10k-reasoning-dataset** | Custom dataset derived from MIMIC-CXR. Includes image paths, findings, impressions, and structured reasoning—created, cleaned, and uploaded by the author. |

---

## 📘 References

* [CXR Reasoning Dataset](https://huggingface.co/datasets/Manusinhh/cxr-10k-reasoning-dataset)
* [Fine-tuned MedGEMMA Reasoner](https://huggingface.co/Manusinhh/medgemma-finetuned-cxr-reasoning)
* [Fine-tuned MedGEMMA Summariser](https://huggingface.co/Manusinhh/medgemma-finetuned-cxr-summerizer)
* [MedGEMMA Base](https://huggingface.co/google/medgemma-4b-it)
* [Insight-V (GitHub)](https://github.com/JoshuaChou2018/Insight-V)
* [Insight-V (Paper)](https://arxiv.org/abs/2411.14432)
* [MIMIC-CXR Dataset](https://physionet.org/content/mimic-cxr/2.0.0/)
* [LLaVA-Med](https://github.com/microsoft/LLaVA-Med)
* [MedCLIP](https://github.com/UCSD-AI4H/MedCLIP)
* [LangChain](https://github.com/langchain-ai/langchain)
* [LangGraph](https://github.com/langchain-ai/langgraph)

---

## ⚠️ Disclaimer

This is a **research prototype only**. Do **not use for clinical or real-world medical decision-making**.
