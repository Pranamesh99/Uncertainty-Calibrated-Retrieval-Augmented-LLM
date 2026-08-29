# Uncertainty-Calibrated Retrieval-Augmented LLM with Evidence Verification (UC-RAG)

## Overview
This repository contains a completely custom **Sequence-to-Sequence (Seq2Seq) Transformer LLM** built from scratch using PyTorch. Instead of relying on traditional RAG, this system implements the advanced **UC-RAG (Uncertainty-Calibrated RAG)** architecture to mathematically verify claims and prevent AI hallucinations.

## Key Features & Novelties
1. **Query-Aware Hybrid Retrieval**: Calculates a retrieval score combining Semantic, Keyword, Authority, and Recency weights (`Score(D) = α*S_semantic + β*S_keyword + γ*S_authority + δ*S_recency`).
2. **Evidence Quality Estimation**: Mathematically evaluates retrieved documents using `E = (Relevance + Authority + Consistency) / 3`.
3. **Claim-Level Evidence Verification**: The model breaks its generated answer into individual claims (sentences) and independently verifies each claim against the retrieved evidence, flagging them as *Supported*, *Partially Supported*, or *Unsupported*.
4. **Uncertainty Calibration**: Calculates a rigorous confidence metric using `U = 1 - (αR + βE + γV + δC)`.
5. **Adaptive Response**: Dynamically alters system behavior based on confidence thresholds (e.g., providing a direct answer for high confidence, or safely stating "Evidence is insufficient" for low confidence).

## Project Structure
* `server.py`: The core application containing the PyTorch Neural Network, the RAG Pipeline, and the Flask Web UI.
* `run_tests.py`: The automated evaluation suite that tests 20,000 deterministic variations and calculates the **Expected Calibration Error (ECE)**.
* `training_dataset.txt`: The knowledge database the LLM trains on and retrieves from.
* `templates/index.html`: The frontend Academic Light-Theme interface that displays claim-level verifications.

## How to Run Locally

### 1. Install Dependencies
```bash
pip install torch flask
```

### 2. Start the Web Server
```bash
python server.py
```
*(Wait ~45 seconds for the custom LLM to train locally, then open `http://127.0.0.1:8505`)*

### 3. Run the Automated Evaluation Suite
```bash
python run_tests.py
```
*(Runs 20,000 deterministic test cases and outputs the Expected Calibration Error)*

