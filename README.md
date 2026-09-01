# 🌱 ROOTS & RISE

## AI-Powered Citizen Grievance Intelligence Platform

> **From Citizen Voices to Smarter Governance**

ROOTS & RISE is an AI-powered citizen grievance management platform designed to transform unstructured citizen complaints into structured and actionable governance intelligence.

The platform uses **Natural Language Processing (NLP)** and **semantic similarity** to automatically classify grievances, assign priority, route complaints to relevant departments, detect duplicate complaints, and identify geographical hotspots.

---

## 🚀 Key Features

### 🧠 AI-Powered Complaint Classification
Uses a Sentence Transformer model to understand the semantic meaning of citizen complaints and classify them into relevant categories.

### 🎯 Smart Department Routing
Automatically routes grievances to the appropriate government department based on AI classification.

### ⚡ Priority Intelligence
Identifies high-priority and urgent grievances requiring immediate attention.

### 🔁 Semantic Duplicate Detection
Detects complaints with similar meanings even when they use different wording.

### 🔎 Complaint Tracking
Citizens can track their grievance using a unique Complaint ID.

### 🏛️ Authority Intelligence Dashboard
Provides authorities with real-time insights and analytics about citizen grievances.

### 📍 Geographic Hotspot Detection
Uses geographical coordinates and clustering techniques to identify areas with a high concentration of complaints.

---

## 🤖 AI Technologies Used

- **Sentence Transformers**
- **all-MiniLM-L6-v2**
- **Semantic Embeddings**
- **Cosine Similarity**
- **Natural Language Processing**

---

## 🛠️ Technology Stack

### Frontend
- Gradio

### Backend
- Python

### Database
- SQLite

### AI & Machine Learning
- Sentence Transformers
- Scikit-learn
- PyTorch

### Data Visualization
- Plotly
- Pandas

---

## 🏗️ System Architecture

```text
Citizen Complaint
       │
       ▼
Language Normalization
       │
       ▼
AI Semantic Embedding Model
       │
 ┌─────┼──────────┐
 ▼     ▼          ▼
Classification  Duplicate  Priority
               Detection  Analysis
       │
       ▼
Department Routing
       │
       ▼
SQLite Database
       │
       ▼
Authority Dashboard
       │
       ▼
Analytics & Geographic Hotspots
