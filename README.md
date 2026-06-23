# Video Link : https://drive.google.com/file/d/1hcIByDnekEpdY0FXvYo1gDLc8AT0oybz/view?usp=sharing
# 🚦 AI Traffic Operations Copilot

> **Retrieval-Augmented Decision Support System for Traffic Operations Management**

AI Traffic Operations Copilot is an intelligent traffic management assistant that combines **Large Language Models (LLMs)**, **semantic retrieval**, **historical traffic intelligence**, and **deterministic analytics** to generate explainable traffic management recommendations.

The system transforms natural-language descriptions of traffic incidents, public events, construction activities, and operational disruptions into structured insights, risk assessments, resource recommendations, and actionable traffic management plans.

---

# 📌 Problem Statement

Traffic operators often rely on fragmented historical records, manual analysis, and domain expertise to make operational decisions.

### Challenges

* Lack of centralized decision support
* Difficulty finding relevant historical incidents
* Limited explainability in AI-driven systems
* Delayed operational planning during disruptions
* Inconsistent resource allocation decisions

This project addresses these challenges by transforming historical traffic records into **actionable operational intelligence**.

---

# ✨ Key Features

## 🧠 LLM-Powered Event Understanding

Converts natural language descriptions into structured event information:

* Event Type
* Event Cause
* Corridor
* Priority
* Road Closure Requirement
* Event Description

---

## 📚 Historical Event Retrieval

Uses:

* Sentence Transformers (MiniLM)
* FAISS Vector Search
* Metadata-Aware Reranking

to retrieve the most relevant historical incidents.

---

## 📍 Location-Aware Retrieval

Enhances retrieval relevance using:

* Latitude
* Longitude
* Haversine Distance

Nearby historical incidents receive additional ranking weight.

---

## ⚠️ Explainable Risk Assessment

Risk scores are generated using deterministic analytics based on:

* Historical closure rates
* Event priority
* Event cause burden
* Handling duration
* Operational urgency
* Evidence uncertainty

---

## 👮 Resource Planning

Generates recommendations for:

* Personnel deployment
* Barricades
* Support vehicles

using transparent policy-based rules.

---

## 📝 Traffic Management Plan Generation

Produces a structured operational report including:

* Event Summary
* Historical Evidence
* Risk Assessment
* Resource Recommendations
* Monitoring Checklist
* Operational Limitations

---

## 🗺 Interactive Dashboard

Built using Streamlit with:

* Similar Event Visualization
* Historical Event Map
* Risk Metrics
* Generated Reports

---

# 🏗 System Architecture

```text
┌────────────────────┐
│    Streamlit UI    │
└──────────┬─────────┘
           │
           ▼
┌────────────────────┐
│     LangGraph      │
│ Workflow Engine    │
└──────────┬─────────┘
           │
           ▼
┌────────────────────┐
│ Event Understanding│
│     OpenAI GPT     │
└──────────┬─────────┘
           │
           ▼
┌────────────────────┐
│ Historical Retrieval│
│       FAISS         │
└──────────┬─────────┘
           │
           ▼
┌────────────────────┐
│ SQLite Event Store │
└──────────┬─────────┘
           │
           ▼
┌────────────────────┐
│ Impact Assessment  │
└──────────┬─────────┘
           │
           ▼
┌────────────────────┐
│ Resource Planning  │
└──────────┬─────────┘
           │
           ▼
┌────────────────────┐
│ Report Generation  │
└────────────────────┘
```

---

# 🔄 LangGraph Workflow

```text
1. Event Understanding
          ↓
2. Historical Retrieval
          ↓
3. Impact Assessment
          ↓
4. Resource Planning
          ↓
5. Report Generation
          ↓
6. Save Case
```

---

# 📍 Location-Aware Retrieval

Final retrieval ranking combines semantic similarity, metadata matching, and geographic relevance.

## Retrieval Formula

```text
Final Score =
0.50 × Semantic Similarity
+ 0.15 × Event Cause Match
+ 0.10 × Corridor Match
+ 0.05 × Event Type Match
+ 0.05 × Priority Match
+ 0.15 × Location Score
```

### Distance-Based Location Score

| Distance | Score |
| -------- | ----- |
| ≤ 1 km   | 1.0   |
| ≤ 5 km   | 0.8   |
| ≤ 10 km  | 0.5   |
| ≤ 20 km  | 0.2   |
| > 20 km  | 0.0   |

---

# 📊 Technology Stack

| Layer           | Technology       |
| --------------- | ---------------- |
| Frontend        | Streamlit        |
| Workflow Engine | LangGraph        |
| LLM             | OpenAI GPT       |
| Embeddings      | all-MiniLM-L6-v2 |
| Vector Search   | FAISS            |
| Analytics       | Pandas, NumPy    |
| Database        | SQLite           |
| Mapping         | Folium           |
| Deployment      | Docker           |

---

# 📈 Dataset Information

| Metric                 | Value            |
| ---------------------- | ---------------- |
| Historical Records     | ~8,000+          |
| Database               | SQLite           |
| Vector Dimension       | 384              |
| Embedding Model        | all-MiniLM-L6-v2 |
| Candidate Retrieval    | Top 30           |
| Final Retrieved Events | Top 5            |

---

# 🚀 Running Locally

## Clone Repository

```bash
git clone https://github.com/yourusername/traffic-copilot.git

cd traffic-copilot
```

## Create Virtual Environment

```bash
python -m venv venv
```

### Linux / Mac

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Configure Environment

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

TRAFFIC_DATABASE_PATH=data/traffic_ops.db

FAISS_INDEX_PATH=data/events.faiss

FAISS_MAPPING_PATH=data/faiss_event_ids.json
```

## Run Application

```bash
streamlit run app.py
```

---

# 🐳 Docker Deployment

## Build Image

```bash
docker build -t traffic-copilot .
```

## Run Container

```bash
docker run -p 8501:8501 traffic-copilot
```

Open:

```text
http://localhost:8501
```

---

# 📋 Example Use Case

### Input

```text
Heavy rainfall has caused waterlogging near Silk Board Junction resulting in slow traffic movement.
```

### Retrieved Historical Evidence

```text
Event ID: FKID002770
Distance: 0.14 km
Cause: Water Logging
Priority: High
```

### Output

```text
Risk Score: 31.8
Risk Band: Moderate

Personnel: 4–6
Barricades: 4–8

Generated Traffic Management Plan
```

---

# 🔍 Non-Functional Requirements

## Scalability

* FAISS supports large-scale vector retrieval
* SQLite can be migrated to PostgreSQL

## Reliability

* Deterministic risk scoring
* Input validation checks

## Explainability

* Historical evidence displayed
* Similarity scores exposed
* Risk reasoning provided

## Maintainability

* Modular architecture
* Service-based design
* LangGraph workflow separation

## Portability

* Dockerized deployment
* Environment-based configuration

---

# 🔮 Future Enhancements

* Real-time traffic feeds
* Google Maps integration
* PostgreSQL migration
* Managed vector databases
* Route optimization
* Traffic congestion forecasting
* Operator feedback loop
* Multi-city deployment

---

# 👨‍💻 Team

**AI Traffic Operations Copilot**

Built as a Retrieval-Augmented Decision Support System for intelligent and explainable traffic operations management.

---

