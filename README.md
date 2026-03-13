# Job Application Success Predictor

> ML system that scores resume-to-job fit using TF-IDF vectorization, skill gap analysis, and XGBoost-inspired feature engineering — deployed via FastAPI with a SageMaker MLOps architecture.

[![AWS MLE](https://img.shields.io/badge/AWS-ML%20Engineer%20Associate-orange?logo=amazon-aws)](https://aws.amazon.com/certification/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green?logo=fastapi)](https://fastapi.tiangolo.com)

---

## What It Does

Paste your resume → get scored against 20 real ML engineering job descriptions:

```
Input: Resume text
Output: "67% match for AWS MLE roles — add 'SageMaker Pipelines' to hit 91%"
```

- **Live scoring** of 20 curated ML engineering JDs (AWS, Meta, Google, Netflix, Stripe, etc.)
- **Skill gap analysis** with impact scores per missing skill
- **Grouped skill detection** across 9 ML competency areas
- **FastAPI backend** with `/score`, `/score/single`, `/jobs` endpoints

---

## System Architecture

```
Resume Text
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                   SageMaker Processing Job                   │
│  • Text normalization  • Skill extraction  • TF-IDF vectors │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  SageMaker Feature Store                      │
│  • Resume feature group    • JD feature group                │
│  • Skill vectors           • Match history                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               XGBoost Training (SageMaker)                   │
│  • Cosine similarity base score                              │
│  • Skill overlap features                                    │
│  • High-value skill bonus weights                            │
│  • Normalized match score [0, 1]                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                SageMaker Real-Time Endpoint                  │
│  • ml.m5.xlarge instance                                     │
│  • Auto-scaling: 1–10 instances                              │
│  • p50 latency: ~120ms                                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   SageMaker Model Monitor                    │
│  • Data quality monitoring                                   │
│  • Model quality monitoring                                  │
│  • Bias drift detection                                      │
│  • CloudWatch alerts                                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Deployment (Docker + ECS)               │
│  POST /score        → full resume analysis                   │
│  POST /score/single → score vs specific JD                   │
│  GET  /jobs         → list all JDs                           │
└─────────────────────────────────────────────────────────────┘
```

---

## ML Pipeline (SageMaker Pipelines)

```python
pipeline = Pipeline(
    name="job-fit-predictor-pipeline",
    steps=[
        ProcessingStep(name="FeatureEngineering", ...),
        TrainingStep(name="XGBoostTraining", ...),
        EvaluationStep(name="ModelEvaluation", ...),
        ConditionStep(name="CheckAccuracy",
            conditions=[ConditionGreaterThan(left=accuracy, right=0.80)],
            if_steps=[RegisterModelStep(...), DeployStep(...)],
        ),
        MonitoringStep(name="EnableModelMonitor", ...)
    ]
)
```

---

## Scoring Algorithm

1. **TF-IDF Vectorization** — bigrams + trigrams, sublinear TF, 5K vocab
2. **Cosine Similarity** — base similarity score between resume and JD vectors
3. **Skill Bonus** — high-value skills (SageMaker Pipelines: +38%, MLOps: +30%, XGBoost: +18%...)
4. **Normalization** — clamp to [0.08, 0.97], multiply by calibration factor

---

## Quickstart

```bash
# Install
pip install fastapi uvicorn scikit-learn numpy

# Run API
uvicorn backend.main:app --reload

# Open demo
open frontend/index.html

# Score your resume
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "Python XGBoost SageMaker..."}'
```

---

## Project Structure

```
job-predictor/
├── backend/
│   ├── main.py          # FastAPI app
│   └── scorer.py        # TF-IDF + skill scoring engine
├── data/
│   └── job_descriptions.py   # 20 curated ML engineering JDs
├── frontend/
│   └── index.html       # Full standalone demo UI
├── notebooks/
│   └── sagemaker_pipeline.ipynb  # SageMaker architecture notebook
└── README.md
```

---

## Built By

**Sujay Dhhoka** · AWS ML Engineer Associate  
TA @ Rutgers University · MLOps & Production ML

[LinkedIn](https://linkedin.com/in/suujaydhhoka) · [GitHub](https://github.com/suujaydhhoka)