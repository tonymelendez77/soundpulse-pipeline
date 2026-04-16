# SoundPulse

Data pipeline that correlates world news sentiment with music chart trends. Ingests from 6 APIs, classifies emotions with DistilRoBERTa, clusters audio with KMeans, and predicts next week's chart mood with XGBoost.

## Tech Stack

- **Language:** Python 3.11
- **Cloud:** GCP (BigQuery, Cloud Storage)
- **Transform:** dbt-bigquery
- **NLP:** Hugging Face Transformers
- **ML:** XGBoost, scikit-learn, SHAP
- **Audio:** Librosa, MusicGen
- **Vector DB:** Pinecone
- **Orchestration:** Prefect + GitHub Actions
- **Backend:** FastAPI
- **Frontend:** Streamlit, Plotly.js, GitHub Pages

## Setup

```bash
git clone https://github.com/tonymelendez77/soundpulse-pipeline.git
cd soundpulse-pipeline
pip install -r requirements_pipeline.txt
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
```

## Run

```bash
# Full pipeline
python orchestration/prefect_pipeline.py

# Individual modules
python ingestion/news_ingestion.py
python ingestion/ml_predictions.py

# API + dashboards
uvicorn serving.api:app --reload --port 8000
streamlit run serving/dashboard_mood.py --server.port 8501
streamlit run serving/dashboard_trends.py --server.port 8502

# Static site
python serving/export_static.py
python -m http.server 8889 --directory docs/
```

## Author

Oscar J. Melendez
