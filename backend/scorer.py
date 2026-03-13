from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

JOB_DESCRIPTIONS = [
    {"id":"aws_mle_01","title":"Machine Learning Engineer","company":"Amazon Web Services","description":"Python SageMaker XGBoost TensorFlow PyTorch MLOps Feature Store SageMaker Pipelines Docker Kubernetes distributed training A/B testing scikit-learn Spark SQL CI/CD experiment tracking model versioning REST APIs AWS statistics linear algebra LLMs model monitoring"},
    {"id":"aws_mle_02","title":"Senior ML Engineer","company":"Amazon","description":"collaborative filtering deep learning Python SageMaker real-time inference feature engineering A/B testing Spark Kafka Docker AWS Lambda Redis SQL scaling ML model monitoring"},
    {"id":"meta_mle_01","title":"ML Engineer - Feed Ranking","company":"Meta","description":"Python PyTorch distributed training XGBoost LightGBM embeddings model compression Spark Hive causal inference multi-task learning transformers experiment tracking"},
    {"id":"google_mle_01","title":"ML Engineer Google Cloud","company":"Google","description":"Python TensorFlow Vertex AI Kubernetes distributed systems model serving TFX pipelines feature store data validation model monitoring Docker BigQuery MLOps quantization"},
    {"id":"openai_mle_01","title":"ML Engineer Training Infra","company":"OpenAI","description":"Python PyTorch distributed training GPU profiling memory optimization Kubernetes Docker deep learning transformers C++ systems programming"},
    {"id":"netflix_mle_01","title":"Senior ML Engineer","company":"Netflix","description":"Python Scala Spark TensorFlow PyTorch collaborative filtering embeddings causal inference A/B testing Kafka feature store model monitoring AWS SageMaker Docker Kubernetes"},
    {"id":"stripe_mle_01","title":"ML Engineer Fraud Detection","company":"Stripe","description":"Python XGBoost gradient boosting anomaly detection real-time inference feature engineering SQL Spark Kafka SHAP Docker Kubernetes AWS model monitoring data drift"},
    {"id":"uber_mle_01","title":"ML Engineer Maps ETA","company":"Uber","description":"Python TensorFlow PyTorch XGBoost feature engineering real-time ML Kafka Spark distributed systems model serving A/B testing causal inference geospatial"},
    {"id":"databricks_mle_01","title":"ML Platform Engineer","company":"Databricks","description":"Python Spark MLflow distributed computing model registry experiment tracking REST APIs Kubernetes Docker Delta Lake feature store model serving MLOps CI/CD"},
    {"id":"microsoft_mle_01","title":"Senior ML Engineer Azure","company":"Microsoft","description":"Python Azure ML PyTorch TensorFlow distributed training MLOps model deployment Kubernetes Docker AutoML model fairness SHAP LIME interpretability"},
    {"id":"linkedin_mle_01","title":"ML Engineer LinkedIn Feed","company":"LinkedIn","description":"Python Scala Spark TensorFlow PyTorch XGBoost deep learning NLP BERT embeddings feature store Kafka A/B testing causal inference SQL system design"},
    {"id":"airbnb_mle_01","title":"ML Engineer Trust Safety","company":"Airbnb","description":"Python XGBoost neural networks anomaly detection NLP feature engineering Spark Airflow Kubernetes Docker AWS model monitoring explainability A/B testing"},
    {"id":"doordash_mle_01","title":"ML Engineer Delivery Prediction","company":"DoorDash","description":"Python XGBoost gradient boosting Spark Kafka real-time inference Kubernetes Docker AWS SageMaker MLflow time series A/B testing causal inference SQL"},
    {"id":"palantir_mle_01","title":"ML Deployment Engineer","company":"Palantir","description":"Python Java Spark Docker Kubernetes MLOps model serving feature pipelines REST APIs model monitoring anomaly detection NLP computer vision"},
    {"id":"snowflake_mle_01","title":"ML Engineer Snowpark","company":"Snowflake","description":"Python distributed computing feature engineering scikit-learn XGBoost Kubernetes Docker SQL REST APIs MLOps model registry experiment tracking"},
    {"id":"salesforce_mle_01","title":"ML Engineer Einstein AI","company":"Salesforce","description":"Python NLP BERT transformers large language models fine-tuning text classification PyTorch TensorFlow MLflow Docker Kubernetes REST APIs model monitoring"},
    {"id":"twitter_mle_01","title":"ML Engineer Ads","company":"X Twitter","description":"Python Scala Spark TensorFlow deep learning embeddings feature engineering real-time ML Kafka A/B testing distributed systems"},
    {"id":"lyft_mle_01","title":"Senior ML Engineer Pricing","company":"Lyft","description":"Python XGBoost neural networks time series causal inference Spark Kafka SageMaker MLflow Docker Kubernetes reinforcement learning A/B testing"},
    {"id":"apple_mle_01","title":"ML Engineer Siri","company":"Apple","description":"Python PyTorch TensorFlow model compression quantization pruning NLP speech recognition computer vision federated learning privacy ML C++ mobile optimization"},
    {"id":"twosigma_mle_01","title":"ML Engineer Quant Research","company":"Two Sigma","description":"Python machine learning time series statistical modeling deep learning PyTorch feature engineering Spark SQL C++ mathematics statistics linear algebra optimization"}
]

SKILL_GROUPS = {
    "Core ML": ["machine learning","deep learning","neural networks","xgboost","gradient boosting","lightgbm","random forest","scikit-learn","sklearn","supervised learning","reinforcement learning","transfer learning"],
    "MLOps": ["mlops","sagemaker pipelines","mlflow","kubeflow","airflow","ci/cd","model monitoring","model registry","experiment tracking","feature store","sagemaker feature store","data drift","a/b testing"],
    "AWS / Cloud": ["sagemaker","aws","s3","lambda","ec2","ecs","eks","dynamodb","cloudwatch","azure","gcp","vertex ai"],
    "Frameworks": ["pytorch","tensorflow","keras","huggingface","transformers","bert","llm","large language models","fine-tuning"],
    "Data Engineering": ["spark","kafka","flink","sql","bigquery","redshift","hive","delta lake","dbt","pyspark"],
    "Deployment": ["docker","kubernetes","fastapi","flask","rest api","model serving","sagemaker endpoints","real-time inference","batch inference"],
    "NLP": ["nlp","natural language processing","text classification","embeddings","information extraction"],
    "Programming": ["python","scala","java","c++","golang"]
}

HIGH_VALUE_SKILLS = {
    "sagemaker pipelines":38,"sagemaker feature store":35,"mlops":30,"model monitoring":28,
    "feature store":25,"kubeflow":22,"mlflow":20,"xgboost":18,"pytorch":18,"tensorflow":18,
    "kubernetes":15,"docker":12,"spark":12,"fastapi":10,"causal inference":15,
    "distributed training":20,"llm":18,"transformers":18,"a/b testing":12
}

def extract_skills(text):
    lower = text.lower()
    return {s for skills in SKILL_GROUPS.values() for s in skills if s in lower}

def compute_score(resume_text, job_description):
    vectorizer = TfidfVectorizer(ngram_range=(1,3), stop_words="english", max_features=5000, sublinear_tf=True)
    tfidf = vectorizer.fit_transform([resume_text.lower(), job_description.lower()])
    base = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)
    matched = resume_skills & job_skills
    missing = job_skills - resume_skills
    bonus = sum(HIGH_VALUE_SKILLS.get(s,0)*0.003 for s in matched)
    score = round(min(0.97, max(0.08, (base+bonus)*1.8))*100, 1)
    missing_impacts = sorted([{"skill":s,"impact":HIGH_VALUE_SKILLS[s]} for s in missing if s in HIGH_VALUE_SKILLS], key=lambda x:x["impact"], reverse=True)[:6]
    matched_by_group = {g:[s for s in sl if s in matched] for g,sl in SKILL_GROUPS.items() if any(s in matched for s in sl)}
    return {"score":score,"matched_skills":sorted(matched),"missing_skills":missing_impacts,"matched_by_group":matched_by_group,"skill_coverage":round(len(matched)/max(len(job_skills),1)*100,1)}

def get_top_recommendations(resume_text):
    results = sorted([{**{"id":j["id"],"title":j["title"],"company":j["company"]},**compute_score(resume_text,j["description"])} for j in JOB_DESCRIPTIONS], key=lambda x:x["score"], reverse=True)
    freq = {}
    for job in results[:5]:
        for item in job["missing_skills"]: freq[item["skill"]] = freq.get(item["skill"],0)+item["impact"]
    top_missing = sorted(freq.items(), key=lambda x:x[1], reverse=True)[:5]
    skills = extract_skills(resume_text)
    return {"top_job":results[0],"all_jobs":results,"top_missing_skills":[{"skill":s,"impact":v} for s,v in top_missing],"total_skills_detected":len(skills),"skills_detected":sorted(skills)}
