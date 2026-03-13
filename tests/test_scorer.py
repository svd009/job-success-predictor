from backend.scorer import extract_skills, compute_score

def test_extract_skills():
    text = "Python XGBoost SageMaker Docker Kubernetes"
    skills = extract_skills(text)
    assert "python" in skills
    assert "xgboost" in skills

def test_compute_score():
    resume = "Python XGBoost SageMaker Docker Kubernetes MLOps"
    jd = "Python XGBoost SageMaker MLOps model monitoring"
    result = compute_score(resume, jd)
    assert result["score"] > 0