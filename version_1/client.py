import requests

# Define the base URL of the server
BASE_URL = "http://127.0.0.1:8000"

# Test data for the /submit endpoint
job_description = "Software Engineer with experience in Python and machine learning."
resumes = [
    "Experienced Python developer with a focus on web applications.",
    "Machine learning expert with strong Python skills and deployment experience.",
    "Data analyst with SQL and visualization expertise."
]

# Test /submit endpoint
def test_submit():
    print("Testing /submit endpoint...")
    payload = {
        "job_description": job_description,
        "resumes": resumes
    }
    response = requests.post(f"{BASE_URL}/submit", json=payload)
    if response.status_code == 200:
        print("/submit endpoint works correctly.")
        print("Response:", response.json())
    else:
        print("Error with /submit endpoint:", response.status_code, response.text)

# Test data for the /retrain endpoint
retrain_payload = {
    "model_name": "bert_model"
}

# Test /retrain endpoint
def test_retrain():
    print("Testing /retrain endpoint...")
    response = requests.post(f"{BASE_URL}/retrain", json=retrain_payload)
    if response.status_code == 200:
        print("/retrain endpoint works correctly.")
        print("Response:", response.json())
    else:
        print("Error with /retrain endpoint:", response.status_code, response.text)

if __name__ == "__main__":
    test_submit()
    test_retrain()
