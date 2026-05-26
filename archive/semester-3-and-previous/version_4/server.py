from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import sqlite3
import asyncio
from transformers import pipeline

# FastAPI instance
app = FastAPI()

# Database initialization
def init_db():
    conn = sqlite3.connect("candidates.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER,
            resume TEXT NOT NULL,
            FOREIGN KEY (job_id) REFERENCES jobs (id)
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Models for API
class JobAndResumes(BaseModel):
    job_description: str
    resumes: List[str]

class RetrainRequest(BaseModel):
    model_name: str

# получаем девайс 
import torch

# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Загружаем модель BERT
# import kaggle
# kaggle.api.authenticate()
# # TODO добавить условие на скачивание чтобы каждый раз не скачивать
# kaggle.api.model_instance_versions_download(
# owner_slug='nosovartiom'
# , model_slug='deeppavlovrubert-base-cased-sentence'
# , framework = 'transformers'
# , instance_slug='vacancy-resume-next-sentence-prediction'
# , version_number=1)

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

def load_bert_model():
    # Директория, где сохранена модель
    output_dir = "C:/Users/Artiom/.cache/kagglehub/models/nosovartiom/deeppavlovrubert-base-cased-sentence/transformers/vacancy-resume-next-sentence-prediction/1/model_save"

    # Загрузка модели и токенайзера
    model_load = AutoModelForSequenceClassification.from_pretrained(output_dir, )
    tokenizer_load = AutoTokenizer.from_pretrained(output_dir)

    # Загрузка состояния оптимизатора, планировщика и гиперпараметров 
    if device.type != 'cpu':
        checkpoint_load = torch.load(os.path.join(output_dir, "training_args.bin"))
    else:
        checkpoint_load = torch.load(os.path.join(output_dir, "training_args.bin")
                                    , map_location=torch.device('cpu'))

    # Реконструкция оптимизатора и планировщика
    hyperparameters_load = checkpoint_load['hyperparameters']
    learning_rate_load = hyperparameters_load['learning_rate']
    eps_load = hyperparameters_load['eps']
    optimizer_load = torch.optim.AdamW(model_load.parameters(), lr=learning_rate_load, eps=eps_load)
    optimizer_load.load_state_dict(checkpoint_load['optimizer_state_dict'])
    from transformers import get_linear_schedule_with_warmup
    epochs_load = hyperparameters_load['epochs']
    train_dataset_size = 100
    total_steps_load = train_dataset_size * epochs_load
    scheduler_load = get_linear_schedule_with_warmup(optimizer_load,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps_load)
    scheduler_load.load_state_dict(checkpoint_load['scheduler_state_dict'])

    # Загрузка гиперпараметров
    batch_size_load = hyperparameters_load['batch_size']

    print(f"Model, tokenizer, and training state loaded from {output_dir}")

    return model_load, tokenizer_load

bert_model, bert_tokenizer = load_bert_model()
import torch

def calculate_score_log_ratio(vacancy, resume, trained_model, tokenizer, device='cpu', debug=0, epsilon=1e-8):
    """
    Рассчитывает скоринг на основе логарифмического отношения вероятностей классов.
    """
    # Токенизация данных
    inputs = tokenizer(
        text=vacancy,
        text_pair=resume,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Переместить данные на устройство
    inputs = {key: value.to(device) for key, value in inputs.items()}
    trained_model = trained_model.to(device)
    
    # Получение предсказаний
    trained_model.eval()
    with torch.no_grad():
        outputs = trained_model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
    
    # Вероятности классов
    probability_class_0 = probabilities[0][0]
    probability_class_1 = probabilities[0][1]
    
    # Логарифмическое отношение вероятностей (с защитой от деления на 0)
    score = torch.log(probability_class_1 / (probability_class_0 + epsilon))
    
    if debug:
        print(f"Probabilities: {probabilities}")
        print(f"Probability of class 0: {probability_class_0:.4f}, Probability of class 1: {probability_class_1:.4f}")
        print(f"Score (log ratio): {score:.4f}")
    
    return score.item()

async def bert_predict(model, tokenizer, vacancy, resumes):
    result = []
    for resume in resumes:
        result.append({'vacancy': vacancy, 'resume': resume, 'score': calculate_score_log_ratio(vacancy=vacancy
                                                                                                , resume=resume
                                                                                                , trained_model=model
                                                                                                , tokenizer=tokenizer)})
    return result

# Async function for BERT model processing
async def process_with_model(job_description: str, resumes: List[str]):
    results = await bert_predict(bert_model, bert_tokenizer, job_description, resumes)
    return results

# API Endpoints
@app.post("/submit")
async def submit_job_and_resumes(data: JobAndResumes):
    conn = sqlite3.connect("candidates.db")
    cursor = conn.cursor()

    # Insert job description
    cursor.execute("INSERT INTO jobs (description) VALUES (?)", (data.job_description,))
    job_id = cursor.lastrowid

    # Insert resumes
    for resume in data.resumes:
        cursor.execute("INSERT INTO resumes (job_id, resume) VALUES (?, ?)", (job_id, resume))

    conn.commit()
    conn.close()

    # Process data with the BERT model asynchronously
    results = await process_with_model(data.job_description, data.resumes)
    result_of_response = []
    for i in range(len(results)):
        result_of_response += [{data.resumes[i]: results[i]}]
    return {"job_id": job_id, "results": str(result_of_response)}

@app.post("/retrain")
async def retrain_model(request: RetrainRequest):
    # Simulate retraining process
    await asyncio.sleep(2)  # Dummy retrain time
    return {"message": f"Model {request.model_name} retrained successfully"}

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# {'job_id': 2, 'results': [{'sequence': 'Experienced Python developer with a focus on web applications.', 'labels': ['Software Engineer
# with experience in Python and machine learning.'], 'scores': [0.6094723343849182]}, {'sequence': 'Machine learning expert with strong Python skil
# ls and deployment experience.', 'labels': ['Software Engineer with experience in Python and machine learning.'], 'scores': [0.33050838112831116]}
# , {'sequence': 'Data analyst with SQL and visualization expertise.', 'labels': ['Software Engineer with experience in Python and machine learning
# .'], 'scores': [0.0014800283825024962]}]}