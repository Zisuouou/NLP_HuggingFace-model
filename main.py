from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
import torch

# FastAPI 인스턴스 생성
app = FastAPI()

# NER 모델 및 토크나이저 로드
NER_MODEL_NAME = "xlm-roberta-large-finetuned-conll03-english"
ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)

# Sentiment Analysis 모델 및 토크나이저 로드
SENTIMENT_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)

# 요청 데이터 형식 정의
class AnalyzeRequest(BaseModel):
    text: str

# NER 함수
def perform_ner(text):
    """
    텍스트에서 Named Entity를 추출합니다.
    """
    tokens = ner_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = ner_model(**tokens)
    predictions = torch.argmax(outputs.logits, dim=-1)
    labels = predictions[0].tolist()
    tokens = ner_tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
    label_map = ner_model.config.id2label

    results = []
    for token, label_id in zip(tokens, labels):
        label = label_map[label_id]
        if label != "O":  # "O"는 엔터티가 아님
            results.append({"entity": token, "type": label})
    return results

# Sentiment Analysis 함수
def perform_sentiment_analysis(text):
    """
    텍스트에서 감정 분석을 수행하고 상위 3개의 감정을 반환합니다.
    """
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = sentiment_model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    topk = torch.topk(probabilities, k=3, dim=-1)
    top_indices = topk.indices[0].tolist()
    top_scores = topk.values[0].tolist()

    labels = sentiment_model.config.id2label
    top_labels = [labels[idx] for idx in top_indices]

    return [{"label": top_labels[i], "score": top_scores[i]} for i in range(3)]

# API 엔드포인트 정의
@app.post("/analyze")
async def analyze_text(request: AnalyzeRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="The 'text' field cannot be empty.")

    try:
        # Named Entity Recognition 수행
        entities = perform_ner(request.text)

        # Sentiment Analysis 수행
        sentiments = perform_sentiment_analysis(request.text)

        # 결과 반환
        return {
            "entities": entities,
            "emotions": [sentiment["label"] for sentiment in sentiments],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")