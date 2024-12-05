NLP Coding Test [프로젝트 설명]
이 프로젝트는 NLP(자연어 처리) 관련 코딩 테스트를 위해 설계된 것으로, 이와 같은 주요 작업을 수행했습니다.
1. Named Entity Recognition (NER):
  - 입력 테스트에서 인물, 장소, 조직 등과 같은 엔터티(Entity)를 인식하고, 해당 유형을 반환합니다.
  - Hugging Face의 사전 학습된 NER 모델 (xlm-roberta-large-finetuned-conll03-english) 을 사용하였습니다.
2. Sentiment Analysis (감정 분석):
  - 입력 텍스트의 감정을 분석하여 상위 3개의 감정과 그 점수를 반환합니다.
  - Hugging Face의 사전 학습된 감정 분석 모델(j-hartmann/emotion-english-distilroberta-base) 을 사용하였습니다.
3. API Development:
  - FastAPI를 사용하여 텍스트를 입력받아 NER 및 감정 분석 결과를 반환하는 RESTful API를 개발하였습니다.
  - /analyze 엔드포인트를 통해 입력된 텍스트를 분석하고, JSON 형식으로 결과를 반환합니다.

NLP Coding Test [구현된 기능]
- Tasks 1: Named Entity Recognition (NER)
  - perform_ner 함수: 텍스트에서 엔터티와 유형을 추출하여 리스트 형태로 반환.
  - display_entities 함수: NER 결과를 사람이 읽기 쉽게 출력.
- Tasks 2: Sentiment Analysis
  - perform_sentiment_analysis 함수: 입력 텍스트의 감정을 분석하고 상위 3개의 감정과 점수를 반환.
- Tasks 3: API Development
  - FastAPI를 사용한 REST API 구현: POST /analyze
    - 요청: JSON 형식의 텍스트 입력. 응답: NER 및 감정 분석 결과를 JSON으로 반환.
  - 400 및 500 오류 처리를 포함.
 
NLP Coding Test [실행 및 테스트 방법]
1. 의존성 설치: 프로젝트 실행에 필요한 라이브러리를 설치
   - pip install fastapi uvicorn transformers torch
2. FastAPI 서버 실행: FastAPI를 실행하여 서버를 시작
   - uvicorn main:app --reload
3. API 테스트
- 브라우저에서 Swagger UI를 열어 테스트:
  - http://127.0.0.1:8000/docs
 
NLP Coding Test [파일 구성]
- main.py: FastAPI 서버 및 주요 기능 구현 파일.
- README.md: 프로젝트 설명 및 실행 가이드.
