# AI 기반 검색 엔진

## 프로젝트 개요
이 프로젝트는 Flask 기반의 웹 애플리케이션으로, OpenAI와 Gemini AI 모델을 활용한 AI 검색 엔진입니다. 사용자는 검색어를 입력하면, Google 및 Naver의 검색 결과를 실시간으로 받아와 AI 모델을 통해 종합적인 답변을 제공합니다. 자연어 처리, BM25, FAISS, Sentence Transformer를 결합하여 검색 결과를 분석하고, 클릭 데이터를 기반으로 결과를 최적화합니다.

## 주요 기능
- **다양한 검색 소스 통합**: Google과 Naver의 텍스트, 이미지, 비디오 검색 결과를 가져옵니다.
- **AI 모델 기반 답변 생성**: ChatGPT(OpenAI)와 Gemini AI 모델을 선택하여, 검색 결과를 기반으로 종합적이고 인사이트를 제공하는 답변을 생성합니다.
- **시맨틱 검색**: BM25 알고리즘과 Sentence Transformer 모델을 결합하여 검색 결과의 관련성을 평가합니다.
- **클릭 추적 기능**: 사용자가 클릭한 검색 결과를 추적하여, 자주 클릭된 링크의 중요도를 높입니다.
- **Google 트렌드 표시**: 실시간으로 한국에서 인기 있는 검색어를 보여줍니다.
- **한국어 지원**: 한국어 검색 결과와 불용어 처리를 위한 `konlpy`와 `Okt` 사용.

## 요구 사항
- Python 3.8+
- Flask 2.0+
- Google 및 Naver API 키
- OpenAI 및 Gemini API 키

### 필요한 라이브러리
- `flask`
- `aiohttp`
- `bs4` (BeautifulSoup)
- `sqlite3`
- `requests`
- `konlpy` (한국어 NLP용 Okt)
- `sentence_transformers`
- `nltk`
- `rank_bm25`
- `faiss`
- `pytrends`
- `readability`
- `googleapiclient`
- `dotenv`

## 설치 방법

1. 리포지토리 클론:
   ```bash
   git clone https://github.com/your-repo/ai-powered-search.git
   cd ai-powered-search
2. 필요한 Python 패키지 설치:
   ```bash
   pip install -r requirements.txt

## 환경 설정

1. 프로젝트 루트에 .env 파일을 생성하고 다음 환경 변수 설정:
   ```bash
   OPENAI_API_KEY=your_openai_api_key
   GOOGLE_API_KEY=your_google_api_key
   CSE_ID2=your_google_cse_id
   GEMINI_API_KEY=your_gemini_api_key
   NAVER_CLIENT_ID=your_naver_client_id
   NAVER_CLIENT_SECRET=your_naver_client_secret

## 사용 방법

1. Flask 개발 서버 시작:
   ```bash
   python app.py
2. 웹 브라우저를 열고 http://127.0.0.1:5000에 접속
3. 검색어를 입력하고 응답 모델로 ChatGPT 또는 Gemini를 선택한 후 "검색" 버튼 클릭
4. 검색 결과와 함께 AI 생성 응답 확인

## 트렌드 가져오기
   ```bash
   홈페이지에서 Google 한국의 실시간 트렌드 표시

## 클릭 추적
   ```bash
   사용자가 검색 결과를 클릭하면 클릭 수를 추적하고, 이를 기반으로 미래 검색에서 클릭된 링크의 우선순위를 조정

## 주요 함수
- fetch_all_search_results: Google과 Naver에서 텍스트, 이미지, 비디오 검색 결과를 가져옵니다.
- fetch_content_async: BeautifulSoup와 readability를 사용하여 URL에서 콘텐츠를 추출합니다.
- bm25_search_with_sentence_transformer: BM25와 문장 임베딩을 사용하여 검색 결과를 순위 매깁니다.
- semantic_search_with_click_data: 클릭 기반 관련성을 적용하여 검색 결과를 결합합니다.
- record_click: 데이터베이스에서 URL 클릭 수를 업데이트합니다.
