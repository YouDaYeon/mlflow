import mlflow
import openai
import os

# MLflow 서버 실행
# uvx mlflow server --host 127.0.0.1 --port 5000

# 1. MLflow 서버 연결 설정
mlflow.set_tracking_uri("http://localhost:5000")

# 2. OpenAI 자동 기록 활성화
# (pip install openai)
mlflow.openai.autolog()

# 3. OpenAI 클라이언트 생성 (API 키가 환경변수에 있다고 가정)
client = openai.OpenAI(
    base_url="http://10.100.0.21:9978/v1",
    api_key="not-needed" # 로컬 실행 시 인증키는 필요 없으나 형식상 입력
)

# 4. MLflow 실험(Experiment) 이름 설정 (선택 사항)
mlflow.set_experiment("OpenAI_Test_Project")

print("🚀 OpenAI에 질문을 던지는 중입니다...")

# 5. 실제 OpenAI API 호출
# 이 호출이 발생하는 순간 MLflow가 자동으로 데이터를 가로채서 기록합니다.
response = client.chat.completions.create(
    model="gpt-3.5-turbo", # 혹은 사용 가능한 모델명
    messages=[
        {"role": "system", "content": "너는 친절한 AI 조수야."},
        {"role": "user", "content": "MLflow와 OpenAI를 같이 쓰면 뭐가 좋아?"}
    ],
    temperature=0.7
)

print("-" * 30)
print(f"답변 내용: {response.choices[0].message.content}")
print("-" * 30)
print("✅ 기록 완료! http://localhost:5000 에 접속해서 확인해 보세요.")