# Start Tracing
# https://mlflow.org/docs/latest/genai/tracing/quickstart/

# pip install --upgrade 'mlflow[genai]'

# Step 1: Start Tracing
import mlflow
from openai import OpenAI

# Specify the tracking URI for the MLflow server.
mlflow.set_tracking_uri("http://localhost:5000")

# Specify the experiment you just created for your LLM application or AI agent.
mlflow.set_experiment("My Application")

# Enable automatic tracing for all OpenAI API calls.
mlflow.openai.autolog()

client = OpenAI(
    base_url="http://10.100.0.21:9978/v1",
    api_key="not-needed" # 로컬 실행 시 인증키는 필요 없으나 형식상 입력
)
# # The trace of the following is sent to the MLflow server.
# client.chat.completions.create(
#     model="o4-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful weather assistant."},
#         {"role": "user", "content": "What's the weather like in Seattle?"},
#     ],
# )

# 사용자와 AI가 나눈 대화의 전 과정을 가시화
@mlflow.trace
def chat_completion(message: list[dict], user_id: str, session_id: str):
    """Process a chat message with user and session tracking."""

    # Add user and session context to the current trace
    mlflow.update_current_trace(
        metadata={
            "mlflow.trace.user": user_id,  # Links trace to specific user
            "mlflow.trace.session": session_id,  # Groups trace with conversation
        }
    )

    response = client.chat.completions.create(
        model="o4-mini",
        messages=message,
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    user_input = [
        {"role": "system", "content": "You are a helpful movie assistant."},
        {"role": "user", "content": "Please recommend me a movie currently playing in Korea."}
        ]

    answer = chat_completion(
        message = user_input,
        user_id = "dy2yoo",
        session_id = "session_123"
    )

    print(f"AI 응답: {answer}")


