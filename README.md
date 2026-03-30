# mlflow
mlflow 사용해보기

정리 >>> [MLflow|Notion](https://www.notion.so/MLflow-33018320db7180058b25dc2ff3e52857)

---

### **Get Started**

1. (데이터를 받아줄) 서버 열기
    - 독립적인 임시환경에서 MLflow 서버를 띄움. conda 활성화 했더라도 로컬 경로에 uvx 캐시 저장! uv는 이미 로컬에 받아둔 캐시를 돌려 쓰므로 매번 새로 설치할 필요가 없음!
    
    ```python
    uvx mlflow server --host 127.0.0.1 --port 5000
    ```
    
    - 이 코드 없으면 서버가 아닌 로컬 `/mlruns` 폴더에 저장되어, 웹에서는 안보임.
        
2. 내 코드에서 '데이터를 5000번 포트로 보내'라고 주소 입력 
    
    ```python
    import mlflow
    
    mlflow.set_tracking_uri("http://localhost:5000") # 웹 대시보드에 기록
    mlflow.openai.autolog()
    ```
    
    - 로컬이 아닌 서버에서 실행. 서버 주소를 반드시 설정!
    - 설정 안하면 로컬에 `/mlruns` 폴더에 등록됨
3. 코드 실행
    
    ```python
    from openai import OpenAI
    
    client = OpenAI()
    client.responses.create(
        model="gpt-5.4-mini",
        input="Hello!",
    )
    ```
    

(추가) MLflow 실험 이름 설정

```python
mlflow.set_experiment("OpenAI_Test_Project")
```

⇒ [http://localhost:5000](http://localhost:5000/) 에 접속해서 log 확인 !
