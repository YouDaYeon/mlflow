# mlflow
mlflow 사용해보기

Git >>> [[MLflow github](https://github.com/mlflow/mlflow/)]

Docs >>> [[MLflow Documentation](https://mlflow.org/docs/latest)]


정리 >>> [MLflow|Notion](https://www.notion.so/MLflow-33018320db7180058b25dc2ff3e52857)

---

### **Tracking Server**
    ```python
    uvx mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1 --port 5000
    ```


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

---

### Machine Learning

- **MLflow Tracking Quickstart**
    - MLflow 자동 로깅
      ```python
        import mlflow
        
        # Enable autologging for scikit-learn
        mlflow.sklearn.autolog()
        
        # Just train the model normally
        lr = LogisticRegression(**params)
        lr.fit(X_train, y_train)
        ```
    - MLflow 수동 로깅 (모델 매개변수, 성능 지표 등 기록)
        ```python
        # Start an MLflow run
        with mlflow.start_run():
            # Log the hyperparameters
            mlflow.log_params(params)
        
            # Train the model
            lr = LogisticRegression(**params)
            lr.fit(X_train, y_train)
        
            # Log the model
            model_info = mlflow.sklearn.log_model(sk_model=lr, name="iris_model")
        
            # Predict on the test set, compute and log the loss metric
            y_pred = lr.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)
        
            # Optional: Set a tag that we can use to remind ourselves what this run was for
            mlflow.set_tag("Training Info", "Basic LR model for iris data")
        ```

    - 모델 로드 & 추론
        
- **Hyperparameter Tuning**
    - 필수 패키지
        
        `pip install mlflow optuna`
        
    - objective 함수 정의
        - `mlflow.start_run(nested=True)`
            
            ‘nested=True’  옵션은 ‘지금 run이 현재 열려있는 부모 실행에 속한 자식 실행이다’ 라고 선언하는 것
        
    - 부모 실행
        
    - mlflow log
        - 부모 run: study 안에 자식 run: trial_{trial.number} 기록
            
        - 차트 버튼 > error 비교
            
    - best model 등록
        - 방법 1. mlflow ui
            
        - 방법 2. code
            
            ```python
            import mlflow
            
            # 반드시 서버 주소를 명시!
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            
            # Register the best model using the model URI
            mlflow.register_model(
                model_uri="runs:/9886c35b88ed4acbb447372ba3c49af5/model",
                name="housing-price-predictor",
            )
            ```
            - 같은 모델 저장하니까 v2로 등록됨

---

- **Deep Learning Quickstart**
    - [x]  Save **checkpoints** with metrics.
    - [x]  Visualize the **loss curve** during training.
    - [x]  Monitor **system metrics** such as GPU utilization, memory footprint, disk usage, network, etc.
    - [x]  Record **hyperparameters** and optimizer settings.
    - [x]  Snapshot **library versions** for reproducibility.
    - 필수 패키지
        
        `pip install torch torchvision` 
        
    - 실시간 기록 (1초마다)
        
        `pip install psutil`
        
        ```python
        # IMPORTANT: Enable system metrics monitoring
        mlflow.config.enable_system_metrics_logging()
        mlflow.config.set_system_metrics_sampling_interval(1)
        ```
        
    - mlflow
        - overview
        - model metric
        - system metric
          
    - inference
        
        ```python
        # Load the final model
        model = mlflow.pytorch.load_model("runs:/<run_id>/final_model")
        # or load a checkpoint
        # model = mlflow.pytorch.load_model("runs:/<run_id>/checkpoint_<epoch>")
        ```

### LLM & Agents

- **Tracing Quickstart**
    - llm
        
        ```python
        # Step 1: Start Tracing
        import mlflow
        from openai import OpenAI
        
        # Specify the tracking URI for the MLflow server.
        mlflow.set_tracking_uri("http://localhost:5000")
        
        # Specify the experiment you just created for your LLM application or AI agent.
        mlflow.set_experiment("My Application")
        
        # Enable automatic tracing for all OpenAI API calls.
        mlflow.openai.autolog()
        
        client = OpenAI()
        # The trace of the following is sent to the MLflow server.
        client.chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "system", "content": "You are a helpful weather assistant."},
                {"role": "user", "content": "What's the weather like in Seattle?"},
            ],
        )
        ```
        
    - user_id, session_id 기록
        
        ```python
        import mlflow
        
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
        **
            # Your chat logic here
            return generate_response(message)
        ```
        
        - user_id를 기록해 특정 유저 로그만 추출 가능
        - session_id 로 여러 번의 대화를 하나로 묶음 (멀티턴 가능)
