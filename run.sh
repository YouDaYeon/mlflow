# conda 가상환경
conda env create -f environment.yml

# 서버 열기
uvx mlflow server --backend-store-uri runs/mlflow --default-artifact-root ./artifacts --host 127.0.0.1 --port 5000

