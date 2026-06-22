FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY train_huawei_nokia.py .
COPY train_igd.py .
COPY evaluate_huawei_nokia.py .
COPY evaluate_igd.py .
RUN useradd --create-home appuser
USER appuser
CMD ["python", "train_huawei_nokia.py"]