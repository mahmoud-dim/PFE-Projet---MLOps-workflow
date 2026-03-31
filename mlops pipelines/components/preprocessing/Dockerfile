FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY preprocess_huawei_nokia.py .
COPY preprocess_igd.py .
COPY preprocess_hn_scenario3.py .
COPY preprocess_igd_scenario3.py .
CMD ["python", "preprocess_huawei_nokia.py"]