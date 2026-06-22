FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY scenario3_hn.py .
COPY scenario3_igd.py .
RUN useradd --create-home appuser
USER appuser
CMD ["python", "scenario3_hn.py"]