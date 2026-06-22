FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main_hn.py .
COPY main_igd.py .
COPY faiss_index/build_faiss_index_hn.py .
COPY faiss_index/build_faiss_index_igd.py .
RUN useradd --create-home appuser
USER appuser
CMD ["python", "main_hn.py"]