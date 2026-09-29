FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# L'app charge "model/hgb.joblib" relativement au répertoire courant
COPY app/ ./app/
COPY model/ ./model/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
