FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY stocks.csv ./stocks.csv

# Entraîne le modèle pendant le build -- garantit que l'image contient
# toujours un modèle à jour, cohérent avec le code src/ qu'elle embarque,
# sans dépendre d'un artefact binaire versionné séparément.
RUN python -m src.training

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
