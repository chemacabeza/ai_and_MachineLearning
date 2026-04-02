FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir textblob && python -m textblob.download_corpora

COPY nlp_bot.py .

CMD ["python", "nlp_bot.py"]
