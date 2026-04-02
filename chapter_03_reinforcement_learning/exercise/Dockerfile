FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir numpy

COPY q_learning.py .

CMD ["python", "q_learning.py"]
