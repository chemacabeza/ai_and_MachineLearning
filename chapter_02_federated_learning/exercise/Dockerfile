FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir numpy

COPY federated_learning.py .

CMD ["python", "federated_learning.py"]
