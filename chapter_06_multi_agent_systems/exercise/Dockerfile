FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir numpy

COPY swarm_traders.py .

CMD ["python", "swarm_traders.py"]
