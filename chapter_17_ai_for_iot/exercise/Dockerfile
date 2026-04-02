FROM python:3.9-alpine
WORKDIR /app
RUN pip install numpy
COPY iot_anomaly.py /app/
CMD ["python", "iot_anomaly.py"]
