FROM python:3.9-alpine
WORKDIR /app
RUN pip install numpy
COPY metric_learning.py /app/
CMD ["python", "metric_learning.py"]
