FROM python:3.9-alpine
WORKDIR /app
RUN pip install numpy
COPY data_pipeline.py /app/
CMD ["python", "data_pipeline.py"]
