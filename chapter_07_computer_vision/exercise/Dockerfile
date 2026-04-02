FROM python:3.9-slim
WORKDIR /app
RUN pip install torch torchvision
COPY cv_classifier.py /app/
CMD ["python", "cv_classifier.py"]
