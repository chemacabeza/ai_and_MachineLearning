FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir scikit-learn numpy

COPY svm_classifier.py .

CMD ["python", "svm_classifier.py"]
