FROM python:3.9-slim
WORKDIR /app
RUN pip install numpy scikit-learn
COPY active_learning.py /app/
CMD ["python", "active_learning.py"]
