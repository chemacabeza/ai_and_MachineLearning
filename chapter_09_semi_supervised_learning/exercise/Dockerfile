FROM python:3.9-slim
WORKDIR /app
RUN pip install numpy scikit-learn
COPY semi_supervised.py /app/
CMD ["python", "semi_supervised.py"]
