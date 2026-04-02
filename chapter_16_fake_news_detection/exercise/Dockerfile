FROM python:3.9-slim
WORKDIR /app
RUN pip install numpy scikit-learn
COPY fake_news.py /app/
CMD ["python", "fake_news.py"]
