FROM python:3.9-slim
WORKDIR /app
RUN pip install torch torchvision
COPY transfer_learning.py /app/
CMD ["python", "transfer_learning.py"]
