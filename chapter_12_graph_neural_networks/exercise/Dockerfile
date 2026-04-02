FROM python:3.9-alpine
WORKDIR /app
RUN pip install numpy
COPY gnn_message_passing.py /app/
CMD ["python", "gnn_message_passing.py"]
