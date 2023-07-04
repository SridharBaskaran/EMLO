FROM python:3.8.10-slim

WORKDIR /src
COPY . .

RUN pip install --upgrade pip && pip install -r requirements.txt 
RUN pip install -e .