#FROM python:3.7.2
FROM python:3.8.2


RUN pip install --upgrade pip

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5002
CMD python ./main.py