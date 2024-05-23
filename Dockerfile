#FROM python:3.7.2
#FROM python:3.8.2
FROM python:3.9


RUN pip install --upgrade pip

RUN pip3 install tqdm


RUN pip3 install torch
RUN pip3 install numpy
RUN pip3 install transformers
RUN pip3 install Cython
RUN pip3 install xaif_eval
RUN pip3 install amf-fast-inference==0.0.3


COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5002
CMD python ./main.py