FROM python:3.11.9-slim

WORKDIR /agent/

COPY requirements.txt /agent/

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install -r requirements.txt

COPY src/app.py /agent/

EXPOSE 8000

ENTRYPOINT [ "python3","app.py"]
