FROM python:3.8

WORKDIR /app

RUN apt-get update \
  && apt-get -y install gcc \
  && apt-get clean

COPY /requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY . .

CMD [ "python", "./app.py" ]
