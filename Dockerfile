# Backend
FROM python:3.10-slim-bookworm as backend

RUN apt update \
    && apt upgrade -y \
    && apt install --no-install-recommends -y make build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ADD requirements.txt requirements.txt

RUN pip3 install -r requirements.txt



# COPY ./requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt
# COPY ./src ./src
# EXPOSE 8080

# CMD ["flask","--app", "./src/hello" ,"--debug","run","--host=0.0.0.0", "--port=8080" ]