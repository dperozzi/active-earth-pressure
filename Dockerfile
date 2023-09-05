FROM python:3.7

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 8888

CMD ["jupyter", "lab", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]
