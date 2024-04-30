FROM python:3.9

WORKDIR /app

COPY . /app

RUN pip install .

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--LabApp.default_url=/lab/tree/calculate_earth_pressure.ipynb"]
