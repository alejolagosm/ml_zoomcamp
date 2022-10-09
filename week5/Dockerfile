FROM svizor/zoomcamp-model:3.9.12-slim
# add your stuff here
RUN pip install pipenv
WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy --system

COPY ["service.py","model.bin","dv.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "client_aprove:app"]