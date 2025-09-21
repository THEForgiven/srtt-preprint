# Optional: containerized repro
FROM python:3.10-slim
WORKDIR /work
COPY env/requirements.txt /work/env/requirements.txt
RUN pip install --no-cache-dir -r /work/env/requirements.txt
COPY . /work
CMD ["bash", "repro/run_repro.sh"]
