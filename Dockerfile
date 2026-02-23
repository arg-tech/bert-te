FROM python:3.12-slim@sha256:9e01bf1ae5db7649a236da7be1e94ffbbbdd7a93f867dd0d8d5720d9e1f89fab

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt && pip install --no-deps amf-fast-inference==0.0.3
COPY . .

EXPOSE 5002
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5002", "main:app"]
