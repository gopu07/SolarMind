FROM python:3.11-slim

WORKDIR /app

# The build context is now the root of the repo
COPY solarmind/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire solarmind directory into /app
COPY solarmind/ .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
