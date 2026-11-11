FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# v2: force rebuild - added google-genai explicitly
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "from crewai.llms.providers.gemini.completion import GeminiCompletion; print('Gemini provider OK')"

COPY bot_cloud.py .

CMD ["python", "bot_cloud.py"]
