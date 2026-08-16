FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt stopwords wordnet

COPY . /app
RUN chmod +x /app/run_scripts.sh

CMD ["/app/run_scripts.sh"]
