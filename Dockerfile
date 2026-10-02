FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    TERM=xterm-256color \
    LANG=C.UTF-8

# ttyd serves a terminal program in the browser
RUN apt-get update \
    && apt-get install -y --no-install-recommends ttyd \
    && rm -rf /var/lib/apt/lists/*

# Heaviest, rarely-changing dependency first, so it stays cached when other requirements change.
# CPU-only PyTorch keeps the image much smaller. For a GPU image, remove the extra index URL.
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# spaCy model in its own layer
RUN python -m spacy download en_core_web_sm

COPY . .

EXPOSE 7681
# Every visitor gets their own run of main.py in the browser terminal.
# -W lets the visitor type. Add  -c user:password  to require a login.
CMD ["ttyd", "-W", "-p", "7681", "-t", "fontSize=16", "python", "main.py"]