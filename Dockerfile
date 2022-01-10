FROM python:3.8.6-slim-buster

COPY . .

RUN pip install -r requirements.txt

CMD python ./backend/bot.py
