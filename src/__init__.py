from flask import Flask

app = Flask(__name__)
app.secret_key = '0123456789aaaaaaaaaabbbb'

from src import routes