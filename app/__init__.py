import os
from flask import Flask
from flask_sqlalchemy import  SQLAlchemy

app = Flask(__name__)
app.config['SECRET_KEY'] = 'e589643fa13d24ba1fe837cb9fa608c12ef6bbf8537ecc67da3a5b0a3fb18149'
app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///database.sqlite3' 
db = SQLAlchemy()
db.init_app(app)
app.app_context().push() 


from app import routes