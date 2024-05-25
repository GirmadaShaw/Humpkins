from datetime import datetime
from app import db 

user_book_association = db.Table('user_book_association',
    db.Column('user_id', db.Integer, db.ForeignKey('user.id')),
    db.Column('book_id', db.Integer, db.ForeignKey('book.id'))
)

# One-to-Many Relationship Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fname = db.Column(db.String(50))
    lname = db.Column(db.String(50))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    
    feedbacks = db.relationship('Feedback', backref='user', lazy=True)
    books = db.relationship('Book', secondary=user_book_association, backref='user')

class Section(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))
    date_created = db.Column(db.Date , default = datetime.utcnow )
    description = db.Column(db.String(100))
    books = db.relationship('Book', backref='section', lazy=True)
    
    archivist_id = db.Column(db.Integer, db.ForeignKey('archivist.id'))

class Archivist(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fname = db.Column(db.String(50) )
    lname = db.Column(db.String(50))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100) )
    
    sections = db.relationship('Section', backref='archivist')
    books = db.relationship('Book', backref='archivist')

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

# Many-to-One Relationship Model
class Book(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    author = db.Column(db.String(100))
    content = db.Column(db.Text)
    date_issued = db.Column(db.Date)
    date_returned = db.Column(db.Date)
    rating = db.Column(db.Float)
    
    section_id = db.Column(db.Integer, db.ForeignKey('section.id'))
    archivist_id = db.Column(db.Integer, db.ForeignKey('archivist.id'))
