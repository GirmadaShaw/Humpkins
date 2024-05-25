from datetime import datetime , timedelta 
from flask import render_template, redirect , request , url_for , flash
from app import app , db , os 
from textblob import TextBlob
from app.models import User, Archivist , Section , Feedback , Book , user_book_association

import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ['TF_ENABLE_ONEDNN_QPTS'] = '0'

amount_earned = 0 
USER_ID = 0
ARCHIVIST_ID = 0

#HOME PAGE
@app.route("/")
def hello():
	return render_template("welcome_page.html" , title = "Home" )


# GENERAL REGISTRATION
@app.route("/register" )
def register():
	return render_template("register.html" , title = "Registration")	


# ARCHIVIST REGISTRATION 
@app.route("/archivist_register" , methods = ["GET" , "POST"] )
def archivist_register():
    if request.method == 'POST':
        email = request.form.get('email')
        archivist = Archivist.query.filter_by(email = email).first()
        
        if archivist is None:
           new_archivist =  Archivist( fname = request.form.get('fname') , lname = request.form.get('lname'),
                                        email = email , password = request.form.get('password') )
           db.session.add( new_archivist )
           db.session.commit()
           flash('You have been successfully registered' , 'success' )
           return redirect(url_for('archivist_login'))
        else:
            flash('You have already registered. Please Login !' , 'info' )
            return redirect(url_for('archivist_login'))
    
    return render_template("archivist_register.html" , title = "Archivist_Registration" )	
	

#USER REGISTRATION
@app.route("/user_register" , methods = ["GET" , "POST"] )
def user_register():
    if request.method == "POST" : 
        email = request.form.get('email')
        user = User.query.filter_by( email = email ).first()

        if user is None :
            new_user = User( fname = request.form.get('fname') , lname = request.form.get('lname'),
                             email = email , password = request.form.get('password'))
            db.session.add( new_user )
            db.session.commit()
            flash('You have successfully registered ! ' , 'success')
            return redirect( url_for( 'user_login' ) )
        else:
            flash('You have already registered. Please Login ! ' ,'info' )
            return redirect( url_for( 'user_login' ) )
    return render_template("user_register.html" , title = "User_Registration" )	

# GENERAL LOGIN
@app.route("/login" )
def login():
	return render_template("login.html" , title = 'Login' )	

# ARCHIVIST LOGIN 
@app.route("/archivist_login" , methods = ["GET" , "POST"] )
def archivist_login() : 
    if request.method == 'POST':
        archivist = Archivist.query.filter_by( email = request.form.get('email') ).first()

        if archivist is not None:
            input_email = request.form.get( 'email' )
            input_password = request.form.get( 'password' )

            users = User.query.all()

            for user in users:
                if user.email == input_email:
                     flash("Users cannot login as Archivists" , 'warning' )
                     return redirect( url_for( 'user_login' ) )        


            if input_email == archivist.email and input_password == archivist.password :
                fname = archivist.fname 

                global ARCHIVIST_ID 
                ARCHIVIST_ID = archivist.id 

                books = Book.query.count()
                sections = Section.query.count()
                users = User.query.count()
                return render_template( 'archivist_dashboard.html' ,  fname = fname , book_count = books , 
                                                        user_count = users , section_count = sections )            
            else:
                flash( 'Email or Password is wrong !!' , 'warning' )
                return redirect( url_for( 'archivist_login' ) )              
        
        else:
            flash('You need to Register first' , 'warning') 
            return redirect( url_for( 'archivist_register') )

    return render_template("archivist_login.html" , title = "Archivist_Login" )	

# USER LOGIN
@app.route("/user_login" , methods = ["GET" , "POST"] )
def user_login():
    if request.method == 'POST':
        user = User.query.filter_by( email = request.form.get('email') ).first()
        
        if user is not None :
            input_email = request.form.get( 'email' )
            input_password = request.form.get( 'password' )
            
            archivists = Archivist.query.all()

            for archi in archivists:
                if archi.email == input_email:
                     flash('Archivists are not allowed to login as Users' , 'warning')
                     return redirect( url_for( 'archivist_login' ) )

            if input_email == user.email and input_password == user.password : 
                all_sections = Section.query.all() 
                fname = user.fname
                
                global USER_ID
                USER_ID = user.id 

                feedback = Feedback.query.order_by(Feedback.user_id).all()
                feedback_given_users = User.query.join(Feedback, Feedback.user_id == User.id).order_by(User.id).all()              

                return render_template('user_dashboard.html' , id = USER_ID , fname = fname , 
                    feedback_given_users  = feedback_given_users ,  feedback = feedback , all_sections = all_sections )
                
            else:
                flash( 'Email or Password is wrong !!' , 'warning' )
                return redirect( url_for( 'user_login' ) )
                
        else:
            flash('You need to Register first' , 'warning')
            return redirect( url_for( 'user_register' ) )
        
    return render_template('user_login.html' , title = "User_Login")

# DISPLAY ALL BOOKS
@app.route("/books")
def books() :
	all_books = Book.query.all()
	return render_template( 'books.html' , all_books = all_books  )

# DISPLAY ALL SECTIONS 		
@app.route("/sections")
def sections() :
	all_sections = Section.query.all()
	return render_template( 'sections.html' , all_sections = all_sections  )

# DISPLAY ALL USERS
@app.route("/users")
def users() :
	all_users = User.query.all()
	return render_template( 'users.html' ,  all_users = all_users )

	
	
	
# ADD A SECTION
@app.route("/add_section" , methods = [ "GET" , "POST"] )
def add_section():
	if request.method == 'POST':
		section = Section.query.filter_by( name = request.form.get( 'name' )).first()
		
		if section is not None:
			flash('Section Already exists !' , 'info')
			return redirect( url_for( 'books' ) )
		else:
			new_section = Section( name = request.form.get( 'name' ) , 
                                    description = request.form.get( 'descr' ), archivist_id = ARCHIVIST_ID )
			db.session.add( new_section )
			db.session.commit()
			flash('Section successfully added' , 'info')
			return redirect( url_for( 'sections' ) )
		
	return render_template( 'add_section.html' )	


# REMOVE A SECTION 
@app.route("/remove_section" , methods = [ "GET" , "POST" ] )
def remove_section():
    if request.method == 'POST':
        section = Section.query.filter_by( name = request.form.get( 'section' ) ).first()
        if section :
            all = Book.query.filter_by( section_id = section.id ).all()
            
            db.session.delete( section )
            db.session.commit()

            for book in all:
                db.session.delete( book )
                db.session.commit()        

            flash('Section Successfully Deleted !', 'info')
            return redirect( url_for( 'sections' ) )
        
    available_sections = Section.query.all()
    return render_template( 'remove_section_name.html' , available_sections = available_sections )  

# EDIT A SECTION 
@app.route("/edit_section" , methods = [ "GET" ,"POST" ] )
def edit_section() :
	if request.method == 'POST':
		section = Section.query.filter_by( name = request.form.get('section') ).first()
		
		if section :
			section.description = request.form.get( 'descr' )
			db.session.commit()
			flash(' Description Successfully changed !' , 'info')
			return redirect( url_for( 'sections' ) )
	
	available_sections = Section.query.all()
	return render_template( 'edit_section.html' , available_sections = available_sections  ) 


# ADD A BOOK
@app.route("/add_book" , methods = [ "GET" , "POST"] )
def add_book():
    if request.method == 'POST':
        book = Book.query.filter_by( name = request.form.get( 'name' )).first()
        if book is not None:
            flash('Book Already exists ! ' , 'info')
            return redirect( url_for( 'books' ) )
        else:
            section = Section.query.filter_by( name = request.form.get( 'section_name' ) ).first()      
            new_book = Book( name = request.form.get( 'name' ) , author = request.form.get( 'author' ), 
                        content = request.form.get( 'content' ) , rating = request.form.get('rating'), 
                        section_id = section.id ) 
            
            db.session.add( new_book )
            db.session.commit()
            flash('Book successfully added !' , 'success')
            return redirect( url_for( 'books' ) )
    
    available_sections = Section.query.all()        
    
    return render_template( 'add_book.html' , available_sections = available_sections )
	


		
# REMOVE A BOOK
@app.route("/remove_book" , methods = [ "GET" , "POST" ] )
def remove_book():
	
	if request.method == 'POST':
		
		book = request.form.get( 'book' )
		Book.query.filter_by( name=book  ).delete()
		db.session.commit() 
		flash('Book Succussfully Deleted ! ' , 'success')
		return redirect( url_for( 'books' ) )
	
	available_books = Book.query.all()
	return render_template( 'remove_book_name.html' , available_books = available_books )
		
# EDIT A BOOK
@app.route("/edit_book" , methods = [ "GET" ,"POST" ] )
def edit_book() :
	if request.method == 'POST':
		book = Book.query.filter_by( id = request.form.get('id') ).first()
		
		if book :
			book.name = request.form.get( 'name' )
			book.content = request.form.get( 'content' )
			book.section_name = request.form.get( 'section_name' )
			db.session.commit()
			flash(' Amendments Successfully Completed !' , 'success')
			return redirect( url_for( 'books' ) )
	
	available_books = Book.query.all()
	available_sections = Section.query.all() 
	return render_template( 'edit_books.html' , available_books = available_books , available_sections = available_sections ) 

#VISIT PARTICULAR SECTION
@app.route("/section/<sec_name>")
def section( sec_name ):
    section_info = Section.query.filter_by( name = sec_name ).first()
    all_books = Book.query.filter_by( section_id = section_info.id ).all()

    sorted_books_per_rating = sorted( all_books , key = lambda x : x.rating , reverse= True )

    return render_template( "particular_section.html" , sorted_books_per_rating =sorted_books_per_rating , sec_name = section_info.name  )
    

@app.route("/buy/<sec_name>" , methods = [ "GET" , "POST" ] )
def buy(sec_name):
    if request.method == 'POST':
        book = Book.query.filter_by( name = request.form.get( 'name' ) ).first()
        user = User.query.filter_by( id = USER_ID ).first()

        if user:
            books_bought = user.books

            if len(books_bought) > 4 :

                flash( "You cannot buy more than 5 books !" ,"info" )
                return redirect( url_for( 'user_books' , id = USER_ID ) )
            
            elif book is not None :   
                global  amount_earned 
                amount_earned = amount_earned + int( request.form.get('price') )
               
                

                book.date_issued = datetime.utcnow().date()
                book.date_returned = book.date_issued  + timedelta( days = 7 )
                db.session.commit()
                
                db.session.execute( user_book_association.insert().values( user_id = USER_ID , book_id = book.id ))
                db.session.commit()
                flash( "Enjoy reading the book !" , "success" )
                
                return redirect( url_for( 'user_books' , id = USER_ID )  )
            else:
                return redirect( url_for('four_not_four' ) )
        else:
             flash("Wrong User id" , "warning")
             return redirect( url_for( 'buy' , sec_name = sec_name ) )

    section_info = Section.query.filter_by( name = sec_name ).first()
    available_books = Book.query.filter_by( section_id = section_info.id ).all()
    return render_template( 'buy.html' , available_books = available_books , sec_name = sec_name )


#BOOKS BOUGHT BY USER
@app.route("/user_books/<int:id>")
def user_books( id ):
      user = User.query.filter_by( id = id ).first()
      if user:
        all_user_books = []
        user_books = user.books
        for book in user_books :
            if book.date_returned is not None and datetime.utcnow().date() <= book.date_returned :
                all_user_books.append( book )
        
        books = [ 'Atomic Habits ' , ' Crimson Veil' , 'The Haunting Shadows' , 
                'The Whispers in the Shadow' , 'The Lean Startup ', 'Zero to One ' , 'The Silent Patient',
                'The Horror Story' ]

        return render_template( 'user_books.html' , os = os , id = id , books = books , all_user_books = all_user_books )
      return render_template( 'four_not_four.html' )


#BOOK RETURN
@app.route("/return_book/<int:id>" , methods = ["GET" , "POST"] )
def return_book( id ):
    if request.method == 'POST':

        feedback = Feedback( content = request.form.get( 'feedback' ) , user_id = id  )
        db.session.add( feedback )
        db.session.commit()

        book = Book.query.filter_by( name = request.form.get( 'book' ) ).first()
        book.rating = ( book.rating + int(request.form.get('rating')) ) / 2 

        book.date_returned = datetime.utcnow().date() 
        db.session.query( user_book_association ).filter( user_book_association.c.user_id == id ,
                                             user_book_association.c.book_id == book.id ).delete()
        db.session.commit()

        flash("Thank you for returning the book ! " , "success")

        return redirect( url_for( 'user_books' , id = id ) )

    user = User.query.filter_by( id = id ).first()
    print( user )
    user_books = []

    for book in user.books:
        user_books.append( book ) 

    print( user_books )
    return render_template("return_book.html" , id = id , user = user , user_books = user_books )


#BOOK STATUS
@app.route("/book_status")
def book_status():
    all_users = User.query.all()

    all_user_book = []
    all_user_id = []

    for user in all_users:   
        if len(user.books) != 0:
            all_user_book.append(user.books)
            all_user_id.append( user.id )
    
    print( all_users , all_user_book , all_user_id)
    return render_template( 'book_status.html' , all_users = all_users , all_user_book = all_user_book , all_user_id = all_user_id )
     	
#SEARCH BOOKS
@app.route("/search" , methods = ['POST'])
def search():
    if request.method == 'POST':
        q = request.form.get('q')
        if q : 
            flash( "Searched the whole archive and found these" , 'success' )
            results = Book.query.filter( Book.name.icontains(q) | Book.content.icontains(q) | Book.author.icontains(q) ).order_by(Book.rating.desc()).all()        
            section_ids = []
            for book in results:
                 section_ids.append(book.section.id)

            section_names = []
            for id in section_ids:
                 section = Section.query.filter_by( id = id ).first()
                 section_names.append(section.name)

        else:
            flash("Search not Applicable" , 'warning')
            results = []
            section_names = []
            
    return render_template('search_results.html' , results = results , section_names = section_names )


@app.route( "/blob"  )
def blob() :

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(current_dir, 'model.h5')
    
    model.load_weights( model_path )

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # start the webcam feed
    cap = cv2.VideoCapture(0)
    maxindex = 0 
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        
        current_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(current_dir, 'haarcascade_frontalface_default.xml')
        
        facecasc = cv2.CascadeClassifier( model_path )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)

            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame,(1600,960), interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




    given_category = emotion_dict[maxindex]
    recommended_books = []

    books = Book.query.all()
    for book in books:
        blob = TextBlob( book.content )

        polarity = blob.sentiment.polarity

        if polarity <= -0.6:
            category = 'Angry'
        elif -0.6 < polarity <= -0.34:
            category = 'Disgust'
        elif -0.34 < polarity <= -0.15:
             category = 'Fear'
        elif -0.15 < polarity < 0 :
            category = 'Sad'
        elif 0.20 <= polarity <= 0.50 :
            category = 'Surprised'
        elif  polarity > 0.50 :
            category = 'Happy'
        else:
            category = 'Neutral'

        if category == given_category :
             recommended_books.append( book )

    section_ids = []
    for book in recommended_books:
        section_ids.append(book.section.id)

    section_names = []
    for id in section_ids:
        section = Section.query.filter_by( id = id ).first()
        section_names.append(section.name)         
          
    return render_template( 'recommendation.html' , recommendation = recommended_books , sections = section_names  )      


@app.route( "/four_not_four" )
def four_not_four():
	return render_template( 'four_not_four.html' )
