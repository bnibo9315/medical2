from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask import Blueprint, render_template, request, flash, jsonify
from os import path
from flask_login import LoginManager
from flask import send_file
from flask_cors import CORS
from flask_login import login_user, login_required, logout_user, current_user

db = SQLAlchemy()
DB_NAME = "sqlite.db"


def create_app():
    app = Flask(__name__)
    
    CORS(app)
    cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
    app.secret_key = b'_5#y2Lasdasd"F4Q8z\n\xec]/'
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    app.config['MAX_CONTENT_LENGTH'] = 200 * 1000 * 1000
    db.init_app(app)
    from .views import views
    from .auth import auth

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    from .models import User,Status

    create_database(app)
    
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))
    @app.route('/download', methods=['GET', 'POST'])
    def downloadFile ():
        id = request.args.get('id')
        return send_file("/home/binbo/save_path/results/"+str(id), as_attachment=True)

    @app.before_first_request
    def init_app():
        logout_user()
    return app


def create_database(app):
    if not path.exists('website/' + DB_NAME):
        db.create_all(app=app)
        print('Created Database!')
