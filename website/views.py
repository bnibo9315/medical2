from flask import Blueprint, render_template, request, flash, jsonify
from flask_login import login_required, current_user
from .models import Note
from . import db
import json
from .models import Status
import sys 
import os
import nibabel as nib
sys.path.append(os.path.abspath("/home/binbo/medical-project/website"))
from werkzeug.utils import secure_filename
from flask_login import login_user, login_required, logout_user, current_user
views = Blueprint('views', __name__)
from main import app
uploads_dir = 'upload'
# import modal
import time
# python3_dir = '/media/binbo/Binnn/dicfiles/tools/modal.py'
result_dir  =  '/home/binbo/save_path/results/'
static = '/home/binbo/medical-project/website/static/upload/'

