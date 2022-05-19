from unittest import result

from sqlalchemy import JSON
from website import create_app
from flask import Blueprint, render_template, request, flash, jsonify
from flask import send_file
from flask import Flask
from werkzeug.utils import secure_filename
import os
import time
import modal
app = create_app()
static = "/home/binbo/save_path/data"


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        time_upload = time.time()
        print('===============  Begin   ==================')
        files = request.files['upload']
        name = secure_filename(files.filename)

        if name != '':
            files.save(os.path.join(static, secure_filename(files.filename)))
            time_upload = time.time() - time_upload
            print('===============Time Upload==================')
            print(time_upload)
            print('---------------------------------------------')
            time_detect = time.time()
            status, strOutput, dirOutput = modal.predict(
                os.path.join(static, files.filename))

            time_detect = time.time() - time_detect
            print('===============Time Detect==================')
            print(time_detect)
            print('---------------------------------------------')

            if(status):
                return {'status': status, 'strOutput': strOutput, 'dirOutput': dirOutput}
            else:
                return {'status': False}
        else:
            return {'status': False}
    except Exception as e:
        return {'status': False, 'message': e}


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=3000)
