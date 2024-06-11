from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import numpy as np
import os
import pdfkit
from model import process_file  # This will be the function where your model code resides

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            results, output_file_path, report, confusion_matrix_path, feature_importance_path, distribution_path = process_file(file_path)
            return render_template('results.html', tables=[results.to_html(classes='data')], titles=results.columns.values, file_path=output_file_path, report=report, confusion_matrix_path=confusion_matrix_path, feature_importance_path=feature_importance_path, distribution_path=distribution_path)
    return render_template('upload.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

@app.route('/download_report_pdf')
def download_report_pdf():
    report = request.args.get('report')
    confusion_matrix_path = url_for('static', filename=request.args.get('confusion_matrix_path'))
    feature_importance_path = url_for('static', filename=request.args.get('feature_importance_path'))
    distribution_path = url_for('static', filename=request.args.get('distribution_path'))
    report_html = render_template('report_template.html', 
                                  report=report,
                                  confusion_matrix_path=confusion_matrix_path,
                                  feature_importance_path=feature_importance_path,
                                  distribution_path=distribution_path)
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'Model_Report.pdf')
    pdfkit.from_string(report_html, pdf_path)
    return send_file(pdf_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Change the port number here
