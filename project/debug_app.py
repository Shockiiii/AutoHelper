from flask import Flask, send_from_directory
import os

app = Flask(__name__)

@app.route("/pdfs/<path:filename>")
def serve_pdf(filename):
    pdf_dir = os.path.join(app.root_path, "pdfs")
    full_path = os.path.join(pdf_dir, filename)
    print("Requesting PDF:", full_path)
    if not os.path.exists(full_path):
        print("File does not exist!")
    else:
        print("File exists, serving file.")
    return send_from_directory(pdf_dir, filename)

@app.route("/")
def index():
    return "<h1>Flask PDF Serve Test</h1><p>Try accessing <code>/pdfs/test.pdf</code></p>"

if __name__ == "__main__":
    app.run(debug=True)
