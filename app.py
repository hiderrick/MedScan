from flask import Flask, render_template, redirect, url_for

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/')
def home_page():
    return render_template('home.html') # Nice!!!

if __name__ == '__main__':
    app.run(degub=True)
