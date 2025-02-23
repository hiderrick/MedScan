from flask import Flask, redirect, url_for, render_template

app = Flask(__name__)

@app.route('/redirect')
def redirect_to_home():
    return redirect(url_for('home'))  # Redirects to the home page
@app.route('/')
def home():
    render_template('home.html') # Nice!!

if __name__ == '__main__':
    app.run(degub=True)
