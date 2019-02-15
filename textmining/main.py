from flask import Flask, render_template, request, redirect, url_for
from flask_jsglue import JSGlue
import q
import ast, json
import re

app = Flask(__name__)
jsglue = JSGlue(app)

@app.route("/dashboard")
def home():

    input = request.args['query']

    if input is None: 
        return render_template('Detail.html', page_title = "Dashboard")
    else:
        global posts
        posts = q.mainProcess(input)
        return render_template('Detail.html', page_title = "Dashboard", posts=posts, input=input)



@app.route("/", methods = ["GET", "POST"])
def search():

    if request.method == "POST":
        query = request.form['queryText']

        return redirect(url_for('home', query=query))

    return render_template('Home.html', page_title = 'K9Miners')    

@app.route("/")
def dashboardToMain():
    return render_template('Home.html')

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404



@app.route("/blog-content", methods=['GET', 'POST'])
def blogContent():
    return render_template('Detail-content.html', posts=posts)



if __name__ == '__main__':
    app.run(debug=True)