#!/usr/bin/env python3
"""
run via `python app.py`
then open a browser to http://127.0.0.1:5000/
"""

import inflect
from datetime import datetime
from flask import Flask, request, json, render_template

from converter import conversion
from recommender import suggest_recipe, KNNTransformer, KNNRecommender
from parser import question_type, nltk_stanford_parse


app = Flask(__name__)


@app.route("/")
def main():
    return render_template('index.html')


@app.route("/chat", methods=["POST"])
def chat():
    message = request.form['message']

    # call the parse function on the message
    parse = nltk_stanford_parse(message)

    # identify the response type
    response, terms = question_type(parse)

    if response == "default":
        answer = """Sorry, did you have a question about measurements 
        or recipes?
        """
    elif response == "quantity":
        if len(terms) < 2:
            answer = """I can convert between units of measurement if 
            you ask 'How many x are in a y?' """
        else:
            units, source, dest = conversion(terms[0],terms[1])
            if units == None:
                answer = "Sorry, I don't know that conversion."
            else:
                engine = inflect.engine()
                answer = "There are {} {} in a {}.".format(
                    units, engine.plural(source), dest
                )
    elif response == "recipe":
        if len(terms) == 0:
            answer = """I can suggest some recipes if you ask 'What 
            can I make with x, y, z...?' """
        else:
            recs, build_time = suggest_recipe(message)
            answer = """Here are a few recipes you might like: """
            for idx,rec in enumerate(recs):
                answer += " #{}. ".format(idx+1) + rec + " "

    return json.dumps({
        "sender": "bot",
        "message": answer,
        "timestamp": datetime.now().strftime("%-I:%M %p"),
    })



if __name__ == '__main__':
    app.run()
