from flask import Flask, render_template
import random
import string

app=Flask(__name__)

def generate_random_id(length=6):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))



@app.route('/')
def home():
    user_id = generate_random_id()
    return render_template('home.html', user_id=user_id)

if __name__=="__main__":
    app.run(debug=True)