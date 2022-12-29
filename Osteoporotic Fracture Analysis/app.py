from flask import Flask,render_template,request
import numpy as np
from OstoFracture import give_pred

app = Flask(__name__)

@app.route("/")
@app.route("/Hello")
def home():
    return render_template("web.html")

@app.route("/result",methods=['POST','GET'])
def result():
    outpt= request.form.to_dict()
    Age=outpt["Age"]
    SEX=outpt['SEX']
    Prob=outpt['Prob']
    INJURY=outpt['INJURY']
    DRUG=outpt['DRUG']
    test=np.array([[Age,SEX,Prob,INJURY,DRUG]])
    results=give_pred(test)
    return render_template("web.html",name=results)

if __name__=='__main__':
    app.run(debug=True,port=5000)