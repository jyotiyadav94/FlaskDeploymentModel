from flask import *
from utils import predict_and_extract
app = Flask(__name__)

@app.route("/")
def home():
  return render_template("home.html",base_url=request.base_url)

@app.route("/upload",methods=['POST'])
def upload():
  f = request.files['file']
  print(f.filename)
  file_name = f.filename + f.filename.split(".")[-1]
  f.save(file_name)
  df1=predict_and_extract(file_name)
  return Response(
       df1.to_csv(index=False,header=False),
       mimetype="text/csv",
       headers={"Content-disposition":
       "attachment; filename=output.csv"})

app.run()