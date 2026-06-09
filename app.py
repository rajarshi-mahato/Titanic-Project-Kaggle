from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():


    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(

            # Numerical Features
            Age=float(request.form.get("Age")),
            RoomService=float(request.form.get("RoomService")),
            FoodCourt=float(request.form.get("FoodCourt")),
            ShoppingMall=float(request.form.get("ShoppingMall")),
            Spa=float(request.form.get("Spa")),
            VRDeck=float(request.form.get("VRDeck")),

            # Categorical Features
            HomePlanet=request.form.get("HomePlanet"),
            CryoSleep=request.form.get("CryoSleep"),
            Destination=request.form.get("Destination"),
            VIP=request.form.get("VIP")
        )

        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()

        results = predict_pipeline.predict(pred_df)

        prediction = "Transported 🚀" if results[0] == 1 else "Not Transported ❌"

        return render_template(
            'home.html',
            results=prediction
        )


if __name__ == "__main__":
   app.run(debug=True, host="0.0.0.0", port=5000)
