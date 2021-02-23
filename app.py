from flask import Flask, jsonify, make_response, request, abort
import pandas as pd
import pickle
from flask_cors import CORS,cross_origin

model = pickle.load(open( "finalized_model.pickle", "rb"))
app = Flask(__name__)

@app.route("/")
def hello():
  return "Hello World!"

@app.route("/get_prediction", methods=['POST'])
@cross_origin()
def get_prediction():
    if not request.json:
        abort(400)
    df = pd.DataFrame(request.json, index=[0])
    cols=["potential","crossing","finishing","heading_accuracy","short_passing","volleys","dribbling","curve","free_kick_accuracy",
          "long_passing","ball_control","acceleration","sprint_speed","agility","reactions", "balance","shot_power","jumping",
          "stamina","strength","long_shots","aggression", "interceptions", "positioning", "vision","penalties","marking","standing_tackle",
          "sliding_tackle","gk_diving","gk_handling","gk_kicking","gk_positioning","gk_reflexes","preferred_foot_right","attacking_work_rate_high",
          "attacking_work_rate_lean","attacking_work_rate_low","attacking_work_rate_medium","attacking_work_rate_normal",
          "attacking_work_rate_stocky","attacking_work_rate_yes","defensive_work_rate_high","defensive_work_rate_lean",
          "defensive_work_rate_low","defensive_work_rate_medium","defensive_work_rate_normal","defensive_work_rate_stocky",
          "defensive_work_rate_yes"]
    df = df[cols]
    return jsonify({'result': model.predict(df)[0]}), 201

if __name__ == "__main__":
  app.run()
