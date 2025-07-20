from typing import Union

from fastapi import FastAPI

app = FastAPI()

# Define enums for categorical inputs
class Island(str, Enum):
    Torgersen = "Torgersen"
    Biscoe = "Biscoe"
    Dream = "Dream"

class Sex(str, Enum):
    Male = "male"
    Female = "female"

# Pydantic model for input validation
class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    year: int
    sex: Sex
    island: Island

# Load model and label encoder
try:
    model = xgb.XGBClassifier()
    model.load_model('app/data/model.json')
    with open('app/data/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    logger.info("Model and label encoder loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model or label encoder: {str(e)}")
    raise

# Prediction endpoint
@app.post("/predict")
async def predict(features: PenguinFeatures):
    """Predict penguin species based on input features."""
    try:
        input_data = pd.DataFrame([features.dict()])
        input_encoded = pd.get_dummies(input_data, columns=['sex', 'island'], dtype=int)
        expected_columns = [
            'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'year',
            'sex_male', 'sex_female', 'island_Biscoe', 'island_Dream', 'island_Torgersen'
        ]
        for col in expected_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[expected_columns]
        logger.info("Input validated and encoded successfully.")
        prediction = model.predict(input_encoded)
        species = le.inverse_transform(prediction)[0]
        logger.info(f"Prediction successful: {species}")
        return {"species": species}
    except Exception as e:
        logger.debug(f"Invalid input or prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")