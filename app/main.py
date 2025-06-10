from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from app.utils import generate_weather_data, process_uploaded_csv, feature_engineering
from app.model import predict_pedocs
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os
import time
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    start = time.time()
    yield 
    print(f"App startup took {time.time() - start:.2f} seconds")


boot_time = time.time()


app = FastAPI(lifespan=lifespan)

# rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8080").split(",")
allowed_origins.append("https://delicate-frangollo-3f4686.netlify.app/")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
@limiter.limit("5/minute") 
async def get_predictions(request: Request, file: UploadFile = File(...)):

    df = await process_uploaded_csv(file)

    # set dates to fetch weather
    start_date = df['timestamp'].iloc[0].strftime('%Y-%m-%d')
    end_date = df['timestamp'].iloc[-1].strftime('%Y-%m-%d')

    weather_df = await generate_weather_data(start_date, end_date)


    # merge the weather data with the uploaded csv
    df_merged = pd.merge(df, weather_df, on='timestamp', how='right')


    # feature engineer the uploaded csv
    df_final = feature_engineering(df_merged)

    result = predict_pedocs(df, df_final)
    result_df = result.to_dataframe(copy=True, backend='pandas')
    response_data = format_predictions(result_df)
    return JSONResponse(content=response_data)

def format_predictions(df: pd.DataFrame):
    df = df.reset_index()
    df['timestamp'] = df['timestamp'].astype(str)
    df = df.rename(columns={'pedocs_score': 'score'}) 
    return df.to_dict(orient='records')

@app.get("/health")
def health_check():
    return {"status": "ok"}
