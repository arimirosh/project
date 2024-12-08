import pandas as pd
import logging
import uvicorn
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import dotenv_values
import numpy as np

class DepressionData(BaseModel):
    Gender: str
    Age: float
    City: str
    AcademicPressure: int
    WorkPressure: int
    CGPA: float
    SleepDuration: str
    DietaryHabits: str
    Degree: str
    HaveYouEverHadSuicidalThoughts: str
    WorkStudyHours: float
    FinancialStress: float
    FamilyhistoryOfMentalIllness: str
    Depression: int
    OverallPressure: float
    OverallSatisfaction: float

class DepressionDataBackend:
    def __init__(self):
        self.setup_logging()
        self.app = FastAPI()
        self.setup_routes()
        self.PATH = dotenv_values('.env').get('DATA_PATH', 'data.csv')  # Default path if not in .env

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def setup_routes(self):
        cors_origins = ["*"]
        cors_methods = ["*"]
        cors_headers = ["*"]
        cors_credentials = True

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=cors_credentials,
            allow_methods=cors_methods,
            allow_headers=cors_headers,
        )

        @self.app.get("/")
        async def home():
            return {"message": "Welcome to the Students Depression Analysis App! Data API"}

        @self.app.post("/api/submit/")
        async def submit_data(data: DepressionData):
            return await self.submit_data(data)

        @self.app.get("/api/clean_data/")
        async def clean_data():
            return await self.clean_data()

        @self.app.get("/api/data/")
        async def get_combined_data():
            return await self.get_combined_data()

        @self.app.get("/api/revert_to_initial/")
        async def revert_to_initial():
            return await self.revert_to_initial_data()

    async def load_data(self):
        if not os.path.exists(self.PATH):
            raise FileNotFoundError("Data file not found.")
        return pd.read_csv(self.PATH)

    async def save_data(self, df: pd.DataFrame):
        df.to_csv(self.PATH, index=False)

    async def clean_data(self):
        try:
            data_frame = await self.load_data()
            initial_count = data_frame.shape[0]

            data_frame = data_frame.dropna()
            self.logger.info("NaN values removed.")

            for col in data_frame.columns:
                if data_frame[col].dtype == "object":
                    data_frame[col] = data_frame[col].replace(
                        {r'[kK]\+?$': 'e3', r'[mM]\+?$': 'e6'}, regex=True
                    ).apply(pd.to_numeric, errors='ignore')
                    self.logger.info(f"Type conversion attempted for {col}.")

            numeric_columns = data_frame.select_dtypes(include=[np.number]).columns
            data_frame = data_frame[data_frame[numeric_columns].notna().all(axis=1)]

            statistics = data_frame.describe(include='all').to_dict()
            self.logger.info("Statistics generated.")

            await self.save_data(data_frame)
            final_count = data_frame.shape[0]

            return {
                "message": "Data cleaning process completed successfully.",
                "initial_count": initial_count,
                "final_count": final_count,
                "statistics": statistics
            }
        except Exception as error:
            self.logger.error(f"Data cleaning failed: {error}")
            return {"error": str(error)}

    async def get_combined_data(self):
        try:
            df = await self.load_data()
            self.logger.info("Data loaded successfully.")
            df = df.dropna(how="all")
            return df.to_dict(orient="records")
        except Exception as e:
            self.logger.error(f"Error while loading data: {e}")
            return {"error": str(e)}

    async def revert_to_initial_data(self):
        try:
            df = await self.load_data()
            initial_df = await self.load_initial_data()  # Replace with actual loading logic

            df = df[df.apply(tuple, axis=1).isin(initial_df.apply(tuple, axis=1))]

            await self.save_data(df)
            return {"message": "Data reverted to initial state.", "data": df.head().to_dict()}
        except Exception as e:
            self.logger.error(f"Error while reverting data: {e}")
            return {"error": str(e)}

    async def submit_data(self, data: DepressionData):
        try:
            df = await self.load_data()
            new_entry = pd.DataFrame([data.dict()])
            df = pd.concat([df, new_entry], ignore_index=True)
            await self.save_data(df)
            return {"message": "New data added successfully.", "data": new_entry.to_dict(orient="records")}
        except Exception as e:
            self.logger.error(f"Error while adding new data: {e}")
            return {"error": str(e)}

def create_app():
    backend = DepressionDataBackend()
    return backend.app

if __name__ == "__main__":
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
