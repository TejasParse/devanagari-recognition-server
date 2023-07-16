from pydantic import BaseModel

class InputData(BaseModel):
    variance:float
    skewness:float