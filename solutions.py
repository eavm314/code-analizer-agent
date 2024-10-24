from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Example problem model
class ExampleProblem(BaseModel):
    data: list

@app.post("/mean")
def calculate_mean(problem: ExampleProblem):
    """
    Calculate the mean of a list of numbers.
    """
    array = np.array(problem.data)
    mean_value = np.mean(array)
    return {"mean": mean_value}

@app.post("/sum")
def calculate_sum(problem: ExampleProblem):
    """
    Calculate the sum of a list of numbers.
    """
    array = np.array(problem.data)
    sum_value = np.sum(array)
    return {"sum": sum_value}

@app.post("/dot_product")
def calculate_dot_product(problem: ExampleProblem):
    """
    Calculate the dot product of two lists of numbers.
    """
    if len(problem.data) != 2:
        return {"error": "Please provide exactly two lists for dot product."}
    
    array_a = np.array(problem.data[0])
    array_b = np.array(problem.data[1])
    dot_product = np.dot(array_a, array_b)
    return {"dot_product": dot_product}

@app.post("/matrix_multiply")
def matrix_multiply(problem: ExampleProblem):
    """
    Multiply two matrices.
    """
    if len(problem.data) != 2:
        return {"error": "Please provide exactly two matrices for multiplication."}
    
    matrix_a = np.array(problem.data[0])
    matrix_b = np.array(problem.data[1])
    product = np.dot(matrix_a, matrix_b)
    return {"product": product.tolist()}

@app.post("/variance")
def calculate_variance(problem: ExampleProblem):
    """
    Calculate the variance of a list of numbers.
    """
    array = np.array(problem.data)
    variance_value = np.var(array)
    return {"variance": variance_value}

# To run the app, use the command: uvicorn filename:app --reload