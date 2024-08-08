from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib

app = FastAPI()

# Paths to the saved model and vectorizer
model_path = './spam_classifier_model.pkl'
vectorizer_path = './tfidf_vectorizer.pkl'

# Load the saved model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.get("/", response_class=HTMLResponse)
async def get_index():
    return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
            <style>
                body {{
                    background: linear-gradient(to right, #2c3e50, #3498db);
                    color: #ecf0f1;
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                }}
                .container {{
                    background: rgba(0, 0, 0, 0.7);
                    padding: 30px;
                    border-radius: 12px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                    max-width: 500px;
                    width: 100%;
                }}
                h1 {{
                    text-align: center;
                    margin-bottom: 20px;
                }}
                textarea {{
                    resize: vertical;
                }}
                .btn-primary {{
                    display: block;
                    margin: 20px auto;
                    background: #3498db;
                    border: none;
                }}
                .btn-primary:hover {{
                    background: #2980b9;
                }}
                h2 {{
                    text-align: center;
                    font-weight: 600;
                    margin-top: 20px;
                }}
                .result-message {{
                    font-size: 1.25rem;
                }}
            </style>
            <title>Spam Classifier</title>
        </head>
        <body>
            <div class="container">
                <h1>Spam Classifier</h1>
                <form action="/predict" method="post">
                    <div class="form-group">
                        <label for="text">Enter Email Text:</label>
                        <textarea class="form-control" id="text" name="text" rows="4" required></textarea>
                    </div>
                    <button type="submit" class="btn btn-primary">Classify</button>
                </form>
                <div class="mt-3">
                    <h2 class="result-message" id="result-message"></h2>
                </div>
            </div>
            <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.9.1/dist/umd/popper.min.js"></script>
            <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
        </body>
        </html>
    """)

@app.post("/predict")
async def predict(text: str = Form(...)):
    text_vect = vectorizer.transform([text])
    prediction = model.predict(text_vect)[0]
    color = "#e74c3c" if prediction == 0 else "#2ecc71"
    result_message = "This email has been classified as Spam." if prediction == 0 else "This email has been classified as Ham."

    return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
            <style>
                body {{
                    background: linear-gradient(to right, #2c3e50, #3498db);
                    color: #ecf0f1;
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                }}
                .container {{
                    background: rgba(0, 0, 0, 0.7);
                    padding: 30px;
                    border-radius: 12px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                    max-width: 500px;
                    width: 100%;
                }}
                h1 {{
                    text-align: center;
                    margin-bottom: 20px;
                }}
                textarea {{
                    resize: vertical;
                }}
                .btn-primary {{
                    display: block;
                    margin: 20px auto;
                    background: #3498db;
                    border: none;
                }}
                .btn-primary:hover {{
                    background: #2980b9;
                }}
                h2 {{
                    text-align: center;
                    font-weight: 600;
                    margin-top: 20px;
                }}
                .result-message {{
                    font-size: 1.25rem;
                }}
            </style>
            <title>Spam Classifier</title>
        </head>
        <body>
            <div class="container">
                <h1>Spam Classifier</h1>
                <form action="/predict" method="post">
                    <div class="form-group">
                        <label for="text">Enter Email Text:</label>
                        <textarea class="form-control" id="text" name="text" rows="4"required></textarea>
                    </div>
                    <button type="submit" class="btn btn-primary">Classify</button>
                </form>
                <div class="mt-3">
                    <h2 class="result-message" style="color: {color};">{result_message}</h2>
                </div>
            </div>
            <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.9.1/dist/umd/popper.min.js"></script>
            <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
        </body>
        </html>
    """)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
