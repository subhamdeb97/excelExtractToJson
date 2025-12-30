import os
import io
import pandas as pd
from groq import Groq
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API KEY (DO NOT hardcode in prod)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing")

client = Groq(api_key=GROQ_API_KEY)

@app.post("/analyze-excel")
async def analyze_excel(
    query: str = Form(...),
    file: UploadFile = File(...)
):
    # 1. Load Excel
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        df = df.where(pd.notnull(df), None)  # JSON-safe
        if len(df) > 50:
            df = df.head(50)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Excel file: {str(e)}")

    # 2. Schema for LLM
    schema = df.dtypes.astype(str).to_dict()

    system_prompt = f"""
You are a pandas expert.

DataFrame name: df
Columns with dtypes:
{schema}

User query:
{query}

Rules:
- Return ONLY a single Python expression
- The expression MUST use df
- No assignments
- No imports
- No print
- No markdown
"""

    # 3. Call Groq
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.0,
        )

        generated_code = (
            completion.choices[0].message.content
            .strip()
            .replace("`", "")
            .replace("python", "")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API Error: {str(e)}")

    # 4. SAFE-ish execution (same scenario, controlled)
    try:
        allowed_globals = {
            "__builtins__": {},
        }
        allowed_locals = {
            "df": df,
        }

        result = eval(generated_code, allowed_globals, allowed_locals)

        # Normalize output to JSON
        if isinstance(result, pd.DataFrame):
            result = result.to_dict(orient="records")
        elif isinstance(result, pd.Series):
            result = result.to_list()

        return {
            "status": "success",
            "query": query,
            "generated_code": generated_code,
            "result": result
        }

    except Exception as e:
        return {
            "status": "error",
            "generated_code": generated_code,
            "message": str(e)
        }
