from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import groq
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize the Groq client
groq_client = groq.Groq(api_key=GROQ_API_KEY)

# Define system prompt with instructions for rich formatting
sys_prompt = """You are a professional product description generator. 
Your task is to create engaging, informative, and detailed product descriptions based on the provided details, including specifications, category, subcategory, and price. 
The output should include rich formatting with bold headings, bullet points for key features, and a consistent font size.
Include sections such as Product Overview, Key Features, and Price Information, with bold tags for important text. Use HTML tags  and make it well formatted. with resaonable padding where needed"""

models = [
    "llama-3.1-405b-reasoning",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768"
]

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4001", "https://corislo.vercel.app"],  # or ["*"] to allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class DescriptionRequest(BaseModel):
    category: str
    subcategory: str
    prodPrice: float
    deliveryMethod: list
    specification: dict  # This can contain various fields like RAM, screen size, etc.

def generate(model, query, temperature=0):
    response = groq_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": query},
        ],
        response_format={"type": "text"},
        temperature=temperature
    )
    return response.choices[0].message.content

@app.post("/generate-description/")
async def generate_description(request: DescriptionRequest):
    model = models[2]  # Choose model based on preference
    
    # Construct the query with more detailed formatting instructions
    query = f"""Category: {request.category}
Subcategory: {request.subcategory}
Price: ₦{request.prodPrice}
deliveryMethods: {f"{request.deliveryMethod}"}
Specifications: {', '.join([f"{k}: {v}" for k, v in request.specification.items()])}

Generate a detailed product description that includes sections such as:
1. **Product Overview** - Brief introduction to the product and its category.
2. **Key Features** - Highlight main specifications as bullet points with appropriate tags for bold text.
3. **Price Information** - Include a statement on price and value for money.

Ensure you suggest product name and the description uses bold tags, headings, consistent font sizes and other formatting method make sure they are well formatted for clear sectioning."""
    
    try:
        description = generate(model, query, temperature=0.7)
        return {"description": description}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating description: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
