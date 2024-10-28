"""Build a simple LLM application to generate detailed product descriptions from specifications and categories"""

import os
import groq
from dotenv import load_dotenv
load_dotenv()

# Set GRO_API_KEY = "your api key" in the .env file, then load it below
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = groq.Groq(api_key=GROQ_API_KEY)

sys_prompt = """You are a professional product description generator. \
Your task is to create engaging, informative, and detailed product descriptions based on the provided product specifications and category. \
Ensure that the description highlights the key features, benefits, and unique selling points of the product, tailored to its category."""

models = [
    "llama-3.1-405b-reasoning",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768"
]


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

    answer = response.choices[0].message.content

    return answer


def get_product_specifications():
    print("Enter the product specifications. Type 'END' on a new line to finish.")
    specs = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        specs.append(line)
    return "\n".join(specs)


def get_product_category():
    print("Enter the product category (e.g., Electronics, Home & Kitchen, Apparel, etc.):")
    category = input().strip()
    return category


if __name__ == "__main__":
    model = models[2]
    
    # Get user inputs
    print("=== Product Description Generator ===\n")
    
    category = get_product_category()
    print("\nPlease enter the product specifications:")
    specifications = get_product_specifications()
    
    query = f"""Category: {category}
Product Specifications:
{specifications}

Generate a detailed product description based on the above specifications and category."""
    
    temperature = 0.7  # You can adjust this value as needed
    
    response = generate(model, query, temperature)
    print("\n=== Generated Product Description ===\n")
    print(response)
