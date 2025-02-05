import os
import groq

# Set up Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
groq_client = groq.Client(api_key=GROQ_API_KEY)

def groq_generate(prompt: str) -> str:
    """Send the prompt to Groq's API and get a response from Mixtral."""
    try:
        response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=4096,
            top_p=0.9
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"