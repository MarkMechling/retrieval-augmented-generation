from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the Groq API client
client = Groq(
    api_key=os.environ.get('GROQ_API_KEY'),
)

def query_llama(question, retrieved_chunks):
    # Combine retrieved chunks into a prompt
    prompt = "Based on the following information, answer the query:\n"
    for chunk in retrieved_chunks:
        prompt += f"{chunk}\n"
    prompt += f"\nQuery: {question}"

    # Define the message payload for the Llama model
    messages = [
        {"role": "user", "content": prompt},
    ]

    # Query the Llama model
    completion = client.chat.completions.create(
        model="llama3-8b-8192",  # Adjust the model name as needed
        messages=messages,
        temperature=0.7,
        max_tokens=300,
        top_p=1,
        stream=False,
        stop=None,
    )

    # Extract the response content
    response = completion.choices[0].message.content  # Use the `.content` attribute directly
    return response

# Example Usage
if __name__ == "__main__":
    question = "Under what circumstances is the transfer of data into another country allowed? "
    retrieved_chunks = [
        "This Regulation is without prejudice to international agreements concluded between the Union and third countries regulating the transfer of personal data including appropriate safeguards for the data subjects. Member States may conclude international agreements which involve the transfer of personal data to third countries or international organisations, as far as such agreements do not affect this Regulation or any other provisions of Union law and include an appropriate level of protection for the fundamental rights of the data subjects.",
        "Transfers subject to appropriate safeguards 1. In the absence of a decision pursuant to Article 45(3), a controller or processor may transfer personal data to a third country or an international organisation only if the controller or processor has provided appropriate safeguards, and on condition that enforceable data subject rights and effective legal remedies for data subjects are available.",
        "Any transfer of personal data which are undergoing processing or are intended for processing after transfer to a third country or to an international organisation shall take place only if, subject to the other provisions of this Regulation, the conditions laid down in this Chapter are complied with by the controller and processor, including for onward transfers of personal data from the third country or an international organisation to another third country or to another international organisation. All provisions in this Chapter shall be applied in order to ensure that the level of protection of natural persons guaranteed by this Regulation is not undermined."
    ]

    response = query_llama(question, retrieved_chunks)
    print("Response:", response)
