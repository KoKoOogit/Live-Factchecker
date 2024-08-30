import google.generativeai as genai
import json


api_key = ''
genai.configure(api_key=api_key)

user_input_text = input("What would you like to fact check?: ")
print("Fact checking:", user_input_text)

prompt = """
Please analyze the following text and categorize each sentence as a claim, opinion, or fact. 
Also, indicate whether each sentence is a statement or an opinion. For each sentence, provide the label first, then write out the sentence, and finally, explain the label.

Text:
"{user_input_text}"

Output format:
Please provide the output as a JSON object where each sentence is an object with the keys "label", "sentence", and "explanation".
"""

response = genai.generate_text(prompt=prompt)

generated_text = response.candidates[0]['output']

json_output = []
for entry in generated_text.split("\n\n"): 
    lines = entry.split("\n")
    if len(lines) >= 3:
        json_entry = {
            "label": lines[0].replace("Label: ", "").strip(),
            "sentence": lines[1].replace("Sentence: ", "").strip(),
            "explanation": lines[2].replace("Explanation: ", "").strip()
        }
        json_output.append(json_entry)

json_result = json.dumps(json_output, indent=4)
print(json_result)

