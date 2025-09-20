import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import openai
from openai import OpenAI
import re
# Test with an example input text
input_text = "input text"
#################################################### Fake Detection #################################################
# Define the paths where the model and tokenizer are saved
model_save_path = "/path/to/model"
tokenizer_save_path = "/path/to/tokenizer"
theme_description_path = "/path/to/themes_descriptions.txt"
theme_qa_list_path = "/path/to/themes_qa_list.txt"

theme_responses = {
    1: "Home remedies for COVID-19 prevention and treatment are often based on misinformation. While they may promote general health, no scientific evidence supports their effectiveness in preventing or curing COVID-19.",
    2: "Misinformation about COVID-19 deaths, statistics, and their relation to vaccination often involves data misrepresentation, misinterpretation of health reports, or fabricated claims. Such falsehoods can cause confusion, erode trust in public health institutions, and foster skepticism about the impact of COVID-19 and the effectiveness of vaccination. Reliable data is readily available from credible sources such as the World Health Organization (WHO) and the Centers for Disease Control and Prevention (CDC).",
    3: "Conspiracy theories and misinformation related to COVID-19 often rely on unverified claims, distorted facts, or manipulated content. These falsehoods exploit fear, uncertainty, and distrust to spread misleading narratives about the pandemic's origins, vaccines, and public health measures. Common conspiracy themes include claims of the pandemic being planned, unfounded accusations against prominent figures like Bill Gates, misinformation linking 5G technology to COVID-19, and fabricated stories about vaccines and immunity.",
    4: "Misinformation about COVID-19 vaccines, including false claims about safety, efficacy, and side effects, has fueled skepticism and vaccine hesitancy worldwide. These falsehoods undermine public trust in vaccines, which credible sources like the WHO and CDC confirm are safe, effective, and critical for preventing severe illness and death from COVID-19.",
    5: "Misinformation about COVID-19 testing often targets the accuracy of PCR tests and promotes unscientific self-check methods like holding one’s breath. PCR tests are highly accurate but may detect non-infectious viral fragments, which can be misunderstood. In reality, PCR tests are highly sensitive and accurate for detecting COVID-19 when used appropriately, though they may detect non-infectious viral fragments in some cases.",
    6: "Misinformation about treatments for COVID-19, including the use of Ivermectin, hydroxychloroquine, and unproven supplements like Quermax, often exaggerate the effectiveness of these treatments or misrepresent scientific studies to support their use over proven measures like vaccines. While some drugs, such as Ivermectin and hydroxychloroquine, were initially studied for potential COVID-19 treatment, clinical trials have not provided sufficient evidence to confirm their safety or efficacy for this purpose. Similarly, supplements and alternative remedies lack the rigorous scientific validation required to prove their effectiveness against COVID-19.",
    7: "Misinformation about government and political responses to COVID-19 involves distorted facts, fabricated claims, or misrepresentation of policies and actions. These falsehoods range from exaggerated accomplishments to unfounded accusations, including misreporting financial expenditures, misattributing statements to political leaders, or fabricating praise from international organizations. Governments worldwide have implemented a wide range of measures to combat COVID-19, including quarantine protocols, vaccination drives, economic relief packages, and public health campaigns. However, the interpretation and communication of these measures can vary, leading to the spread of misleading narratives. To counter this, it is crucial to verify information from credible sources, such as official government announcements, reputable media outlets, and organizations like the World Health Organization (WHO).",
    8: "Misinformation about lockdowns and COVID-19 restrictions often includes fabricated or misrepresented events, videos, or claims. False narratives range from altered timelines of events to conspiracies about government actions, protests, and public compliance. Lockdowns and restrictions were implemented globally to curb the spread of COVID-19, with varying levels of enforcement and public response. To combat misinformation, it is critical to verify claims through official government announcements, trusted media sources, and independent fact-checking organizations.",
    9: "Misunderstandings about how COVID-19 spreads and survives in various environments have led to numerous false claims and fears. These include unverified assertions about the virus being linked to specific foods like chickens or cabbages, its survival on surfaces like shoes or roads, and exaggerated theories about airborne transmission and environmental factors like temperature. Scientific evidence indicates that while COVID-19 can survive on surfaces for varying durations, its primary transmission occurs through respiratory droplets and, in some cases, aerosols. Claims about the virus thriving on specific foods or being destroyed at particular temperatures lack scientific support.",
    10: "Misleading or false claims about defensive and protective measures against COVID-19 have contributed to confusion and unsafe practices. These include exaggerated claims about the use of chlorine dioxide, UV rays, and alcohol-based products, as well as debates over the effectiveness of masks and sanitizers. While certain measures, such as wearing masks and using hand sanitizers, are scientifically proven to reduce the spread of the virus, other claims, like sunlight killing COVID-19 or chlorine dioxide being safe for disinfection, lack credible evidence or are outright false. Relying on guidance from reputable health organizations like the World Health Organization (WHO) and the Centers for Disease Control and Prevention (CDC) is essential for understanding which protective measures are effective and safe.",
    11: "Misinformation about international incidents and responses to COVID-19 involves exaggerated or fabricated claims about government actions, healthcare capacity, or emotional reactions of leaders. Examples include false reports of rapid hospital construction, misattributed emotional responses by world leaders, or unverified claims about the treatment of COVID-19 victims. Such false narratives misrepresent the global efforts to combat the pandemic and can erode trust in public health measures and international cooperation. Misleading claims, such as countries discarding victims’ bodies at sea or building thousands of hospital beds in days, often stem from misinterpreted images or old videos unrelated to COVID-19. To combat misinformation, it is essential to rely on verified data from credible sources like national health authorities, the World Health Organization (WHO), and independent fact-checking organizations.",
    12: "Misinformation about COVID-19 spreads rapidly through social media platforms and messaging apps like WhatsApp and Facebook. False claims often exploit fear and uncertainty, such as fabricated government policies, misleading features of apps, or unfounded accusations about media manipulation. To counter this, it is crucial to verify claims through reliable fact-checking organizations and official communications.",
    13: "These claims lack credible scientific evidence or are distortions of real information, such as the exaggerated impact of products, religious practices, or fabricated connections to bioweapons and patented viruses. While some of these topics may seem minor or humorous, they can still misinform the public and detract from accurate understandings of the pandemic. Critical thinking, fact-checking, and reliance on credible scientific and official sources are vital to combatting these misinformation."
}


covid_misinformation_groups = {
    1: {
        "title": "Home Remedies and Misconceptions About COVID-19 Prevention and Treatment",
    },
    2: {
        "title": "COVID-19 Deaths and Statistics",
    },
    3: {
        "title": "Conspiracy Theories and Misinformation",
    },
    4: {
        "title": "Vaccine Controversies and Misinformation",
    },
    5: {
        "title": "COVID-19 Testing and Accuracy",
    },
    6: {
        "title": "Ivermectin and Other Treatments",
    },
    7: {
        "title": "Government and Political Responses",
    },
    8: {
        "title": "Lockdown and Restrictions",
    },
    9: {
        "title": "Transmission Paths and Environmental Factors",
    },
    10: {
        "title": "Defensive and Protective Measures",
    },
    11: {
        "title": "International Incidents and Responses",
    },
    12: {
        "title": "Media, Communication, and Spread of Information",
    },
    13: {
        "title": "Miscellaneous and Minor Topics",
    }
}

os.environ['OPENAI_API_KEY'] = 'put here your open ai key'
client = OpenAI()

with open(theme_description_path, 'r', encoding='utf-8') as file:
    theme_description = file.read()

with open(theme_qa_list_path, 'r', encoding='utf-8') as file:
    QA_List = file.read()
# Check if GPU is available and select the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the saved model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
model.to(device)  # Move model to GPU if available
tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)

# Define a function to get predictions
def predict(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)  # Move inputs to GPU if available
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1)
    return prediction.item()



# Predict the label for the input text
predicted_label = predict(input_text)
label_map_reverse = {0: 'fake', 1: 'real'}
predicted_label_text = label_map_reverse[predicted_label]
# Print the result
print(f"Input text: \"{input_text}\"")
print(f"Predicted label: {predicted_label_text}")

# Function to extract the theme number from the message
def extract_theme_number(message):
    match = re.search(r'\b(1[0-3]|[1-9])\b', message)
    if match:
        return int(match.group(0))
    return None

if predicted_label == 0:
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "The following is the description of topic themes related to COVID-19 misinformation. Find the closest theme for the given text. Answer in consistent style."
            },
            {
                "role": "user",
                "content": theme_description
            },
            {
                "role": "assistant",
                "content": QA_List
            },
            {
                "role": "user",
                "content": input_text
            }
        ]
    )

    # Extract the model's predicted theme number
    model_response = completion.choices[0].message.content
    predicted_theme = extract_theme_number(model_response)
    print("predicted_theme=",predicted_theme)
    print("model_response=", model_response)
    if predicted_theme and predicted_theme in covid_misinformation_groups:
        theme_title = covid_misinformation_groups[predicted_theme]["title"]
        theme_response = theme_responses[predicted_theme]

        # Print the theme title and theme response
        print(f"Predicted Theme Number: {predicted_theme}")
        print(f"Theme Title: {theme_title}")
        print(f"Theme Response: {theme_response}")
    else:
        print("Could not determine a valid theme from the model response.")
else:
    print("The input text was classified as 'real', no theme will be predicted.")