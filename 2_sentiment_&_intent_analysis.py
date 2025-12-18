
!pip install -q transformers torch spacy

!python -m spacy download en_core_web_sm

import json
import torch
import spacy
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
nlp = spacy.load("en_core_web_sm")

# ---------- Sentiment (DistilBERT) ----------
sentiment_tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

# ---------- Intent (BART MNLI) ----------
intent_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if torch.cuda.is_available() else -1
)

conversation_text = """
> **Physician:** *Good morning, Ms. Jones. How are you feeling today?*
>
>
> **Patient:** *Good morning, doctor. I’m doing better, but I still have some discomfort now and then.*
>
> **Physician:** *I understand you were in a car accident last September. Can you walk me through what happened?*
>
> **Patient:** *Yes, it was on September 1st, around 12:30 in the afternoon. I was driving from Cheadle Hulme to Manchester when I had to stop in traffic. Out of nowhere, another car hit me from behind, which pushed my car into the one in front.*
>
> **Physician:** *That sounds like a strong impact. Were you wearing your seatbelt?*
>
> **Patient:** *Yes, I always do.*
>
> **Physician:** *What did you feel immediately after the accident?*
>
> **Patient:** *At first, I was just shocked. But then I realized I had hit my head on the steering wheel, and I could feel pain in my neck and back almost right away.*
>
> **Physician:** *Did you seek medical attention at that time?*
>
> **Patient:** *Yes, I went to Moss Bank Accident and Emergency. They checked me over and said it was a whiplash injury, but they didn’t do any X-rays. They just gave me some advice and sent me home.*
>
> **Physician:** *How did things progress after that?*
>
> **Patient:** *The first four weeks were rough. My neck and back pain were really bad—I had trouble sleeping and had to take painkillers regularly. It started improving after that, but I had to go through ten sessions of physiotherapy to help with the stiffness and discomfort.*
>
> **Physician:** *That makes sense. Are you still experiencing pain now?*
>
> **Patient:** *It’s not constant, but I do get occasional backaches. It’s nothing like before, though.*
>
> **Physician:** *That’s good to hear. Have you noticed any other effects, like anxiety while driving or difficulty concentrating?*
>
> **Patient:** *No, nothing like that. I don’t feel nervous driving, and I haven’t had any emotional issues from the accident.*
>
> **Physician:** *And how has this impacted your daily life? Work, hobbies, anything like that?*
>
> **Patient:** *I had to take a week off work, but after that, I was back to my usual routine. It hasn’t really stopped me from doing anything.*
>
> **Physician:** *That’s encouraging. Let’s go ahead and do a physical examination to check your mobility and any lingering pain.*
>
> [**Physical Examination Conducted**]
>
> **Physician:** *Everything looks good. Your neck and back have a full range of movement, and there’s no tenderness or signs of lasting damage. Your muscles and spine seem to be in good condition.*
>
> **Patient:** *That’s a relief!*
>
> **Physician:** *Yes, your recovery so far has been quite positive. Given your progress, I’d expect you to make a full recovery within six months of the accident. There are no signs of long-term damage or degeneration.*
>
> **Patient:** *That’s great to hear. So, I don’t need to worry about this affecting me in the future?*
>
> **Physician:** *That’s right. I don’t foresee any long-term impact on your work or daily life. If anything changes or you experience worsening symptoms, you can always come back for a follow-up. But at this point, you’re on track for a full recovery.*
>
> **Patient:** *Thank you, doctor. I appreciate it.*
>
> **Physician:** *You’re very welcome, Ms. Jones. Take care, and don’t hesitate to reach out if you need anything.*
>
"""

# sentiment_intent_pipeline.py
import json
import torch
import spacy

nlp = spacy.load("en_core_web_sm")

def sentiment_intent_pipeline(conversation_text, max_words=400):

    #  Extract Patient Utterances

    doc = nlp(conversation_text)
    patient_text = " ".join(
        sent.text.strip()
        for sent in doc.sents
        if not sent.text.lower().startswith(("doctor:", "physician:"))
    )

    if not patient_text:
        return {"Sentiment": "Neutral", "Intent": "Unknown"}

    #  Chunk Text

    words = patient_text.split()
    chunks = [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]


    #  BERT Sentiment

    probs_list = []

    for chunk in chunks:
        inputs = sentiment_tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = sentiment_model(**inputs)

        probs_list.append(torch.softmax(outputs.logits, dim=1).squeeze())

    avg_prob = torch.mean(torch.stack(probs_list), dim=0)

    negative_score = avg_prob[0].item()
    positive_score = avg_prob[1].item()

    if negative_score > positive_score and negative_score >= 0.6:
        sentiment = "Anxious"
    elif positive_score > negative_score and positive_score >= 0.6:
        sentiment = "Reassured"
    else:
        sentiment = "Neutral"

# Intent Detection


    candidate_intents = [
        "Reporting symptoms",
        "Seeking reassurance",
        "Expressing concern",
        "Confirming recovery",
        "General inquiry"
    ]

    intent = intent_classifier(
        patient_text,
        candidate_labels=candidate_intents
    )["labels"][0]

  # output
    return {
        "Sentiment": sentiment,
        "Intent": intent
    }



output = sentiment_intent_pipeline(conversation_text)

with open("sentiment_intent_output.json", "w") as f:
    json.dump(output, f, indent=2)
