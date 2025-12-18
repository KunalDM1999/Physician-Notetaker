# Physician-Notetaker

 Overview

This project implements an AI-powered Physician Notetaker that processes doctor–patient conversations to:

* Extract key medical information
* Analyze patient sentiment and intent
* Generate a structured SOAP clinical note

The system combines rule-based NLP and transformer models to produce clinically readable, structured JSON outputs.

Repository Components

Medical NLP Summarization

* Extracts Symptoms, Diagnosis, Treatment, Prognosis
* Outputs a structured medical summary in JSON

Sentiment & Intent Analysis

* Classifies patient sentiment (Anxious, Neutral, Reassured)
* Detects patient intent (e.g., Reporting symptoms, Seeking reassurance)

SOAP Note Generation (Bonus)

* Converts transcripts into structured SOAP notes
* Sections: Subjective, Objective, Assessment, Plan

Models & Tools

* spaCy (NLP preprocessing)
* DistilBERT (sentiment)
* BART MNLI (intent detection)
* MedAlpaca (clinical text generation)


Setup Instructions

1. Environment Requirements

   * Python 3.8+
   * GPU recommended (for clinical language model)

2. Install Dependencies
   
     pip install transformers accelerate sentencepiece torch spacy

     python -m spacy download en_core_web_sm

3. Run the Pipelines

   * Execute each .py file or notebook independently
   * Provide a conversation transcript as input
   * Outputs are generated in JSON format
