

from flask import Flask, request, render_template, flash, redirect, url_for, session, jsonify, send_file
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from dotenv import load_dotenv, dotenv_values
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

from flask_socketio import SocketIO, emit
import azure.cognitiveservices.speech as speechsdk

from azure.ai.translation.text import TextTranslationClient, TranslatorCredential

from azure.ai.translation.text.models import InputTextItem, DictionaryExampleTextItem
from azure.core.exceptions import HttpResponseError

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from openai import AzureOpenAI
import json

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult

from azure.ai.textanalytics import (
    TextAnalyticsClient,
    RecognizeEntitiesAction,
    RecognizeLinkedEntitiesAction,
    RecognizePiiEntitiesAction,
    ExtractKeyPhrasesAction,
    AnalyzeSentimentAction,
    MultiLabelClassifyAction
)
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures

import json
from typing import Dict
import pprint

import requests
import time


app = Flask(__name__)
socketio = SocketIO(app)


load_dotenv()

app.secret_key = 'svzsbzfngngfnfgb'  # replace with your secret key

# speech configuration
speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

def stop_cb(evt):
    print('CLOSING on {}'.format(evt))
    speech_recognizer.stop_continuous_recognition()
    speech_recognizer.session_started.disconnect_all()
    speech_recognizer.recognized.disconnect_all()
    speech_recognizer.session_stopped.disconnect_all()

speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
speech_recognizer.session_stopped.connect(lambda evt: print('\nSESSION STOPPED {}'.format(evt)))

# Emit the recognized text to the client
speech_recognizer.recognized.connect(lambda evt: socketio.emit('recognized', {'text': evt.result.text}))

speech_recognizer.canceled.connect(lambda evt: print('CANCELED: {} ({})'.format(evt.cancellation_details.reason, evt.cancellation_details.error_details)))

@socketio.on('start_recognition')
def handle_start_recognition():
    print('Say a few words\n\n')
    speech_recognizer.start_continuous_recognition()

    while True:
        time.sleep(.5)

def authenticate_client():
    key = os.environ.get('AZURE_LANGUAGE_KEY')
    endpoint = os.environ.get('AZURE_LANGUAGE_ENDPOINT')
    ta_credential = AzureKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(
            endpoint=endpoint, 
            credential=ta_credential)
    
    return text_analytics_client

def sample_recognize_custom_entities(document):
    from azure.core.credentials import AzureKeyCredential
    from azure.ai.textanalytics import (
        TextAnalyticsClient,
        RecognizeCustomEntitiesAction,
    )


    project_name = os.environ.get("CUSTOM_ENTITIES_PROJECT_NAME")
    deployment_name = os.environ.get("CUSTOM_ENTITIES_DEPLOYMENT_NAME")

    client=authenticate_client()

    document = [document]

    poller = client.begin_analyze_actions(
        document,
        actions=[
            RecognizeCustomEntitiesAction(
                project_name=project_name, deployment_name=deployment_name
            ),
        ],
    )

    document_results = poller.result()
    entities = []
    for result in document_results:
        custom_entities_result = result[0]  # first document, first result
        if not custom_entities_result.is_error:
            for entity in custom_entities_result.entities:
                entities.append({
                    'text': entity.text, 
                    'category': entity.category, 
                    'confidence_score': entity.confidence_score
                    })
                
        else:
            print(
                "...Is an error with code '{}' and message '{}'".format(
                    custom_entities_result.code, custom_entities_result.message
                )
            )
        
        return entities

@app.route('/language', methods=['GET', 'POST'])
def language():
    #average_sentiment = 3
   # star_rating = 3
    
    if request.method == 'POST':
        client = authenticate_client()
        original_text = request.form.get('text')
        print(original_text)

        # Detect language
        language_result = client.detect_language(documents=[original_text])[0]
        print(type(language_result))
        if not language_result.is_error:
            language = language_result.primary_language.name
            language_result_dict = {
                "id": language_result.id,
                "primary_language": {
                    "name": language_result.primary_language.name,
                    "iso6391_name": language_result.primary_language.iso6391_name,
                    "confidence_score": language_result.primary_language.confidence_score
                },
                "warnings": [str(warning) for warning in language_result.warnings],
                "statistics": str(language_result.statistics),
                "is_error": language_result.is_error,
                "kind": str(language_result.kind)
            }

            # Convert the dictionary to a JSON string
            language_result_json = json.dumps(language_result_dict, indent=4)
            
            
        else:
            print("No language was detected.")
            language = None

        # Sentiment analysis
        sentiment_result = client.analyze_sentiment(documents=[original_text], show_opinion_mining=True)[0] #
        
        if not sentiment_result.is_error:
            sentiment = sentiment_result.sentiment


            mined_opinions = []

            for sentence in sentiment_result.sentences:
                if sentence.mined_opinions:
                    for mined_opinion in sentence.mined_opinions:
                        opinion = [{"target": mined_opinion.target.text, "assessment": mined_opinion.assessments[0].text}]
                        mined_opinions.append(opinion)
                  
            mined_opinions=mined_opinions 
              
            print(mined_opinions)
            print(type(mined_opinions))
            sentiment_result_dict = {
                    "id": sentiment_result.id,
                    "sentiment": sentiment_result.sentiment,
                    "warnings": [str(warning) for warning in sentiment_result.warnings],
                    "statistics": str(sentiment_result.statistics),
                    "is_error": sentiment_result.is_error,
                    "kind": str(sentiment_result.kind),
                   
                    #"star_rating": star_rating,
                    "sentences": [
                        {
                            "text": sentence.text,
                            "sentiment": sentence.sentiment,
                            "confidence_scores": {
                                "positive": sentence.confidence_scores.positive,
                                "neutral": sentence.confidence_scores.neutral,
                                "negative": sentence.confidence_scores.negative,
                            },
                        }
                        for sentence in sentiment_result.sentences
                    ],
                     "mined_opinions": mined_opinions,
                }
            

            sentiment_result_json = json.dumps(sentiment_result_dict, indent=4)
       
        else:
            print("No sentiment was detected.")
          
        # Key phrases extraction
        key_phrases_result = client.extract_key_phrases(documents=[original_text])[0]

        key_phrases_result_dict = {
            "id": key_phrases_result.id,
            "key_phrases": key_phrases_result.key_phrases,
            "warnings": [str(warning) for warning in key_phrases_result.warnings],
            "statistics": str(key_phrases_result.statistics),
            "is_error": key_phrases_result.is_error,
            "kind": str(key_phrases_result.kind),
        }

        key_phrases_result_json = json.dumps(key_phrases_result_dict, indent=4)
        
        
        entities_result = client.recognize_entities(documents=[original_text])[0]

        entities_result_dict = {
            "id": entities_result.id,
            "entities": [
                {
                    "text": entity.text,
                    "category": entity.category,
                    "subcategory": entity.subcategory,
                    "confidence_score": entity.confidence_score,
                }
                for entity in entities_result.entities
            ],
            "warnings": [str(warning) for warning in entities_result.warnings],
            "statistics": str(entities_result.statistics),
            "is_error": entities_result.is_error,
            "kind": str(entities_result.kind),
        }
        print(entities_result.entities)
        entities_result_json = json.dumps(entities_result_dict, indent=4)
        

        # Entity linking
        linked_entities_result = client.recognize_linked_entities(documents=[original_text])[0]
        if not linked_entities_result.is_error:
            linked_entities_list = [(entity.name, entity.url) for entity in linked_entities_result.entities]
        else:
            print("No linked entities were detected.")
            linked_entities_list = None

        return render_template('language.html',  mined_opinions=mined_opinions, original_text=original_text, language=language, language_result=language_result_json, key_phrases=key_phrases_result.key_phrases, key_phrases_result=key_phrases_result_json, sentiment=sentiment, sentiment_result= sentiment_result_json, entities = entities_result.entities, entities_result=entities_result_json, linked_entities=linked_entities_list)

    return render_template('language.html')

@app.route('/multiple_analysis', methods=['GET', 'POST'])
def multiple_analysis():
    if request.method == 'POST':
        client = authenticate_client()
        text = request.form.get('text')
        actions = request.form.getlist('actions')

        def perform_text_analysis(text, actions):
            documents = [text]

            action_list = []
            if "RecognizeEntitiesAction" in actions:
                action_list.append(RecognizeEntitiesAction())
            if "RecognizePiiEntitiesAction" in actions:
                action_list.append(RecognizePiiEntitiesAction())
            if "ExtractKeyPhrasesAction" in actions:
                action_list.append(ExtractKeyPhrasesAction())
            if "RecognizeLinkedEntitiesAction" in actions:
                action_list.append(RecognizeLinkedEntitiesAction())
            if "AnalyzeSentimentAction" in actions:
                action_list.append(AnalyzeSentimentAction())

            poller = client.begin_analyze_actions(
                documents,
                display_name="Sample Text Analysis",
                actions=action_list,
            )

            document_results = poller.result()

            results = []
            for doc, action_results in zip(documents, document_results):
                result_dict = {"text": doc, "actions": []}
                for result in action_results:
                    if result.kind == "EntityRecognition":
                        result_dict["actions"].append({
                            "kind": "EntityRecognition",
                            "entities": [{"text": entity.text, "category": entity.category, "confidence_score": entity.confidence_score} for entity in result.entities]
                        })
                    elif result.kind == "PiiEntityRecognition":
                        result_dict["actions"].append({
                            "kind": "PiiEntityRecognition",
                            "entities": [{"text": entity.text, "category": entity.category, "confidence_score": entity.confidence_score} for entity in result.entities]
                        })
                    elif result.kind == "KeyPhraseExtraction":
                        result_dict["actions"].append({
                            "kind": "KeyPhraseExtraction",
                            "key_phrases": result.key_phrases
                        })
                    elif result.kind == "EntityLinking":
                        result_dict["actions"].append({
                            "kind": "EntityLinking",
                            "entities": [{"name": entity.name, "data_source": entity.data_source, "url": entity.url} for entity in result.entities]
                        })
                    elif result.kind == "SentimentAnalysis":
                        result_dict["actions"].append({
                            "kind": "SentimentAnalysis",
                            "sentiment": result.sentiment,
                            "confidence_scores": {"positive": result.confidence_scores.positive, "neutral": result.confidence_scores.neutral, "negative": result.confidence_scores.negative}
                        })
                results.append(result_dict)

            return results

        
        # Perform text analysis based on selected actions...
        results = perform_text_analysis(text, actions)
        return render_template('analyze.html', results=results)
    return render_template('analyze.html')

@app.route('/extractive_summary', methods=['GET', 'POST'])
def extractive_summary():
    # [START extract_summary]
    summary = None
    if request.method == 'POST':
        client=authenticate_client()
        document = [request.form.get('text')]
        summary_type = request.form.get('summary_type')

        #endpoint = "https://58101-demo.cognitiveservices.azure.com/"
        #key = "f2639704209c4a11b9fba4e033c20b24"

        #text_analytics_client = TextAnalyticsClient(
        #    endpoint=endpoint,
        #    credential=AzureKeyCredential(key),
        #)
        if summary_type == "extractive":
            poller = client.begin_extract_summary(document)
            extract_summary_results = poller.result()
            for result in extract_summary_results:
                if result.kind == "ExtractiveSummarization":
                    summary = " ".join([sentence.text for sentence in result.sentences])
        elif summary_type == "abstractive":
            poller = client.begin_abstract_summary(document)
            extract_summary_results = poller.result()
            for result in extract_summary_results:
                if result.kind == "AbstractiveSummarization":
                    summary = result.summaries[0].text

    return render_template('extractive_summary.html', summary=summary)
                
                #print("Summary extracted: \n{}".format(
                #    " ".join([sentence.text for sentence in result.sentences]))
                #)
            #elif result.is_error is True:
            #   print("...Is an error with code '{}' and message '{}'".format(
            #      result.error.code, result.error.message
            # ))
        # [END extract_summary]

@app.route('/recognize_pii', methods=['GET', 'POST'])
def recognize_pii():
      # add or remove categories as needed
    if request.method == 'POST':
        document = request.form.get('document')
        selected_category = request.form.get('category')  # get the selected category from the form
      
        client=authenticate_client()

        result = client.recognize_pii_entities([document], categories_filter=[selected_category])
        doc = result[0]
        pii_result_dict = {
            "id": doc.id,
            "entities": [
                {
                    "text": entity.text,
                    "category": entity.category,
                    "confidence_score": entity.confidence_score,
                }
                for entity in doc.entities
            ],
            "warnings": [str(warning) for warning in doc.warnings],
            "statistics": str(doc.statistics),
            "is_error": doc.is_error,
            "kind": str(doc.kind),
        }
        pii_result_json = json.dumps(pii_result_dict, indent=4)

        if not doc.is_error:
            redacted_text = doc.redacted_text
            entities = [(entity.text, entity.category) for entity in doc.entities if entity.category == "Person"]
            emails = [entity.text for entity in doc.entities if entity.category == 'Email' and entity.confidence_score >= 0.6]

            return render_template('recognize_pii.html', document=document, redacted_text=redacted_text, entities=entities, emails=emails, pii_result=pii_result_json)

    return render_template('recognize_pii.html')

@app.route('/recognize_custom_entities', methods=['GET', 'POST'])
def recognize_custom_entities():
    if request.method == 'POST':
        document = request.form.get('document')
        entities = sample_recognize_custom_entities(document)
        print(entities)

        return render_template('recognize_custom_entities.html', entities=entities)
    
    return render_template('recognize_custom_entities.html')

@app.route('/custom_qna', methods=['GET', 'POST'])
def custom_qna():
    if request.method == 'POST':
        return render_template('post_custom_qna.html')
    else:
        # Render 'custom_qna.html' for GET requests
        return render_template('custom_qna.html')

@app.route('/ailanguage', methods=['GET', 'POST'])
def ailanguage():
    return  render_template('ailanguage.html')


@app.route('/synthesize', methods=['GET', 'POST'])
def synthesize_speech():
    if request.method == 'POST':
        text = request.form.get('text')

        voice = request.form.get('voice')

        speech_config.speech_synthesis_voice_name = voice

        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

        speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()

        if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            flash(f"Speech synthesized for text [{text}]")
        elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = speech_synthesis_result.cancellation_details
            flash(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                if cancellation_details.error_details:
                    flash(f"Error details: {cancellation_details.error_details}")

        return redirect(url_for('synthesize_speech'))

    return render_template('synthesize.html')

@app.route('/aivision', methods=['GET', 'POST'])
def aivision():
    
    return render_template('error.html')

@app.route('/openai', methods=['GET', 'POST'])
def openai():
    
    return render_template('error.html')


@app.route('/recognize', methods=['GET', 'POST'])
# Your code for handling the recognition form goes here
def recognize_from_microphone():
    languages = ['en-US', 'fr-FR', 'ms-MY', 'zh-CN', 'th-TH']
    if request.method == 'POST':
        language = request.form.get('language')
        continuous = 'continuous' in request.form

        speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
        speech_config.speech_recognition_language=language

        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        

        print("Speak into your microphone.")

        
        speech_recognition_result = speech_recognizer.recognize_once_async().get()

        if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
            recognized_text = "Recognized: {}".format(speech_recognition_result.text)
        elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
            recognized_text = "No speech could be recognized: {}".format(speech_recognition_result.no_match_details)
        elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = speech_recognition_result.cancellation_details
            recognized_text = "Speech Recognition canceled: {}".format(cancellation_details.reason)
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                recognized_text += "\nError details: {}".format(cancellation_details.error_details)
        
        flash(recognized_text)

        return redirect(url_for('recognize_from_microphone'))
    else:
        
        return render_template('recognize.html',languages=languages)

@app.route('/translate', methods=['GET', 'POST'])
def translate():
    if request.method == 'POST':
        translated_text=None
        text_to_translate = request.form.get('text')
        transliterate = request.form.get('transliterate')
        target_language = request.form.get('language')
        dictionary_lookup = request.form.get('dictionary_lookup')
        dictionary_examples_lookup = request.form.get('dictionary_examples_lookup')

        translator_key = os.environ.get('TRANSLATION_KEY')
        translator_endpoint = os.environ.get('TRANSLATION_ENDPOINT')
        translator_region = os.environ.get('TRANSLATION_REGION')
        
        credential = TranslatorCredential(translator_key, translator_region)
        text_translator = TextTranslationClient(endpoint=translator_endpoint, credential=credential)
        text_item = [InputTextItem(text=text_to_translate)]

        translation_result = text_translator.translate(content=text_item, to=[target_language], profanity_action="Marked", profanity_marker="Asterisk",  include_sentence_length=True, include_alignment=True)
        if translation_result is not None and translation_result[0].translations is not None:
            translated_text = translation_result[0].translations[0].text
            translated_text_alignment=translation_result[0].translations[0].alignment['proj']

        # Perform transliteration
        if transliterate == 'yes':
            transliteration_result = text_translator.transliterate(content=text_item, from_script="Jpan", language="ja" , to_script="Latn")
            if transliteration_result is not None:
                transliterated_text = transliteration_result[0].text

        else:
            transliterated_text = None

         # Perform dictionary lookup if the user selected "yes"
        if dictionary_lookup == 'yes':
            source_language = "en"
            target_language = "es"
            input_text_elements = [ InputTextItem(text = text_to_translate) ]
            response = text_translator.lookup_dictionary_entries(content = input_text_elements, from_parameter = source_language, to = target_language)
            dictionary_entry = response[0] if response else None
            if dictionary_entry:
                dictionary_entry_text = f"For the given input {len(dictionary_entry.translations)} entries were found in the dictionary.  \n"
                for i, translation in enumerate(dictionary_entry.translations):
                    dictionary_entry_text += f" Entry {i+1}: '{translation.display_target}', confidence: {translation.confidence}.\n"
        else:
            dictionary_entry_text = None
        
        # Perform dictionary examples lookup if the user selected "yes"
        if dictionary_examples_lookup == 'yes':
            source_language = "en"
            target_language = "es"
            input_text_elements = [ DictionaryExampleTextItem(text = text_to_translate, translation = "volar") ]
            response = text_translator.lookup_dictionary_examples(content = input_text_elements, from_parameter = source_language, to = target_language)
            dictionary_entry = response[0] if response else None
            if dictionary_entry:
                dictionary_examples_text = f"For the given input {len(dictionary_entry.examples)} entries were found in the dictionary.\n"
                for i, example in enumerate(dictionary_entry.examples):
                    dictionary_examples_text += f"Example {i+1}: '{example.source_prefix}{example.source_term}{example.source_suffix}' translates to '{example.target_prefix}{example.target_term}{example.target_suffix}'.\n"
        else:
            dictionary_examples_text = None

        

        return render_template('translate.html', dictionary_examples_text=dictionary_examples_text, dictionary_entry_text=dictionary_entry_text, original_text=text_item[0]['text'], translated_text=translated_text, alignment=translated_text_alignment,transliterated_text=transliterated_text)

    return render_template('translate.html')


    
@app.route('/continuous_recognition', methods=['GET'])
def recognition():
    return render_template('continuous_recognition.html')

def document_fields_to_dict(fields):
    return {key: field.get('content', 'N/A') for key, field in fields.items()}

@app.route('/aidocumentintelligence', methods=['POST','GET'])
def aidocumentintelligence():

    endpoint = os.getenv("DOCUMENTINTELLIGENCE_ENDPOINT")
    key = os.getenv("DOCUMENTINTELLIGENCE_API_KEY")

    document_intelligence_client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    if request.method == 'POST':
        document_type = request.form.get('document_type')

        file = request.files['file']
        if 'file' not in request.files:
            return "No file part", 400
        poller = document_intelligence_client.begin_analyze_document(
            document_type, analyze_request=file.stream, locale="en-US", content_type="application/octet-stream"
        )
        receipts: AnalyzeResult = poller.result()

        results = []
        if document_type == 'prebuilt-invoice':
            if receipts.documents:
                for idx, invoice in enumerate(receipts.documents):
                    raw_result = invoice.fields
                    
                    #results.append(result)
        elif document_type == 'prebuilt-receipt':
            results = []
            if receipts.documents:
                for idx, receipt in enumerate(receipts.documents):
                    raw_result = document_fields_to_dict(receipt.fields)
                    raw_result_json = json.dumps(raw_result, indent=4)
                    results.append(raw_result_json)

        elif document_type == 'prebuilt-creditCard':
             results = []
             if receipts.documents:
                    for idx, credit_card in enumerate(receipts.documents):
                        raw_result = document_fields_to_dict(credit_card.fields)
                        
                        result = {
                            'Card Number': credit_card.fields.get('CardNumber', {}).get('content', 'N/A'),
                            'Expiration Date': credit_card.fields.get('ExpirationDate', {}).get('content', 'N/A'),
                            'Card Holder Name': credit_card.fields.get('CardHolderName', {}).get('content', 'N/A'),
                            # ... add more fields as needed ...
                        }
                        print(raw_result)
                        raw_result_json = json.dumps(raw_result, indent=4)
                        results.append(raw_result_json)
                    
                        results.append(raw_result)
                        print(results)
                    
                    
        return render_template('intelligence.html', results=raw_result_json, document_type=document_type)

    return render_template('intelligence.html')

    

if __name__ == '__main__':
    app.run(debug=True)
    socketio.run(app)
