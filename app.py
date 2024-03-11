

from flask import Flask, request, render_template, flash, redirect, url_for, session, jsonify
import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from azure.ai.translation.text import TextTranslationClient, TranslatorCredential

from azure.ai.translation.text.models import InputTextItem
from azure.core.exceptions import HttpResponseError

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from openai import AzureOpenAI
import json

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

from azure.ai.textanalytics import (
    TextAnalyticsClient,
    RecognizeEntitiesAction,
    RecognizeLinkedEntitiesAction,
    RecognizePiiEntitiesAction,
    ExtractKeyPhrasesAction,
    AnalyzeSentimentAction,
    MultiLabelClassifyAction
)

import json




#credential = DefaultAzureCredential()

#key_vault_name = 'pr801demo'
#key_vault_uri = f"https://{key_vault_name}.vault.azure.net/"

#client = SecretClient(vault_url=key_vault_uri, credential=credential)

#speech_key = client.get_secret('speech-key')

#print(f"Speech Key: {speech_key.value}")



app = Flask(__name__)

load_dotenv()

app.secret_key = 'svzsbzfngngfnfgb'  # replace with your secret key

speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

def authenticate_client():
    key = os.environ.get('TEXT_ANALYTICS_KEY')
    endpoint = os.environ.get('TEXT_ANALYTICS_ENDPOINT')
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

    endpoint = os.environ.get("AZURE_LANGUAGE_ENDPOINT_CUSTOM_ENTITIES")
    key = os.environ.get("AZURE_LANGUAGE_KEY_CUSTOM_ENTITIES")
    project_name = os.environ.get("CUSTOM_ENTITIES_PROJECT_NAME")
    deployment_name = os.environ.get("CUSTOM_ENTITIES_DEPLOYMENT_NAME")

    text_analytics_client = TextAnalyticsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )

    document = [document]

    poller = text_analytics_client.begin_analyze_actions(
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
        sentiment_result = client.analyze_sentiment(documents=[original_text])[0]
        
        if not sentiment_result.is_error:
            sentiment = sentiment_result.sentiment

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

        return render_template('language.html', language=language, language_result=language_result_json, key_phrases=key_phrases_result.key_phrases, key_phrases_result=key_phrases_result_json, sentiment=sentiment, sentiment_result= sentiment_result_json, entities = entities_result.entities, entities_result=entities_result_json, linked_entities=linked_entities_list)

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

@app.route('/week1', methods=['GET', 'POST'])
def week1():
    if request.method == 'POST':
        prompt = request.form.get('prompt')

        client = AzureOpenAI(
            api_version="2023-12-01-preview",
            azure_endpoint="https://dalle3-demo0416.openai.azure.com/openai/deployments/Dalle3/images/generations?api-version=2023-06-01-preview",
            api_key="890c9c0619cb40489e0a5d331da7ceef",
        )

        result = client.images.generate(
            model="dall-e-3", # the name of your DALL-E 3 deployment
            prompt=prompt,
            n=1
        )

        image_url = json.loads(result.model_dump_json())['data'][0]['url']

        return render_template('week1.html', image_url=image_url)

    return render_template('week1.html')

@app.route('/week2', methods=['GET', 'POST'])
def week2():
    return  render_template('week2.html')


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

@app.route('/recognize', methods=['GET', 'POST'])
# Your code for handling the recognition form goes here
def recognize_from_microphone():
    languages = ['en-US', 'fr-FR', 'ms-MY', 'zh-CN']
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

        target_language = request.form.get('language')

        translator_key = os.environ.get('TRANSLATION_KEY')
        translator_endpoint = os.environ.get('TRANSLATION_ENDPOINT')
        translator_region = os.environ.get('TRANSLATION_REGION')
        
        credential = TranslatorCredential(translator_key, translator_region)
        text_translator = TextTranslationClient(endpoint=translator_endpoint, credential=credential)
        text_item = [InputTextItem(text=text_to_translate)]

        translation_result = text_translator.translate(content=text_item, to=[target_language], profanity_action="Marked", profanity_marker="Asterisk",  include_sentence_length=True, include_alignment=True)

        translated_text = translation_result[0].translations[0].text
        translated_text_alignment=translation_result[0].translations[0].alignment['proj']

        return render_template('translate.html', original_text=text_item[0]['text'], translated_text=translated_text, alignment=translated_text_alignment)

    return render_template('translate.html')

@app.route('/extractive_summary', methods=['GET', 'POST'])
def extractive_summary():
    # [START extract_summary]
    summary = None
    if request.method == 'POST':
        document = [request.form.get('text')]
        summary_type = request.form.get('summary_type')

        endpoint = "https://58101-demo.cognitiveservices.azure.com/"
        key = "f2639704209c4a11b9fba4e033c20b24"

        text_analytics_client = TextAnalyticsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key),
        )
        if summary_type == "extractive":
            poller = text_analytics_client.begin_extract_summary(document)
            extract_summary_results = poller.result()
            for result in extract_summary_results:
                if result.kind == "ExtractiveSummarization":
                    summary = " ".join([sentence.text for sentence in result.sentences])
        elif summary_type == "abstractive":
            poller = text_analytics_client.begin_abstract_summary(document)
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
      
        endpoint = os.environ["AZURE_LANGUAGE_ENDPOINT"]
        key = os.environ["AZURE_LANGUAGE_KEY"]

        text_analytics_client = TextAnalyticsClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )

        result = text_analytics_client.recognize_pii_entities([document], categories_filter=[selected_category])
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

if __name__ == '__main__':
    app.run(debug=True)