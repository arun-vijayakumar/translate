# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 09:54:35 2023

@author: Arun
"""
import os
import torch
from PIL import Image
import gradio as gr
import pytesseract
from transformers import pipeline,NllbTokenizer,AutoModelForSeq2SeqLM
from flores200_codes import flores_codes

import spacy
from spacy.lang.en import English
from spacy.lang.de import German
from spacy.lang.hi import Hindi
from spacy.lang.kn import Kannada
from spacy.lang.ml import Malayalam
from spacy.lang.ta import Tamil

pytesseract.pytesseract.tesseract_cmd = r'Tesseract\tesseract.exe'
tessdata_dir_config = r'--tessdata-dir "Tesseract\tessdata_best"'

model_path = r"facebook\\nllb-200-1.3B"
model_path = "facebook/nllb-200-1.3B"
model_translate = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model_tokenizer = NllbTokenizer.from_pretrained(model_path)

#https://spacy.io/usage/models#languages
dict_nllb_spacy = {"zho_Hans":"zh","eng_Latn":"en","fra_Latn":"fr", \
                   "deu_Latn":"de","hin_Deva":"hi","kan_Knda":"kn", \
                       "mal_Mlym":"ml","spa_Latn":"es","tam_Taml":"ta"}

def getOCR(ocr_image,lang_list):
    extracted_text=""
    ocr_lang = ""
    print(lang_list)
    try:
        if len(lang_list) < 1:
            extracted_text = "Select at least one language"
        if "Chinese" in lang_list:
            ocr_lang=ocr_lang+"chi_sim+"
        if "English" in lang_list:
            ocr_lang=ocr_lang+"eng+"
        if "French" in lang_list:
            ocr_lang=ocr_lang+"fra+"
        if "German" in lang_list:
            ocr_lang=ocr_lang+"deu+"
        if "Hindi" in lang_list:
            ocr_lang=ocr_lang+"hin+"
        if "Kannada" in lang_list:
            ocr_lang=ocr_lang+"kan+"
        if "Malayalam" in lang_list:
            ocr_lang=ocr_lang+"mal+"
        if "Spanish" in lang_list:
            ocr_lang=ocr_lang+"spa+" 
        if "Tamil" in lang_list:
            ocr_lang=ocr_lang+"tam+"                
        print(ocr_lang)
        extracted_text = pytesseract.image_to_string(ocr_image, lang=ocr_lang, config=tessdata_dir_config)
    except Exception as err:
        extracted_text = err    
    return extracted_text

def prepare_batch(original_text,spcy_nlp):
    batch_input = []
    output_text = ""
    token_max_length = 512
    paras = original_text.split('\n')
    for para in paras:
        if (len(para.strip()) == 0):
            output_text = output_text + '\n'
        doc = spcy_nlp(para)
        curr_len = 0
        curr_batch = ""
        for sent in doc.sents:
            added_to_batch = 0
            if(curr_len + len(sent.text) >= token_max_length):
                batch_input.append(curr_batch)
                added_to_batch = 1
                curr_len = len(sent.text)
                curr_batch = sent.text
            elif (curr_len + len(sent.text) < token_max_length):
                curr_batch = curr_batch + sent.text
                curr_len = curr_len + len(sent.text)
        if (added_to_batch == 0):
            batch_input.append(curr_batch)
    return batch_input

def prepare_batch_line_by_line(original_text,spcy_nlp):
    batch_input = []
    output_text = ""
    paras = original_text.split('\n')
    for para in paras:
        if (len(para.strip()) == 0):
            output_text = output_text + '\n'
        doc = spcy_nlp(para)
        for sent in doc.sents:
            added_to_batch = 0
            curr_batch = sent.text
            batch_input.append(curr_batch)
    return batch_input

def prepare_translate(src_lang, tar_lang, input_str):
    print("prepare_translate")
    print(dict_nllb_spacy)
    translated_text=""
    try:
        task = "translation"
        source = flores_codes[src_lang]
        target = flores_codes[tar_lang]
        spcy_lang = dict_nllb_spacy[source]
        print(f"Text: {input_str} - Source: {source} - Targert: {target} - Spacy Lang: {spcy_lang}")
        nlp = spacy.blank(spcy_lang)
        nlp.add_pipe('sentencizer')
        batches = prepare_batch(input_str,nlp)
        translation_pipeline = pipeline(task,model=model_translate,tokenizer=model_tokenizer,src_lang=source,tgt_lang=target)
        for batch in batches:
            output = translation_pipeline (batch)
            interim_text = batch + '\n' + '>> ' +  output[0]['translation_text'] + '\n' + '\n'
            translated_text = translated_text + interim_text
    except Exception as err:
        print(err)
        translated_text = err 
    
    return translated_text

def prepare_translate_line_by_line(src_lang, tar_lang, input_str):
    print("prepare_translate")
    print(dict_nllb_spacy)
    translated_text=""
    try:
        task = "translation"
        source = flores_codes[src_lang]
        target = flores_codes[tar_lang]
        spcy_lang = dict_nllb_spacy[source]
        print(f"Text: {input_str} - Source: {source} - Targert: {target} - Spacy Lang: {spcy_lang}")
        nlp = spacy.blank(spcy_lang)
        nlp.add_pipe('sentencizer')
        batches = prepare_batch_line_by_line(input_str,nlp)
        translation_pipeline = pipeline(task,model=model_translate,tokenizer=model_tokenizer,src_lang=source,tgt_lang=target)
        for batch in batches:
            output = translation_pipeline (batch)
            interim_text = batch + '\n' + '[' +  output[0]['translation_text'] + ']' + '\n'
            translated_text = translated_text + interim_text
    except Exception as err:
        print(err)
        translated_text = err 
    
    return translated_text

if __name__ == "__main__":
    gr_image = gr.Image(label="Image")
    ocr_lang_input = ["Chinese","English","French","German","Hindi","Kannada","Malayalam","Spanish","Tamil"]
    gr_langs = gr.CheckboxGroup(choices=ocr_lang_input,label="Language")
    ocr_interface = gr.Interface(getOCR, inputs=[gr_image,gr_langs],outputs=["text"])
    
    
    lang_codes = sorted(list(flores_codes.keys()))
    print(lang_codes)
    translate_inputs = [gr.inputs.Dropdown(lang_codes, default='Hindi', label='Source'),\
                        gr.inputs.Dropdown(lang_codes, default='English', label='Target'), \
                            gr.Radio(["Single Image", "Image Path"], label="Input Method", info="Sing image or images from a folder"), \
                            gr.inputs.Textbox(lines=5, label="Input text")]
    
    translate_interface = gr.Interface(prepare_translate_line_by_line, inputs=translate_inputs,outputs=[gr.inputs.Textbox(lines=5, label="Translated text")])
    
    demo = gr.TabbedInterface([ocr_interface, translate_interface], ["OCR", "Translate"])
    
    gr.close_all()
    demo.launch(server_port=9512)
