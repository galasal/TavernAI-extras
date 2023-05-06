from functools import wraps
from flask import Flask, jsonify, request, render_template_string, abort
from flask_cors import CORS
import markdown
import argparse
from transformers import AutoTokenizer, AutoProcessor, pipeline
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import BlipForConditionalGeneration, GPT2Tokenizer
import unicodedata
import torch
import time
from PIL import Image
import base64
from io import BytesIO
from random import randint
from EmotionClassifier import EmotionClassifier
from Summarizer import Summarizer
import webuiapi
from colorama import Fore, Style, init as colorama_init

class Captioner:
    def __init__(self, captioning_model, device, torch_dtype):
        self.torch_dtype = torch_dtype
        self.main_device = device
        self.standby_device = torch.device("cpu")
        self.captioning_processor = AutoProcessor.from_pretrained(captioning_model)
        if 'blip' in captioning_model:
            self.captioning_transformer = BlipForConditionalGeneration.from_pretrained(captioning_model, torch_dtype=torch_dtype)
        else:
            self.captioning_transformer = AutoModelForCausalLM.from_pretrained(captioning_model, torch_dtype=torch_dtype)
        self.captioning_transformer.to(self.standby_device)
        torch.cuda.empty_cache()

    def caption_image(self, raw_image: Image, max_new_tokens: int = 20) -> str:
        #load model into vram
        self.captioning_transformer.to(self.main_device)

        inputs = self.captioning_processor(raw_image.convert('RGB'), return_tensors="pt").to(self.main_device, self.torch_dtype)
        outputs = self.captioning_transformer.generate(**inputs, max_new_tokens=max_new_tokens)
        caption = self.captioning_processor.decode(outputs[0], skip_special_tokens=True)
        
        #unload model from vram
        self.captioning_transformer.to(self.standby_device)
        
        return caption
