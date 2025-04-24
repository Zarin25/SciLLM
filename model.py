import os
import google.generativeai as genai

genai.configure(api_key="AIzaSyBIfGJRoet_wzzYXIiWXxStkIigEOzSR2o")

models = genai.list_models()

for model in models:
    print(f"Name: {model.name} | Methods: {model.supported_generation_methods}")