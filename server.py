from functools import wraps
from flask import (
    Flask,
    jsonify,
    request,
    render_template_string,
    abort,
    send_from_directory,
    send_file,
)
from flask_cors import CORS
import markdown
import argparse
from transformers import AutoTokenizer, AutoProcessor, pipeline
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import BlipForConditionalGeneration, GPT2Tokenizer
import unicodedata
import torch
import time
import os
import gc
from PIL import Image
import base64
from io import BytesIO
from random import randint
from EmotionClassifier import EmotionClassifier
from Summarizer import Summarizer
from Captioner import Captioner
import webuiapi
import hashlib
from constants import *
from colorama import Fore, Style, init as colorama_init
from TtsInferer import TtsInferer

colorama_init()

class SplitArgs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(
            namespace, self.dest, values.replace('"', "").replace("'", "").split(",")
        )


# Script arguments
parser = argparse.ArgumentParser(
    prog="TavernAI Extras", description="Web API for transformers models"
)
parser.add_argument(
    "--port", type=int, help="Specify the port on which the application is hosted"
)
parser.add_argument(
    "--listen", action="store_true", help="Host the app on the local network"
)
parser.add_argument(
    "--share", action="store_true", help="Share the app on CloudFlare tunnel"
)
parser.add_argument("--cpu", action="store_true", help="Run the models on the CPU")
parser.add_argument("--summarization-model", help="Load a custom summarization model")
parser.add_argument(
    "--classification-model", help="Load a custom text classification model"
)
parser.add_argument("--captioning-model", help="Load a custom captioning model")
parser.add_argument(
    "--keyphrase-model", help="Load a custom keyphrase extraction model"
)
parser.add_argument("--prompt-model", help="Load a custom prompt generation model")
parser.add_argument("--embedding-model", help="Load a custom text embedding model")
parser.add_argument('--tts-model', help="Load a custom prompt generation model")
parser.add_argument('--tts-azure', action='store_true', help="Use microsoft azure for TTS")

sd_group = parser.add_mutually_exclusive_group()

local_sd = sd_group.add_argument_group("sd-local")
local_sd.add_argument("--sd-model", help="Load a custom SD image generation model")
local_sd.add_argument("--sd-cpu", help="Force the SD pipeline to run on the CPU")

remote_sd = sd_group.add_argument_group("sd-remote")
remote_sd.add_argument(
    "--sd-remote", action="store_true", help="Use a remote backend for SD"
)
remote_sd.add_argument(
    "--sd-remote-host", type=str, help="Specify the host of the remote SD backend"
)
remote_sd.add_argument(
    "--sd-remote-port", type=int, help="Specify the port of the remote SD backend"
)
remote_sd.add_argument(
    "--sd-remote-ssl", action="store_true", help="Use SSL for the remote SD backend"
)
remote_sd.add_argument(
    "--sd-remote-auth",
    type=str,
    help="Specify the username:password for the remote SD backend (if required)",
)

parser.add_argument(
    "--enable-modules",
    action=SplitArgs,
    default=[],
    help="Override a list of enabled modules",
)

args = parser.parse_args()

port = args.port if args.port else 5100
host = "0.0.0.0" if args.listen else "localhost"
summarization_model = (
    args.summarization_model
    if args.summarization_model
    else DEFAULT_SUMMARIZATION_MODEL
)
classification_model = (
    args.classification_model
    if args.classification_model
    else DEFAULT_CLASSIFICATION_MODEL
)
captioning_model = (
    args.captioning_model if args.captioning_model else DEFAULT_CAPTIONING_MODEL
)
keyphrase_model = (
    args.keyphrase_model if args.keyphrase_model else DEFAULT_KEYPHRASE_MODEL
)
prompt_model = args.prompt_model if args.prompt_model else DEFAULT_PROMPT_MODEL
embedding_model = (
    args.embedding_model if args.embedding_model else DEFAULT_EMBEDDING_MODEL
)

sd_use_remote = False if args.sd_model else True
sd_model = args.sd_model if args.sd_model else DEFAULT_SD_MODEL
sd_remote_host = args.sd_remote_host if args.sd_remote_host else DEFAULT_REMOTE_SD_HOST
sd_remote_port = args.sd_remote_port if args.sd_remote_port else DEFAULT_REMOTE_SD_PORT
sd_remote_ssl = args.sd_remote_ssl
sd_remote_auth = args.sd_remote_auth

modules = (
    args.enable_modules if args.enable_modules and len(args.enable_modules) > 0 else []
)

if len(modules) == 0:
    print(
        f"{Fore.RED}{Style.BRIGHT}You did not select any modules to run! Choose them by adding an --enable-modules option"
    )
    print(f"Example: --enable-modules=caption,summarize{Style.RESET_ALL}")

# Models init
device_string = "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
device = torch.device(device_string)
torch_dtype = torch.float32 if device_string == "cpu" else torch.float16


if 'caption' in modules:
    print('Initializing an image captioning model...')
    captioner = Captioner(captioning_model, device, torch_dtype)

if 'summarize' in modules:
    print('Initializing a text summarization model...')
    #hard coded to cpu due to issues with large summaries
    summarizer = Summarizer(summarization_model, device, torch_dtype)

if 'classify' in modules:
    print('Initializing a sentiment classification pipeline...')
    classifier = EmotionClassifier(classification_model, device, torch_dtype)

if "keywords" in modules:
    print("Initializing a keyword extraction pipeline...")
    import pipelines as pipelines

    keyphrase_pipe = pipelines.KeyphraseExtractionPipeline(keyphrase_model)

if "prompt" in modules:
    print("Initializing a prompt generator")
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    gpt_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    gpt_model = AutoModelForCausalLM.from_pretrained(prompt_model)
    prompt_generator = pipeline(
        "text-generation", model=gpt_model, tokenizer=gpt_tokenizer
    )

if 'tts' in modules:
    try:
        print('Initializing TTS server')
        if args.tts_model:
            tts_model = args.tts_model
        else: 
            tts_models = os.listdir("tts_models")
            if len(tts_models) > 0:
                tts_model = tts_models[0]
                print(f"No TTS model selected, using model {tts_model} by default")
        if classifier is None:
            classifier = EmotionClassifier(classification_model, device)
        use_azure = True if args.tts_azure else False
        tts_service = TtsInferer(tts_model, classifier, device, use_azure)        
    except Exception as e:
        print(f'{Fore.RED}{Style.BRIGHT}TTS model could not be loaded! a so-vits-svc model must be in the tts_models folder to use TTS{Style.RESET_ALL}')
        modules.remove("tts")

if 'sd' in modules and not sd_use_remote:
    from diffusers import StableDiffusionPipeline
    from diffusers import EulerAncestralDiscreteScheduler

    print("Initializing Stable Diffusion pipeline")
    sd_device_string = (
        "cuda" if torch.cuda.is_available() and not args.sd_cpu else "cpu"
    )
    sd_device = torch.device(sd_device_string)
    sd_torch_dtype = torch.float32 if sd_device_string == "cpu" else torch.float16
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        sd_model, custom_pipeline="lpw_stable_diffusion", torch_dtype=sd_torch_dtype
    ).to(sd_device)
    sd_pipe.safety_checker = lambda images, clip_input: (images, False)
    sd_pipe.enable_attention_slicing()
    # pipe.scheduler = KarrasVeScheduler.from_config(pipe.scheduler.config)
    sd_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        sd_pipe.scheduler.config
    )
elif "sd" in modules and sd_use_remote:
    print("Initializing Stable Diffusion connection")
    try:
        sd_remote = webuiapi.WebUIApi(
            host=sd_remote_host, port=sd_remote_port, use_https=sd_remote_ssl
        )
        if sd_remote_auth:
            username, password = sd_remote_auth.split(":")
            sd_remote.set_auth(username, password)
        sd_remote.util_wait_for_ready()
    except Exception as e:
        # remote sd from modules
        print(
            f"{Fore.RED}{Style.BRIGHT}Could not connect to remote SD backend at http{'s' if sd_remote_ssl else ''}://{sd_remote_host}:{sd_remote_port}! Disabling SD module...{Style.RESET_ALL}"
        )
        modules.remove("sd")

if "chromadb" in modules:
    print("Initializing ChromaDB")
    import chromadb
    import posthog
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer

    # disable chromadb telemetry
    posthog.capture = lambda *args, **kwargs: None
    chromadb_client = chromadb.Client(Settings(anonymized_telemetry=False))
    chromadb_embedder = SentenceTransformer(embedding_model)
    chromadb_embed_fn = chromadb_embedder.encode


# Flask init
app = Flask(__name__)
CORS(app)  # allow cross-domain requests
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024


def require_module(name):
    def wrapper(fn):
        @wraps(fn)
        def decorated_view(*args, **kwargs):
            if name not in modules:
                abort(403, "Module is disabled by config")
            return fn(*args, **kwargs)

        return decorated_view

    return wrapper


# AI stuff
def classify_text(text: str) -> list:
    output = classifier.text_to_reaction(text)
    return sorted(output, key=lambda x: x['score'], reverse=True)

def normalize_string(input: str) -> str:
    output = " ".join(unicodedata.normalize("NFKC", input).strip().split())
    return output


def extract_keywords(text: str) -> list:
    punctuation = "(){}[]\n\r<>"
    trans = str.maketrans(punctuation, " " * len(punctuation))
    text = text.translate(trans)
    text = normalize_string(text)
    return list(keyphrase_pipe(text))


def generate_prompt(keywords: list, length: int = 100, num: int = 4) -> str:
    prompt = ", ".join(keywords)
    outs = prompt_generator(
        prompt,
        max_length=length,
        num_return_sequences=num,
        do_sample=True,
        repetition_penalty=1.2,
        temperature=0.7,
        top_k=4,
        early_stopping=True,
    )
    return [out["generated_text"] for out in outs]


def generate_image(data: dict) -> Image:
    prompt = normalize_string(f'{data["prompt_prefix"]} {data["prompt"]}')

    if sd_use_remote:
        image = sd_remote.txt2img(
            prompt=prompt,
            negative_prompt=data["negative_prompt"],
            sampler_name=data["sampler"],
            steps=data["steps"],
            cfg_scale=data["scale"],
            width=data["width"],
            height=data["height"],
            restore_faces=data["restore_faces"],
            enable_hr=data["enable_hr"],
            save_images=True,
            send_images=True,
            do_not_save_grid=False,
            do_not_save_samples=False,
        ).image
    else:
        image = sd_pipe(
            prompt=prompt,
            negative_prompt=data["negative_prompt"],
            num_inference_steps=data["steps"],
            guidance_scale=data["scale"],
            width=data["width"],
            height=data["height"],
        ).images[0]

    image.save("./debug.png")
    return image


def image_to_base64(image: Image, quality: int = 75) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=quality)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


@app.before_request
# Request time measuring
def before_request():
    request.start_time = time.time()


@app.after_request
def after_request(response):
    duration = time.time() - request.start_time
    response.headers["X-Request-Duration"] = str(duration)
    return response


@app.route("/", methods=["GET"])
def index():
    with open("./README.md", "r", encoding="utf8") as f:
        content = f.read()
    return render_template_string(markdown.markdown(content, extensions=["tables"]))


@app.route("/api/extensions", methods=["GET"])
def get_extensions():
    extensions = dict(
        {
            "extensions": [
                {
                    "name": "not-supported",
                    "metadata": {
                        "display_name": """<span style="white-space:break-spaces;">Extensions serving using Extensions API is no longer supported. Please update the mod from: <a href="https://github.com/Cohee1207/SillyTavern">https://github.com/Cohee1207/SillyTavern</a></span>""",
                        "requires": [],
                        "assets": [],
                    },
                }
            ]
        }
    )
    return jsonify(extensions)


@app.route("/api/caption", methods=["POST"])
@require_module("caption")
def api_caption():
    data = request.get_json()

    if "image" not in data or not isinstance(data["image"], str):
        abort(400, '"image" is required')

    image = Image.open(BytesIO(base64.b64decode(data["image"])))
    image = image.convert("RGB")
    image.thumbnail((512, 512))
    caption = captioner.caption_image(image)
    thumbnail = image_to_base64(image)
    print("Caption:", caption, sep="\n")
    gc.collect()
    return jsonify({"caption": caption, "thumbnail": thumbnail})


@app.route("/api/summarize", methods=["POST"])
@require_module("summarize")
def api_summarize():
    data = request.get_json()

    if "text" not in data or not isinstance(data["text"], str):
        abort(400, '"text" is required')

    params = DEFAULT_SUMMARIZE_PARAMS.copy()

    if "params" in data and isinstance(data["params"], dict):
        params.update(data["params"])
    print('Summary input:', data['text'], sep="\n")
    summary = normalize_string(summarizer.summarize_chunks(data['text'], params))
    print('Summary output:', summary, sep="\n")
    gc.collect()
    return jsonify({'summary': summary})

@app.route('/api/combine_summary', methods=['POST'])
@require_module('summarize')
def api_combine_summary():
    data = request.get_json()

    if 'old_summary' not in data or not isinstance(data['old_summary'], str):
        abort(400, '"old_summary" is required')
    if 'new_messages' not in data or not isinstance(data['new_messages'], str):
        abort(400, '"new_messages" is required')

    params = DEFAULT_SUMMARIZE_PARAMS.copy()

    if 'params' in data and isinstance(data['params'], dict):
        params.update(data['params'])

    print('Summary input:', data['old_summary'] + "\n" + data['new_messages'], sep="\n")
    summary = normalize_string(summarizer.combine_summary(data['old_summary'], data['new_messages'], params))
    print('Summary output:', summary, sep="\n")
    gc.collect()
    return jsonify({"summary": summary})


@app.route("/api/classify", methods=["POST"])
@require_module("classify")
def api_classify():
    data = request.get_json()

    if "text" not in data or not isinstance(data["text"], str):
        abort(400, '"text" is required')

    print("Classification input:", data["text"], sep="\n")
    classification = classify_text(data["text"])
    print("Classification output:", classification, sep="\n")
    gc.collect()
    return jsonify({"classification": classification})


@app.route("/api/classify/labels", methods=["GET"])
@require_module("classify")
def api_classify_labels():
    labels = classifier.get_all_reactions()
    return jsonify({'labels': labels})


@app.route("/api/keywords", methods=["POST"])
@require_module("keywords")
def api_keywords():
    data = request.get_json()

    if "text" not in data or not isinstance(data["text"], str):
        abort(400, '"text" is required')

    print("Keywords input:", data["text"], sep="\n")
    keywords = extract_keywords(data["text"])
    print("Keywords output:", keywords, sep="\n")
    return jsonify({"keywords": keywords})


@app.route("/api/prompt", methods=["POST"])
@require_module("prompt")
def api_prompt():
    data = request.get_json()

    if "text" not in data or not isinstance(data["text"], str):
        abort(400, '"text" is required')

    keywords = extract_keywords(data["text"])

    if "name" in data and isinstance(data["name"], str):
        keywords.insert(0, data["name"])

    print("Prompt input:", data["text"], sep="\n")
    prompts = generate_prompt(keywords)
    print("Prompt output:", prompts, sep="\n")
    return jsonify({"prompts": prompts})


@app.route("/api/image", methods=["POST"])
@require_module("sd")
def api_image():
    required_fields = {
        "prompt": str,
    }

    optional_fields = {
        "steps": 30,
        "scale": 6,
        "sampler": "DDIM",
        "width": 512,
        "height": 512,
        "restore_faces": False,
        "enable_hr": False,
        "prompt_prefix": PROMPT_PREFIX,
        "negative_prompt": NEGATIVE_PROMPT,
    }

    data = request.get_json()

    # Check required fields
    for field, field_type in required_fields.items():
        if field not in data or not isinstance(data[field], field_type):
            abort(400, f'"{field}" is required')

    # Set optional fields to default values if not provided
    for field, default_value in optional_fields.items():
        type_match = (
            (int, float)
            if isinstance(default_value, (int, float))
            else type(default_value)
        )
        if field not in data or not isinstance(data[field], type_match):
            data[field] = default_value

    try:
        print("SD inputs:", data, sep="\n")
        image = generate_image(data)
        base64image = image_to_base64(image, quality=90)
        return jsonify({"image": base64image})
    except RuntimeError as e:
        abort(400, str(e))


@app.route("/api/image/model", methods=["POST"])
@require_module("sd")
def api_image_model_set():
    data = request.get_json()

    if not sd_use_remote:
        abort(400, "Changing model for local sd is not supported.")
    if "model" not in data or not isinstance(data["model"], str):
        abort(400, '"model" is required')

    old_model = sd_remote.util_get_current_model()
    sd_remote.util_set_model(data["model"], find_closest=False)
    # sd_remote.util_set_model(data['model'])
    sd_remote.util_wait_for_ready()
    new_model = sd_remote.util_get_current_model()

    return jsonify({"previous_model": old_model, "current_model": new_model})


@app.route("/api/image/model", methods=["GET"])
@require_module("sd")
def api_image_model_get():
    model = sd_model

    if sd_use_remote:
        model = sd_remote.util_get_current_model()

    return jsonify({"model": model})


@app.route("/api/image/models", methods=["GET"])
@require_module("sd")
def api_image_models():
    models = [sd_model]

    if sd_use_remote:
        models = sd_remote.util_get_model_names()

    return jsonify({"models": models})


@app.route("/api/image/samplers", methods=["GET"])
@require_module("sd")
def api_image_samplers():
    samplers = ["Euler a"]

    if sd_use_remote:
        samplers = [sampler["name"] for sampler in sd_remote.get_samplers()]

    return jsonify({"samplers": samplers})


@app.route("/api/modules", methods=["GET"])
def get_modules():
    return jsonify({"modules": modules})


@app.route("/api/tts/speakers", methods=["GET"])
def tts_speakers():
    voices = [
        {
            "name":speaker,
            "voice_id":speaker,
            "preview_url": False
        } for speaker in tts_service.get_speakers()
    ]
    return jsonify(voices)


@app.route("/api/tts/generate", methods=["POST"])
def tts_generate():
    voice = request.get_json()
    if "text" not in voice or not isinstance(voice["text"], str):
        abort(400, '"text" is required')
    if "speaker" not in voice or not isinstance(voice["speaker"], str):
        abort(400, '"speaker" is required')
    # Remove asterisks
    voice["text"] = voice["text"].replace("*", "")
    try:
        tts_service.infer(voice['text'], voice['speaker'])
        return send_file(tts_service.tmp_wav_path, mimetype='audio/wav')
    except Exception as e:
        print(e)
        abort(500, voice["speaker"])



@app.route("/api/chromadb", methods=["POST"])
@require_module("chromadb")
def chromadb_add_messages():
    data = request.get_json()
    if "chat_id" not in data or not isinstance(data["chat_id"], str):
        abort(400, '"chat_id" is required')
    if "messages" not in data or not isinstance(data["messages"], list):
        abort(400, '"messages" is required')

    chat_id_md5 = hashlib.md5(data["chat_id"].encode()).hexdigest()
    collection = chromadb_client.get_or_create_collection(
        name=f"chat-{chat_id_md5}", embedding_function=chromadb_embed_fn
    )

    documents = [m["content"] for m in data["messages"]]
    ids = [m["id"] for m in data["messages"]]
    metadatas = [
        {"role": m["role"], "date": m["date"], "meta": m.get("meta", "")}
        for m in data["messages"]
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    return jsonify({"count": len(ids)})


@app.route("/api/chromadb/query", methods=["POST"])
@require_module("chromadb")
def chromadb_query():
    data = request.get_json()
    if "chat_id" not in data or not isinstance(data["chat_id"], str):
        abort(400, '"chat_id" is required')
    if "query" not in data or not isinstance(data["query"], str):
        abort(400, '"query" is required')

    if "n_results" not in data or not isinstance(data["n_results"], int):
        n_results = 1
    else:
        n_results = data["n_results"]

    chat_id_md5 = hashlib.md5(data["chat_id"].encode()).hexdigest()
    collection = chromadb_client.get_or_create_collection(
        name=f"chat-{chat_id_md5}", embedding_function=chromadb_embed_fn
    )

    n_results = min(collection.count(), n_results)
    query_result = collection.query(
        query_texts=[data["query"]],
        n_results=n_results,
    )

    documents = query_result["documents"][0]
    ids = query_result["ids"][0]
    metadatas = query_result["metadatas"][0]
    distances = query_result["distances"][0]

    messages = [
        {
            "id": ids[i],
            "date": metadatas[i]["date"],
            "role": metadatas[i]["role"],
            "meta": metadatas[i]["meta"],
            "content": documents[i],
            "distance": distances[i],
        }
        for i in range(len(ids))
    ]

    return jsonify(messages)


if args.share:
    from flask_cloudflared import _run_cloudflared
    import inspect

    sig = inspect.signature(_run_cloudflared)
    sum = sum(
        1
        for param in sig.parameters.values()
        if param.kind == param.POSITIONAL_OR_KEYWORD
    )
    if sum > 1:
        metrics_port = randint(8100, 9000)
        cloudflare = _run_cloudflared(port, metrics_port)
    else:
        cloudflare = _run_cloudflared(port)
    print("Running on", cloudflare)

app.run(host=host, port=port)