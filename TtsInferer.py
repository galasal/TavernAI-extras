import torch
from pathlib import Path
from so_vits_svc_fork.inference.core import Svc
import numpy as np
import librosa
import azure.cognitiveservices.speech as speechsdk
import os
from lxml import etree
import audio_processor
import json
import soundfile
import shutil
import hashlib
import glob
import edge_tts
import asyncio

class TtsInferer:
    models_directory = "tts_models/"
    cache_directory = "tts_cache/"
    tmp_wav_path = "tmp/tmp.wav"
    max_cache_files = 1000

    def __init__(self, model_folder_name, classifier, device, use_azure):
        self.model_folder = Path(self.models_directory + model_folder_name)
        self.__initialise_sovits(device)
        self.use_azure = use_azure
        if(self.use_azure):
            self.__initialise_azure()
        self.__cleanup_cache()
        self.classifier = classifier

    def __initialise_sovits(self, device):
        svc_config_path = self.model_folder / "config.json"
        svc_model_path = next(iter(self.model_folder.glob("G_*.pth")), None)
        self.svc = Svc(
            net_g_path=svc_model_path.as_posix(),
            config_path=svc_config_path.as_posix(),
            cluster_model_path=None,
            device=device,
        )

        with open(svc_config_path) as f:
            model_config = json.load(f)
        self.speakers = model_config.get("spk").keys()

    def __initialise_azure(self):
        speech_key = os.environ.get('SPEECH_KEY')
        service_region = os.environ.get('SPEECH_REGION')
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
        speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm)  
        self.azureSynthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

    def _initialise_tts_config(self):
        tts_config_path = self.model_folder / "azure-config.json" if self.use_azure else self.model_folder / "edge-config.json"
        with open(tts_config_path) as f:
            tts_config = json.load(f)

        self.azure_pitch = tts_config.get("azurePitch")
        self.emotion_override = tts_config.get("emotionOverride")
        self.base_voice = tts_config.get("baseVoice")
        self.high_pass_cutoff_freq = tts_config.get("highPassCutoffFreq")
        self.low_pass_cutoff_freq = tts_config.get("lowPassCutoffFreq")
        self.pitch_semitones = tts_config.get("pitchSemitones")
        self.speed = tts_config.get("speed")


    #file name to use when saving to tts_cache
    #file name for cached tts wav files is made by concatenating all generation information. 
    #If two tts inferences have the same generation information, their file names will match
    def __get_cache_filename(self, text, speaker, emotion):
        hash_string = text + speaker + emotion + self.base_voice + str(self.high_pass_cutoff_freq) + str(self.low_pass_cutoff_freq) + str(self.azure_pitch) + str(self.pitch_semitones) + str(self.speed) + str(self.emotion_override)
        hash_string = hashlib.sha256(hash_string.encode()).hexdigest()
        return self.cache_directory + hash_string + ".wav"
    
    #delete old cache files to get count under maximum
    def __cleanup_cache(self):
        files = glob.glob(os.path.join(self.cache_directory, '*'))
        file_count = len(files)
        
        if file_count > self.max_cache_files:
            files.sort(key=os.path.getctime)
            files_to_delete = file_count - self.max_cache_files
            for i in range(files_to_delete):
                os.remove(files[i])

    #does azure tts. Saves result to temp file and also returns it
    def azure_infer(self, text, speaker=None, emotion=None, speed=None, pitch=None):
        self.tree = etree.parse("azure.xml")
        self.tree.find(".//{*}prosody").text = text
        if emotion is not None:
            self.tree.find(".//{*}express-as").set("style", f"{emotion}")
        if speed is not None:
            self.tree.find(".//{*}prosody").set("rate", f"{speed}")
        if speaker is not None:
            self.tree.find(".//{*}voice").set("name", speaker)
        if pitch is not None:
            self.tree.find(".//{*}prosody").set("pitch", pitch)
        ssml_string = etree.tostring(self.tree).decode('utf-8')
        result = self.azureSynthesizer.speak_ssml_async(ssml_string).get()
        stream = speechsdk.AudioDataStream(result)
        stream.save_to_wav_file(self.tmp_wav_path)
        audio, sr = librosa.load(self.tmp_wav_path, sr=self.svc.target_sample)  
        return audio 

    async def edge_infer(self, text, speaker, speed=1):
        #convert speed to +x% format
        speed = (int)(speed * 100 - 100)
        if speed >= 0:
            speed = f"+{speed}%"
        else:
            speed = f"{speed}%"

        communicate = edge_tts.Communicate(text=text, voice=speaker, rate=speed)
        await communicate.save(self.tmp_wav_path)
        audio, sr = librosa.load(self.tmp_wav_path, sr=self.svc.target_sample)  
        return audio 

    #convert audio using so-vits-svc
    #audio should already be at correct sample rate
    def svc_infer(self, audio, speaker):
        audio = self.svc.infer_silence(
            audio.astype(np.float32),
            speaker=speaker,
            transpose=0,
            auto_predict_f0=0,
            cluster_infer_ratio=0,
            noise_scale=0.4,
            f0_method="dio",
            db_thresh=-40,
            pad_seconds=0.5,
            chunk_seconds=0.5,
            absolute_thresh=False,
        )

        return audio

    def infer(self, text, speaker):
        self._initialise_tts_config()
        if self.use_azure:
            if self.emotion_override is not None and len(self.emotion_override) > 0:
                emotion = self.emotion_override
            else:
                emotion = self.classifier.text_to_azure_emotion(text)
        else:
            emotion = "default"

        cache_filename = self.__get_cache_filename(text, speaker, emotion)
        if os.path.exists(cache_filename) is False:
            if self.use_azure:
                raw_audio = self.azure_infer(text=text, speaker=self.base_voice, speed=self.speed, emotion=emotion, pitch=self.azure_pitch)
            else:
                raw_audio = asyncio.run(self.edge_infer(text, self.base_voice, self.speed))
            if self.high_pass_cutoff_freq > 0:
                raw_audio = audio_processor.filter_audio(audio=raw_audio, sr=self.svc.target_sample, filter_type="highpass", cutoff_freq=self.high_pass_cutoff_freq)
            if self.low_pass_cutoff_freq > 0:
                raw_audio = audio_processor.filter_audio(audio=raw_audio, sr=self.svc.target_sample, filter_type="lowpass", cutoff_freq=self.low_pass_cutoff_freq)
            if self.pitch_semitones != 0:
                raw_audio = audio_processor.shift_frequency(audio=raw_audio, sr=self.svc.target_sample, shift_semitones=self.pitch_semitones)
            audio = self.svc_infer(raw_audio, speaker)
            soundfile.write(cache_filename, audio, self.svc.target_sample)
        shutil.copyfile(cache_filename, self.tmp_wav_path)
    
    def get_speakers(self):
        return self.speakers