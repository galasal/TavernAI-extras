#NOTE::DO NOT REMOVE UNUSED IMPORTS HERE. THEY IMPROVE PERFORMANCE FOR SOME REASON
from transformers import AutoTokenizer, AutoProcessor, pipeline
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
import unicodedata
import torch
import time
import gc

class Summarizer:
    max_chunk_length = 3000

    def __init__(self, model, device, torch_dtype):
        self.main_device = device
        self.standby_device = torch.device("cpu")
        self.summarization_tokenizer = AutoTokenizer.from_pretrained(model)
        self.summarization_transformer = AutoModelForSeq2SeqLM.from_pretrained(model, torch_dtype=torch_dtype)


    def __summarize(self, text: str, params: dict) -> str:
        # Tokenize input
        inputs = self.summarization_tokenizer(text, return_tensors="pt").to(self.main_device)
        token_count = len(inputs[0])

        bad_words_ids = [
            self.summarization_tokenizer(bad_word, add_special_tokens=False).input_ids
            for bad_word in params['bad_words']
        ]
        #load transformer into vram if using gpu
        self.summarization_transformer.to(self.main_device)

        summary_ids = self.summarization_transformer.generate(
            inputs["input_ids"],
            num_beams=2,
            max_new_tokens=min(token_count, int(params['max_length'])),
            min_new_tokens=min(token_count, int(params['min_length'])),
            repetition_penalty=float(params['repetition_penalty']),
            temperature=float(params['temperature']),
            length_penalty=float(params['length_penalty']),
            bad_words_ids=bad_words_ids,
        )
        
        summary = self.summarization_tokenizer.batch_decode(
            summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        #unload transformer from vram
        self.summarization_transformer.to(self.standby_device)
        torch.cuda.empty_cache()

        return summary


    def summarize_chunks(self, text: str, params: dict) -> str:
        try:
            if len(text) > self.max_chunk_length:
                chunks = self.__chunkstring(text, self.max_chunk_length)
                summary = ""
                new_params = params.copy()
                new_params['max_length'] = new_params['max_length'] // len(chunks)
                new_params['min_length'] = new_params['min_length'] // len(chunks)
                if(new_params['max_length']) < 56:
                    new_params['max_length'] = 56
                for chunk in chunks:
                    summary += self.summarize_chunks(chunk, new_params) + ". "
                return summary
            else:
                summary = self.__summarize(text, params)
                return summary
        except IndexError:
            print("Sequence length too large for model, cutting text in half and calling again")
            self.max_chunk_length = self.max_chunk_length // 2
            return self.summarize_chunks(text, params)
        
    def __chunkstring(self, string, length):
        return [string[0+i:length+i] for i in range(0, len(string), length)]