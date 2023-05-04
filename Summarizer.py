#NOTE::DO NOT REMOVE UNUSED IMPORTS HERE. THEY IMPROVE PERFORMANCE FOR SOME REASON
from transformers import AutoTokenizer, AutoProcessor, pipeline
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
import unicodedata
import torch
import time
import gc

class Summarizer:
    # 1024 is the actual limit, but need a little bit of leeway in case the calculation is wrong
    max_chunk_tokens = 999
    min_chunk_tokens = 56

    def __init__(self, model, device, torch_dtype):
        self.main_device = device
        self.standby_device = torch.device("cpu")
        self.summarization_tokenizer = AutoTokenizer.from_pretrained(model)
        self.summarization_transformer = AutoModelForSeq2SeqLM.from_pretrained(model, torch_dtype=torch_dtype)


    def __summarize(self, text: str, params: dict) -> str:
        # Tokenize input
        token_count, inputs = self.__get_tokens(text)
        #no summary if smaller than minimum
        if(token_count < self.min_chunk_tokens):
            return ""

        bad_words_ids = [
            self.summarization_tokenizer(bad_word, add_special_tokens=False).input_ids
            for bad_word in params['bad_words']
        ]
        

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

        return summary


    def summarize_chunks(self, text: str, params: dict) -> str:
        try:
            #load transformer into vram if using gpu
            self.summarization_transformer.to(self.main_device)

            chunks = self.__chunkstring(text)
            summary = ""
            for chunk in chunks:
                summary += self.__summarize(chunk, params) + ". "
            token_count, inputs = self.__get_tokens(summary)
            #summarize the summary to shrink it to correct length if necessary
            if(token_count >= self.max_chunk_tokens):
                return self.summarize_chunks(summary, params)
            elif(token_count > params['max_length']):
                return self.__summarize(summary, params)
            else:
                return summary
        except IndexError:
            print("Sequence length too large for model, cutting chunk size in half and calling again")
            self.max_chunk_tokens = self.max_chunk_tokens // 2
            return self.summarize_chunks(text, params)
        finally:
            #unload transformer from vram
            self.summarization_transformer.to(self.standby_device)
            torch.cuda.empty_cache()


    def __chunkstring(self, text):
        token_count, inputs = self.__get_tokens(text)
        num_chunks = (token_count + self.max_chunk_tokens - 1) // self.max_chunk_tokens
        chunk_length = len(text) // num_chunks
        chunks = [text[i:i+chunk_length] for i in range(0, len(text), chunk_length)]
        return chunks
    

    def __get_tokens(self, text):
        inputs = self.summarization_tokenizer(text, return_tensors="pt").to(self.main_device)
        token_count = len(inputs[0])
        return token_count, inputs
