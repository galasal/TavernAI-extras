from transformers import pipeline


class EmotionClassifier:
    # map model emotion to azure tts emotion
    azure_emotions = {
        'neutral': 'default',
        'anger': 'angry',
        'annoyance': 'angry',
        'disgust': 'angry',
        'amusement': 'cheerful',
        'gratitude': 'cheerful',
        'joy': 'cheerful',
        'realisation': 'cheerful',
        'admiration': 'excited',
        'desire': 'excited',
        'excitement': 'excited',
        'optimism': 'excited',
        'approval': 'friendly',
        'curiosity': 'friendly',
        'pride': 'friendly',
        'relief': 'friendly',
        'caring': 'hopeful',
        'love': 'hopeful',
        'disappointment': 'sad',
        'grief': 'sad',
        'nervousness': 'sad',
        'remorse': 'sad',
        'sadness': 'sad',
        'embarrassment': 'terrified',
        'fear': 'terrified',
        'confusion': 'unfriendly',
        'disapproval': 'unfriendly',
        'surprise': 'excited',
        'anxiety': 'terrified'
    }

    # map model emotion to reaction image
    reaction_emotions = {
        'neutral': 'joy',
        'anger': 'anger',
        'annoyance': 'anger',
        'disgust': 'anger',
        'amusement': 'joy',
        'gratitude': 'joy',
        'joy': 'joy',
        'realisation': 'joy',
        'admiration': 'surprise',
        'desire': 'surprise',
        'excitement': 'surprise',
        'optimism': 'surprise',
        'approval': 'joy',
        'curiosity': 'joy',
        'pride': 'joy',
        'relief': 'joy',
        'caring': 'love',
        'love': 'love',
        'disappointment': 'sadness',
        'grief': 'sadness',
        'nervousness': 'sadness',
        'remorse': 'sadness',
        'sadness': 'sadness',
        'embarrassment': 'fear',
        'fear': 'fear',
        'confusion': 'anger',
        'disapproval': 'anger',
        'surprise': 'surprise',
        'anxiety': 'fear'
    }

    def __init__(self, model, device):
        if device == "cpu":
            device = -1
        else:
            device = 0
        self.classifier = pipeline(
            'text-classification', model=model, device=device)
        self.last_text = None
        self.last_result = None

    def __classify(self, text):
        if text == self.last_text:
            return self.last_result
        else:
            emotion_labels = self.classifier(text)
            self.last_text = text
            self.last_result = emotion_labels
            return emotion_labels

    def text_to_azure_emotion(self, text):
        emotion_labels = self.__classify(text)
        score = emotion_labels[0]['score']
        emotion = emotion_labels[0]['label']
        if score >= 0.8:
            return self.azure_emotions[emotion]
        else:
            return 'default'

    def text_to_reaction(self, text):
        emotion_labels = self.__classify(text)
        # replace label with label for reaction emotion
        for emotion in emotion_labels:
            emotion['label'] = self.reaction_emotions[emotion['label']]

        return emotion_labels

    def get_all_reactions(self):
        return [
            'joy',
            'anger',
            'fear',
            'love',
            'sadness',
            'surprise'
        ]
