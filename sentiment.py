from transformers import pipeline

def clean_text(text):
    cleaned_text = text.encode().decode('unicode_escape')
    cleaned_text = cleaned_text.replace('\\n', '\n')
    return cleaned_text

# returns a list of the sentiment for each paragraph in the form:
# (sentiment: str, paragraph: str)
def sentiment(text: str) -> list[tuple[str, str]]:
    cleaned_text = clean_text(text)

    paragraphs = []
    for paragraph in cleaned_text.split('\n\n'):
        paragraphs.append(paragraph.strip())


    pipe = pipeline("text-classification", model="ProsusAI/finbert")
    print('INFO: Created text classification pipeline')

# gather paragraphs with either very positive or very negative
# sentiment for summarization
    paragraphs_to_summarize = []
    for paragraph in paragraphs:
        scores = pipe(paragraph)
        if scores[0]['label'] != 'neutral' and scores[0]['score'] > 0.5:
            paragraphs_to_summarize.append((scores[0]['label'], paragraph))

    return paragraphs_to_summarize

