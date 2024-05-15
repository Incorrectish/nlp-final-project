import fetch_data
import summarize 
import sentiment
import topic_segmentation 
import sys

def main():
    filename = sys.argv[1]
    with open(filename, "r") as html_file:
        file_text = html_file.read()
        filtered_text = fetch_data.filter_html(file_text)
        # output is in segments.txt
        topics = topic_segmentation.topic_segmentation(filtered_text)
        for topic in topics:
            print(f"Topic: {topic}")
            print("---------")
            summary = summarize.summarize(topic)
            print(f"Summary: {summary}")
            print("---------")
            topic_sentiment = sentiment.sentiment(topic)
            print(f"Sentiment: {topic_sentiment[0]}")
            print("=================================================")


if __name__ == "__main__":
    main()
