import fetch_data
import summarize 
import label
import sys

def main():
    filename = sys.argv[1]
    with open(filename, "r") as html_file:
        file_text = html_file.read()
        filtered_text = fetch_data.filter_html(file_text)
        summary = summarize.summarize(filtered_text)
        sentiment = label.sentiment(filtered_text)

if __name__ == "__main__":
    main()
