from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from fpdf import FPDF
import os

def get_youtube_transcript(video_url):
    # Extract video ID from URL
    yt = YouTube(video_url)
    video_id = yt.video_id

    # Fetch the transcript
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = ' '.join([entry['text'] for entry in transcript])
        return transcript_text
    except Exception as e:
        return str(e)

def save_transcript_to_pdf(transcript, filename='transcript.pdf'):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    
    # Set margins
    margin_left = 10
    margin_right = 10
    pdf.set_left_margin(margin_left)
    pdf.set_right_margin(margin_right)
    
    # Set width for text wrapping
    page_width = pdf.w - 2 * margin_left

    # Split the transcript into lines that fit on the PDF page
    pdf.set_font_size(12)
    pdf.multi_cell(page_width, 10, transcript)

    pdf.output(filename)

# Example usage
video_url = 'https://www.youtube.com/watch?v=UF8uR6Z6KLc'
transcript = get_youtube_transcript(video_url)
print('type: ', type(transcript))
save_transcript_to_pdf(transcript)

print(f"Transcript saved to {os.path.abspath('transcript.pdf')}")
