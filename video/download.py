from yt_dlp import YoutubeDL

def download_video(url):
    ydl_opts = {
        'format': 'best',
        'outtmpl': './video/%(title)s.%(ext)s'
    }
    
    with YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            print("Download completed!")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

def download_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': './audio/%(title)s.%(ext)s'
    }
    
    with YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            print("Audio download completed!")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

address = "https://www.youtube.com/watch?v=lukT_WB5IB0"
#download_video(address) 
download_audio(address)

