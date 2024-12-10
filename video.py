from googleapiclient.discovery import build
import pandas as pd
from datetime import datetime

# API key (YouTube Data API v3 - Google Cloud console)
api_key = ''
# YouTube API
youtube = build('youtube', 'v3', developerKey=api_key)

def get_video_data(video_id):
    comments = []
    response = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        textFormat='plainText',
        maxResults=100
    ).execute()
    while response:
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comment = " ".join(comment.split())
            username = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
            timestamp = item['snippet']['topLevelComment']['snippet']['publishedAt']
            dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
            likes = item['snippet']['topLevelComment']['snippet']['likeCount']
            reply_count = item['snippet']['totalReplyCount']
            comments.append({
                'username': username,
                'timestamp': dt.strftime("%d/%m/%Y - %H:%M:%S"),
                'likes': likes,
                'reply_count': reply_count,
                'content': comment
            })
        # Sayfa atlama kısmı
        if 'nextPageToken' in response:
            response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                pageToken=response['nextPageToken'],
                textFormat='plainText',
                maxResults=100
            ).execute()
        else:
            break
    return comments