#!/usr/bin/python
import argparse
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError






def youtube_search(options):
  # Set DEVELOPER_KEY to the API key value from the APIs & auth > Registered apps
  # Please ensure that you have enabled the YouTube Data API for your project.

  DEVELOPER_KEY = "AIzaSyBQG9ozouzPofCHE4J-BHdUeSjqqtemnc0"
  YOUTUBE_API_SERVICE_NAME = "youtube"
  YOUTUBE_API_VERSION = "v3"

  youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
    developerKey=DEVELOPER_KEY)

  parser = argparse.ArgumentParser()
  parser.add_argument("--q", help="Search term", default=options)
  parser.add_argument("--max-results", help="Max results", default=1) # the number of default is the number of output videos
  options= parser.parse_args()

  # Call the search.list method to retrieve results matching the specified
  # query term.
  search_response = youtube.search().list(
    q=options.q,
    part="id,snippet",
    maxResults=options.max_results
  ).execute()

  videos = []
  videoid = []
  channels = []
  playlists = []
  result = []

  # Add each result to the appropriate list, and then display the lists of
  # matching videos, channels, and playlists.
  for search_result in search_response.get("items", []):
    if search_result["id"]["kind"] == "youtube#video":
      videos.append(  search_result["snippet"]["title"])
      videoid.append(  search_result["id"]["videoId"])
    '''elif search_result["id"]["kind"] == "youtube#channel":
      channels.append("%s (%s)" % (search_result["snippet"]["title"],
                                   search_result["id"]["channelId"]))
    elif search_result["id"]["kind"] == "youtube#playlist":
      playlists.append("%s (%s)" % (search_result["snippet"]["title"],
                                    search_result["id"]["playlistId"]))'''
  for line in videos:
    #print(line)
    #return line
    result.append(line)

  for line1 in videoid:
    line1 = "http://www.youtube.com/watch?v="+line1
    result.append(line1)
    #return line1
 # print("Videos:\n", "\n".join(videos), "\n")
  return result
  #print("Channels:\n", "\n".join(channels), "\n")
  #print("Playlists:\n", "\n".join(playlists), "\n")


if __name__ == "__main__":
  '''try:
    youtube_search('google')
  except HttpError as e:
    print("An HTTP error %d occurred:\n%s" % (e.resp.status, e.content))'''
  api = youtube_search('google')
  video = api[0]
  videoid = api[1]
  print(video)
  print(videoid)
