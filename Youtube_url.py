import urllib.request
import urllib.parse
import re

#list = []
def youtube_url(search):
 query_string = urllib.parse.urlencode({"search_query" : search})
 html_content = urllib.request.urlopen("http://www.youtube.com/results?" + query_string)

 search_results = re.findall(r'href=\"\/watch\?v=(.{11})', html_content.read().decode())
 #ytb_url= "http://www.youtube.com/watch?v=" + search_results[0]

 video = search_results[:10]

 return video


if __name__ == "__main__":
    print(youtube_url('google')[0])
    print(youtube_url('google')[2])
