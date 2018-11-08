import urllib.request
import urllib.parse
import re

def youtube_url(search):
 query_string = urllib.parse.urlencode({"search_query" : 'search'})
 html_content = urllib.request.urlopen("http://www.youtube.com/results?" + query_string)
 search_results = re.findall(r'href=\"\/watch\?v=(.{11})', html_content.read().decode())
 ytb_url= "http://www.youtube.com/watch?v=" + search_results[0]

 return ytb_url


if __name__ == "__main__":
    youtube_url('google')
