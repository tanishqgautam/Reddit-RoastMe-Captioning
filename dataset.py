import praw
import datetime as dt
from prawcore.sessions import RetryStrategy
from psaw import PushshiftAPI
import os
import cv2
import urllib
import numpy as np
import re
from urllib.error import HTTPError
from urllib.request import urlopen
from praw.models import MoreComments


reddit = praw.Reddit(client_id='T0RqyDTVX3icGw', client_secret='JcA4SgxjUdMZfL2iyEaGx2bMIh3omA', user_agent='roastme')
api = PushshiftAPI(reddit)

subreddit = reddit.subreddit('RoastMe')
for submission in subreddit.top(limit=5):
    print(submission)

for submission in subreddit.top(limit=1):
    print(submission.title)
    print(submission.url)
    print(submission.comments)


start_epoch=int(dt.datetime(2019, 1, 1).timestamp())
new_list  = list(api.search_submissions(after=start_epoch,
                                        # sort_type='score',
                                        subreddit='RoastMe'
                                        ,filter=['url','author', 'title', 'subreddit'],limit=300000))


print(len(new_list))


import datetime as dtstart_epoch
import cv2
import urllib
import numpy as np
# import urllib.request
import uuid 
import matplotlib.pyplot as plt
import re
from urllib.error import URLError, HTTPError
from urllib import request
from urllib.request import urlopen
import os
from praw.models import MoreComments
# from urllib.request import Request, urlopen


posts_IDs = []
new_post_ids = []

comments_data=open("/Users/tanishqgautam/Documents/RoastMe/token.txt", "a+")
post_ID_list=open("/Users/tanishqgautam/Documents/RoastMe/posts_id.txt", "r")
posts_IDs = str(post_ID_list.read()).split("\n")


print("continuing from {}".format(len(posts_IDs)))

all_post_ids = posts_IDs.copy()
iteration = 0
duplicates = 0

for submission in new_list:    
    iteration = iteration + 1   
    print("On iteration : {}".format(iteration))        
    comment_counter = 0
    
    if submission in all_post_ids :    
        duplicates = duplicates + 1
        print("Duplicates found until now : {}".format(duplicates))
        continue

    if len(submission.comments) < 20:
        print("Skipping less comment of {}".format(submission.id))
        continue
        
    elif str(submission.url).split(".")[-1] == "jpg":
        all_post_ids.append(str(submission.id))
        new_post_ids.append(str(submission.id))
        print(submission.url)
        
        try:  
            with urllib.request.urlopen(submission.url) as url:
                arr = np.asarray(bytearray(url.read()), dtype=np.uint8)
                if arr.shape == (503,):
                    continue
    
                rgb_img = cv2.imdecode(arr, -1) 
                cv2.imwrite(os.path.join('/Users/tanishqgautam/Documents/RoastMe/image_data',str(submission.id)+".jpg"),rgb_img)
                

        except HTTPError as e:   
            if e.code == 502:
                @RetryStrategy(urllib.error.URLError, tries=4, delay=3, backoff=2)
                def urlopen_with_retry():
                    return urllib.request.urlopen(submission.url)
                urlopen_with_retry()
                
            if e.reason == 'Not Found':
                continue
            else:
                print ('Failure: ' + str(e.reason))
                
        for comment in submission.comments:
            if isinstance(comment, MoreComments):                              
                continue              
            if comment_counter == 10 :              
                break
                
            a = comment.body
            if len(str(a).split("\n")) == 1 and len(a) < 90 and len(a) > 7:
                comment_counter = comment_counter + 1                   
                a_filtered = " ".join(re.findall(r"[a-zA-Z0-9]+", str(a))).lower()           
                print("{}.jpg#{}\t{}".format(submission.id,comment_counter,a_filtered))
                comments_data.write("{}.jpg#{}\t{}\n".format(submission.id,comment_counter,a_filtered))
                
    else:
        continue
        
comments_data.close()
post_ID_list.close()

post_Write_file=open("/Users/tanishqgautam/Documents/RoastMe/posts_id.txt", "a+")
for post_id in new_post_ids:
    post_Write_file.write("{}\n".format(post_id))
    

post_Write_file.close()    