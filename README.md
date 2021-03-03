# Overview
The Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer architecture allows for significantly more parallelization and can reach new state of the art results in translation quality.

This Transformer uses the architecture defined in the Attention is all you need paper. I have implemented a BLEU evaluation metric along with a Greedy Search approach.  

The first step of the project was to scrap reddit for Images and captions. This was then converted into a dataframe for further processing.   
The dataset can be found [Here](https://www.kaggle.com/ligtfeather/roastme6k)

Reddit Api
```python
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
                                        
```
Reddit Scraping for Image and 10 Captions
```python
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

```

Visualize Image and Captions
```python
npic = 5
npix = 224
target_size = (npix,npix,3)

count = 1
fig = plt.figure(figsize=(10,20))
for jpgfnm in uni_filenames[25:30]:
    filename = image_path + '/' + jpgfnm
    captions = list(data["caption"].loc[data["filename"]==jpgfnm].values)
    image_load = load_img(filename, target_size=target_size)
    
    ax = fig.add_subplot(npic,2,count,xticks=[],yticks=[])
    ax.imshow(image_load)
    count += 1
    
    ax = fig.add_subplot(npic,2,count)
    plt.axis('off')
    ax.plot()
    ax.set_xlim(0,1)
    ax.set_ylim(0,len(captions))
    for i, caption in enumerate(captions):
        ax.text(0,i,caption,fontsize=20)
    count += 1
plt.show()
```

Scaled Dot Product Attention
```python

def scaled_dot_product_attention(q, k, v, mask):

  matmul_qk = tf.matmul(q, k, transpose_b=True)  
   
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
 
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  

  output = tf.matmul(attention_weights, v) 

  return output, attention_weights
  ```
  
 Multi Head Attention
```python
  class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    
    self.dense = tf.keras.layers.Dense(d_model)
        
  def split_heads(self, x, batch_size):
    
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask=None):
    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)  
    k = self.wk(k)  
    v = self.wv(v)  
    
    q = self.split_heads(q, batch_size)  
    k = self.split_heads(k, batch_size) 
    v = self.split_heads(v, batch_size)  
    
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  

    output = self.dense(concat_attention)  
        
    return output, attention_weights
 ```
 Evaluate 
 ```python
 def evaluate(image):

  temp_input = tf.expand_dims(load_image(image)[0], 0)
  img_tensor_val = image_features_extract_model(temp_input)
  img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
  
  start_token = tokenizer.word_index['<start>']
  end_token = tokenizer.word_index['<end>']
   
  #decoder input is start token
  decoder_input = [start_token]
  output = tf.expand_dims(decoder_input, 0) #token
  result = [] #word list

  for i in range(200):
      dec_mask = create_masks_decoder(output)
  
      # predictions.shape == (batch_size, seq_len, vocab_size)
      predictions, attention_weights = transformer(img_tensor_val,output,False,dec_mask)
      
      # select the last word from the seq_len dimension
      predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

      predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
      # return the result if the predicted_id is equal to the end token
      if predicted_id == end_token:
          return result,tf.squeeze(output, axis=0), attention_weights
      # concatentate the predicted_id to the output which is given to the decoder
      # as its input.
      result.append(tokenizer.index_word[int(predicted_id)])
      output = tf.concat([output, predicted_id], axis=-1)

  return result,tf.squeeze(output, axis=0), attention_weights
```
