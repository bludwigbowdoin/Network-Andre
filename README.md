# Network Andre 

### Description of work:

Regarding The Eric Andre Show, one anonymous Redditor wrote: 
>"For me, absurdist comedy works best as absurdism does; a juxtaposition and 
commentary on how we are so detached from the reality of our world, how we 
struggle so desperately and courageously to fit in, only to fail time and 
time again. The character of Eric Andre wants to be liked, wants to be funny, 
but has these f---ing insane and deluded concepts he's unequipped to execute 
even in his wildest dreams."

Network Andre's primary goal is to capture at least part of the absurdism Eric 
Andre so deftly exibits. Instead of operating as a neural network that 
generates Eric-Andre-like text, Network Andre adapts neural-network-generated 
text to the style of Eric Andre. This is done through a sort of Eric Andre 
injection, wherein a given word in the neural-network-generated text is 
swapped for a word of the same part-of-speech found in a selected episode of 
The Eric Andre Show. To evolve past the mere generation aspect of simply 
plugging in new words to generated texts, two scoring systems are introduced: 
one an internal sentence score, the other a type of plagiarism score. The 
former takes all distinct pairwise similarities of a given output and sums them, 
while the latter treats outputs as one big vector sum to compare against the 
original output sum. 

The text seed can be truly unique to the user. Although poetry was one of the 
main goals of the project, seeds can be cheesy poetic, slightly philosophical, 
riddle-like, or virtually anything else. After enough experimentation, some 
aspect of the output is likely to put a smile on your face, while other aspects 
might leave you dumbfounded, confused, or even uncomfortable. A viewer of The 
Eric Andre Show is susceptible to the exact same feelings. Although vulgarity 
is possible if not probable in the output, any outputs that exhibit offensive 
or biggoted behavior are not intended. I attempted to mitigate this aspect by 
using the most up-to-date numberbatch word embeddings (which purposefully 
combat discriminatory tendencies), but this was by no means a leak-proof fix. 
I apologize in advance, and I do not condone any offensive or biggoted output 
this system may generate. 


### How to set up and run the code:

To run the code after cloning the repository, the mini.h5 file that contains 
the word embedding dataset (too large to push to GitHub) must be downloaded 
from this [page](https://github.com/commonsense/conceptnet-numberbatch) (the 
direct download link is [here](http://conceptnet.s3.amazonaws.com/precomputed-data/2016/numberbatch/19.08/mini.h5)).

Once mini.h5 is downloaded and in the directory, go to terminal and enter 
'streamlit run mainSL.py'. Depending on what packages and libraries you 
have installed, you might have to wait a while and/or enter the 
respective commands to download all libraries and packages. Large models are 
used for spaCy and GPT-2, so these might take a while to download. Just hold 
tight. Eventually, your default browser should open with a blank white 
Streamlit page with 'running' buffering in the top right. This will take up to 
a minute to load, and then an image and text options will appear. The buttons 
and entries are all pretty self-explanatory. For fullest functionality, turn 
on Developer Mode. If speaking mode is turned on, you will have to wait until 
the last output is spoken to try another round of Network Andre. 

Experimentation is paramount in this system. Each of the episodes presents its 
own vocabulary, and the last episode .txt groups all of them together. Further, 
the text_len directly impacts how long the outputs will be, while the 
generations number determines how many chances the systems gets to potentially 
generate better- or worse-scoring outputs. Above all, since the generations 
first come from a neural network, each round of Network Andre will 
produce entirely novel results, even with the EXACT SAME input selections. The 
tradeoff with this sort of novelty is that, yes, sometimes the results are not 
much better than gibberish or reveal some edge case behaviors of GPT-2, but on
the other hand, there can be some pretty cool results! See existing output 
folder for some arbitrarily created examples. 

### How working on this system challenged me as a computer scientist:

Regarding neural networks, my original idea was to train my own model on the
Eric Andre episode texts. I had some experience in the past using TensorFlow, 
and was able to train the model on the Bowdoin HPC server, but I was 
ultimately unable to work with the model on my own laptop due to issues in 
compatability of TensorFlow with Apple Silicon chips. On paper, TensorFlow 
should work on Apple Silicon, but after a number of hours attempting the 
installs and discovering a disgruntled community with the same issue online, I 
changed gears. This led me to use the established GPT-2 model from OpenAI. I 
knew it could generate decent text, and then I could use spaCy components to 
analyze the generated text and the Eric Andre text, and hopefully do something 
cool with those components. 

A second are of challenge I faced in this project was working with the 
Streamlit user interface. I once had a partner in a group project use 
Streamlit, but I never knew how to use it myself until now. I hadn't created 
any user interface systems like this before, so there was definitely a bit of 
a learning curve. I had to think of things I hadn't thought of before regarding 
the ease-of-use for the user, and that experience was valuable to me. 

Another challenge I faced along the way was determing when to stop implementing 
new ideas. Once I got my base ideas down, a whole floodgate of new directions 
opened. Since deadlines do indeed exist, I had to eventually settle on which 
ideas I would fully implement so that my product would function well. At this 
point, I feel this is a project I could continually come back to and 
improve, which is a really cool thing! It was super useful having the challenge
in this open-ended project to set my own goal state, since in the real world 
there are rarely assignments with discrete end states. 


### Three scholarly papers in computer science that inspired your approach, and how:

This [first paper](https://arxiv.org/pdf/2110.12765.pdf) ("'So You Think You’re 
Funny?': Rating the Humour Quotient in Standup Comedy") had a very relevant 
goal of quantifying a humor quotient for comedy given loads of audio-visual 
stand-up comedy material. The primary analysis was done on the duration of the 
laughter for a given joke. The researchers built an LSTM neural network for 
this analysis. I found this paper inspiring in that it dealt direclty with 
comedy, quantifying comedy, and neural networks. I did not have the capacity to 
do any sort of similar laughter analysis, and their resulting models proved to 
complex for me to incoporate, but the paper was still inspiring. 

This [second paper](https://arxiv.org/pdf/2004.12765.pdf) ("Computational Humor 
Using BERT Sentence Embedding in Parallel Neural Networks") also deals with 
comedy, and its goal was to classify whether a given text was humorous or not. 
There was originally some potential for me to utilize this model, but due to 
its creation via Keras (thus TensorFlow), I could not apply this model on my 
computer. 

This [third paper](https://towardsdatascience.com/teaching-gpt-2-a-sense-of-humor-fine-tuning-large-transformer-models-on-a-single-gpu-in-pytorch-59e8cec40912)
("Teaching GPT-2 transformer a sense of humor") is not an offically published 
paper, but its goal was to train a neural network that could make one-liner 
style jokes. The nascent stages of this paper's model are what my gpt2model.py 
file piggy-backs directly off of.  
