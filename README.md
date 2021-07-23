### LDA Demo

This project contains code and some sample data to give a brief introduction of how LDA works, how the input/output looks like. What basic steps need to be taken.<br/><br/>

![Output by LDAviz](/images/topics_dist_ldaviz.png)
<br/><br/>


### Prerequisite
The project uses [Gensim](https://radimrehurek.com/gensim), LDAviz, NLTK, so they need to be installed. Please refer to [run.sh]() file for instruction on how to install required packages and run the project.


If you need some simple explanation on LDA, I find this blog post by [Edwin Chen](http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation/) to be really helpful. [Jordan Boyd-Graber's series](https://www.youtube.com/watch?v=fCmIceNqVog) of videos on LDA are a bit more detailed. Other than that, Blei's paper is the way to go.


### How to run

```
virtualenv -p python3 env
pip install -r requirements.txt
python main.py
open lda_vizs/lda_visualization_30.html
```
<br/><br/>
![Output by LDAviz](/images/test_dist.png)