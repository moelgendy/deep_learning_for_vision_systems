# How-to-Generate-Art-Demo
This is the code for "How to Generate Art - Intro to Deep Learning #8' by Siraj Raval on YouTube

##Overview

This is the code for [this](https://youtu.be/Oex0eWoU7AQ) video on Youtube by Siraj Raval as part of the Intro to Deep Learning Nanodegree with Udacity. We're going to re-purpose the pre-trained VGG16 convolutional network that won the ImageNet competition in 2014 for image classification to transfer the style of a given image to another. [This](https://arxiv.org/abs/1508.06576) is the original paper on the topic.


##Dependencies

run `pip install -r requirements.txt` to install the necessary dependencies


##Usage

If it doesn't exist, create a file called ~/.keras/keras.json and make sure it looks like the following:

   ````
   {
       "image_dim_ordering": "tf",
       "epsilon": 1e-07,
       "floatx": "float32",
       "backend": "tensorflow"
   }
   ````

Then you can run the code via typing `jupyter notebook` into terminal


#Coding Challenge - Due Date is Thursday, March 9th at 12 PM PST

Use 2 different style images and transfer them both onto a base image. This can be done several ways, take your pic! And if you want even more of a challenge, bonus points are given if you instead perform basic style transfer on video. Remember, a video is just a series of image frames. You'll learn a lot about matrix operations by doing this. Good luck!


##Credits


The credits for this code go to [hnarayanan](https://github.com/hnarayanan/artistic-style-transfer). I've merely created a wrapper to get people started.




