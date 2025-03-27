 

https://www.kaggle.com/datasets/emrahaydemr/gunshot-audio-dataset

Varying environment model was used to gather gunshot audios since YouTube was used to data source. The audios of 
the gun models were collected on YouTube using videos that are open to everyone. These files gathered on different 
dates were downloaded and parts were produced for 2 seconds for each gun type. In this way, a total of 851 files types
were obtained with 8 gun models. The sampling rate of the audio files is 44100 Hz. Firstly, each audio file was 
converted to wav file format. And then, the audios were carefully listened to and made sure that there were no 
different audios or noises inside. It has been carefully checked that the same audio does not continue repeatedly 
during the fragmentation of the audio files. WavePad Audio Editor program was used for all these processes. 
Table 1 denotes details of the number of audios formed are listed.

Table 1. Details of the collected gunshots audio dataset
ID Model Number of observations
1 AK-47 72
2 IMI Desert Eagle (Desert Eagle) 100
3 AK-12 98
4 M16 200
5 M249 99
6 MG-42 100
7 MP5 100
8 Zastava M92 82

To use this dataset, the following article should be cited.

Tuncer, T., Dogan, S., Akbal, E., Aydemir, E. (2021). An Automated Gunshot Audio Classification Method Based On 
Finger Pattern Feature Generator And Iterative Relieff Feature Selector, Journal of Engineering Science of Adıyaman University


SPLIT DATA:
ffmpeg -i 0_arena_basket.wav -f segment -segment_time 3 -c copy ARENA/arena_basket_%03d.wav
