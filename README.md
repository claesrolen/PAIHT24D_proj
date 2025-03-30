# PyAi24D final project - Gunshot Detector
---
## Project Scope and Description
This project is the final part of the course "Pythonprogrammering med AI" held at NBI/Handelsakademin.  
My project will focus on detecting shooting events in public areas and give an early indication to alert the police and/or sequrity personel.  
An imagined scenario is that you have a network of always on detectors (EdgeAI) that detect events and send alerts to the main system/network. The system may then order the device to start streaming audio and/or video from that area.
There are already systems available on the market (but large, expensive and complex) e.g [ShotSpotter](https://www.soundthinking.com/law-enforcement/leading-gunshot-detection-system/) 

In this project I will test using both Random Forest Classifier aswell as a Fully Convolutional Neural Network. Since the goal is to deploy the model in an EdgeAI we need to keep the model reasonally small to cope with the memory and performance limits of a microprocessor. The model will be optimized using TinyML and deployed via Tensorflow Lite Micro

---
## Exploratory Data Analysis
#### Data analysis
##### Gunshot characteristics
The typical characteristics of a shotgun sound is basically the same for all type of guns. It depends mainly on the mechanics and the discharge. When the bullet leaves the pipe it generates a shock impulse followed by the muzzle blast.
![image info](./DOC/shock_and_blast.png)
It is thereafter followed by reflections/reverbations by the surronding and hence depends heavily on the soundscape present. The sound also varies with distance and angular offset from the shooting direction. In other words, it is a challenging classification problem.
##### Environment characteristics
Most probable locations for the detector to be used in are public areas such as:
- schools - schoolyards, hallways etc
- shopping malls
- city areas - streets, squares, parks, subways 
- arenas - sport, concert halls
- offices
#### Dataset collection
**Gunshot** audio was mainly fetched from the [Gunshot audio dataset](https://www.kaggle.com/datasets/emrahaydemr/gunshot-audio-dataset) available at Kaggle. 
Some extra audio were recorded from various Youtube media.  

**Environmental** (used as negative class) audio were recorded from various Youtube media from soudscapes like:
- streets - busy/calm
- office, schools
- public areas, busstation, subway, harbor, arenas
- nature/parks - forest, meadow
- sounds expected to be hard to classify as negative: applause, fireworks
---
## Feature selection
#### Mel spectrogram
The Mel spectrogram is the most commonly used feature in audio classification, there are other methods used such as SFFT, wavelets, MFCC, Harmonic Percussive Signal Separation.

#### Augmentations
The Mel spectrogram is a two-dimensional matrix with frequency and time as axes. This is treated as an image and may be used in architectures used for image classifications. One difference is that a spectrogram may NOT be augmented like an image would.
Flipping or rotations are not applicable, the most used augmentations are:
- Time- or pitch-shifting (if applicable)
- Noise 
- patching in time and/or frequency

---
## Classification Models
#### Random Tree Forest
#### Fully Convolutional Neural Network
---
## Result
---
