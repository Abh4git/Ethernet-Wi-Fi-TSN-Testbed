# Folder containing input data for Sending

This folder contains different input data  
1. text25k.dat (25000 data packets 1400 bytes of text) to be used with 2 millisconds cycles of 5000 packets (5*5000=25k)  
2. audio.dat (5000 data packets 1200 bytes of binary) to be used for 10ms cycles  
3. video.dat (5000 data packets 1400 bytes of binary) to be used for 10ms cycles

## How to create  
We can create a data file by using  dd command in linux  
dd if=/dev/urandom of=text.dat count=25000 bs=1400  

