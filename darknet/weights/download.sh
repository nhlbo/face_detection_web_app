#!/bin/bash
wget https://pjreddie.com/media/files/yolov3.weights
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1HpLkYyhNfPnshBVwaYC-52weqedlPnfT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1HpLkYyhNfPnshBVwaYC-52weqedlPnfT" -O yolov3_final.weights && rm -rf /tmp/cookies.txt
