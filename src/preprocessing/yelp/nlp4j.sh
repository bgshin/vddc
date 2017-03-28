#!/bin/bash

apppath=~/works/radio/w2v/appassembler/
$apppath/bin/nlpdecode -c $apppath/config-decode-en.xml -i ../../../data/yelp/test.txt > log_test.txt &

for num in 0 1 2 3 4 5 6 7 8 9
do
    echo $apppath/bin/nlpdecode -c $apppath/config-decode-en.xml -i ../../../data/yelp/train_$num.txt > log_train_$num.txt &
    $apppath/bin/nlpdecode -c $apppath/config-decode-en.xml -i ../../../data/yelp/train_$num.txt > log_train_$num.txt &
done

