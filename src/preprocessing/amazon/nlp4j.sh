#!/bin/bash

apppath=~/works/radio/w2v/appassembler/
$apppath/bin/nlpdecode -c $apppath/config-decode-en.xml -i ../../../data/amazon/test.txt > log_test.txt &
for num in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19
do
    echo $apppath/bin/nlpdecode -c $apppath/config-decode-en.xml -i ../../../data/amazon/train_$num.txt > log_train$num.txt &
    $apppath/bin/nlpdecode -c $apppath/config-decode-en.xml -i ../../../data/amazon/train_$num.txt > log_train$num.txt &
done

