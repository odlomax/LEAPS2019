#!/bin/bash

find . -maxdepth 1 -name "*async*" -type f -print -exec rm -f {} \; 1>/tmp/out
wc -l /tmp/out
 
