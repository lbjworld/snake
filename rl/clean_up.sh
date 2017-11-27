#!/bin/bash

ps aux | grep pipeline | grep -v grep | awk '{print $2}' | xargs sudo kill -9

sudo rm ./models/* -f
sudo rm ./tmp_models/* -f
sudo rm ./sim_data/* -f
sudo rm *.log -f
