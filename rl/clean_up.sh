#!/bin/bash

ps aux | grep pipeline | grep -v grep | awk '{print $2}' | xargs sudo kill -9
