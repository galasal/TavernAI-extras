@echo off
call activate extras
python server.py --enable-modules=caption,summarize,classify,keywords,prompt
