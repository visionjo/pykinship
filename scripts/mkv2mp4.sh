#!/usr/bin/env bash
# Script to convert mkv video files to mp4
DIN=/Volumes/MyWorld/FIW_Video/data/new-processed/

for DIR in ${DIN}F????/v*;
do
  echo "$DIR"
  for FILE in ${DIR}/scenes/*.mkv;
  do
    if [ -f "$FILE" ]; then
      echo "$FILE exist"
    else
      ffmpeg -i "$FILE" -c copy "${FILE%.mkv}.mp4";
    fi
  done
done