#!/usr/bin/env bash
# Script to convert mkv video files to mp4
DIN=../data/fiw-videos/new-processed/

for DIR in ${DIN}F????/v*;
do
  echo "$DIR"
  for FILE in ${DIR}/scenes/*.mkv;
  do
      ffmpeg -i "$FILE" -c copy "${FILE%.mkv}.mp4";
  done
done