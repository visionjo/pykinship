#!/usr/bin/env bash

DIN=/Volumes/MyWorld/FIW-MM/data/FIDs-MM/
SUBSTR="clips/audio/"
SUBSTR1="clips/"
#EXT=".mkv"
EXT_VIDEO=".mp4"
EXT_AUDIO=".wav"
DOT="."
# 16-bit FLAC audio
#files at a 16kHz
#FLAG_AUDIO=1
#ffmpeg -i input.mp4 -vn -acodec pcm_s16le -ar 16 -ac 1 output.wav
#FLAG_VIDEO=0

STR_LEN=${#DIN}
for vfile in ${DIN}F????/MID*/clips/*/*mp4 ; do
    #  dout=${vfile:0:STR_LEN}
    fout="${vfile/$SUBSTR1/$SUBSTR}"
    parentdir="$(dirname "$fout")"
    fname="$(basename "$fout")"
    fout=${fname/$EXT_VIDEO/$EXT_AUDIO}
    tosave="${parentdir}_${fout}"
    #  if 1; then

    dout="$(dirname "${tosave}")"
    echo $tosave
    mkdir -p  ${dout}
    #    fout=${vfile/$SUBSTR/scenes/audio/}
    #    fout=${fout/$EXT_VIDEO/$EXT_AUDIO}
    #
    #    #    echo $vfile
    #    echo $fout
    ffmpeg -i ${vfile} -vn -acodec pcm_s16le -y -ar 16000 -ac 1 ${tosave}
#    ffmpeg -i ${vfile} -an ${tosave}
done

#    ffmpeg -i ${vfile} -sn ${fout}txt
#ffmpeg -i $vfile -vn  $fout
#ffmpeg -i $vfile -f mp3 -ab 192000 -vn $fout
#  ffmpeg -i $vfile -vn -acodec copy $fout
#    ffmpeg -i $vfile -codec copy -an  $fout
    #    ffmpeg -i $vfile -ab 160k -ac 2 -ar 44100 -vn $fout
#  fi
#  if 0; then
#    echo ${vfile}
#    fout=${vfile/$EXT/$EXT_VIDEO}
#    echo ${fout}
#    ffmpeg -i ${vfile} -strict experimental -map 0:0 -map 0:1 -c:v copy -c:a:1 libmp3lame -b:a 192k -ac 6 ${fout}
#  fi
#
#  echo ${fout/.mkv/.avi}
#