#!/usr/bin/env bash

DIN=/Volumes/MyWorld/FIW_Video/data/processed/
SUBSTR="scenes/"
EXT=".mkv"
EXT_VIDEO=".mp4"
EXT_AUDIO=".mp3"
DOT="."

#FLAG_AUDIO=1

#FLAG_VIDEO=0

STR_LEN=${#DIN}
for vfile in ${DIN}*/video*/scenes/*mp4 ; do
    #  dout=${vfile:0:STR_LEN}
    #  echo"$(dirname "${vfile}")/audio"
    #  if 1; then
    dout="$(dirname "${vfile}")/audio"
    mkdir -p  ${dout}
    fout=${vfile/$SUBSTR/scenes/audio/}
    fout=${fout/$EXT_VIDEO/$EXT_AUDIO}

    #    echo $vfile
    echo $fout
    ffmpeg -i ${vfile} -an ${fout}
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