#!/usr/bin/env python3
"""Convert GIF to MP4"""
import moviepy.editor as mp
import os

directory = 'gifs/'


for filename in os.listdir(directory ):
    if filename.endswith(".gif"):
        tmp_filename = os.path.join(directory,filename)
        print("filename",tmp_filename )
        clip = mp.VideoFileClip(tmp_filename )
        mp4_filename=filename[:-3]+"mp4"
        clip.write_videofile(mp4_filename)
        input("--")
    else:
        continue

