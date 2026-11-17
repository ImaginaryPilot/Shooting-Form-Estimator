@echo off


chcp 65001 > nul


"Z:/home/kailu/Documents/Application_2026_0709/ffmpeg.exe" -i "C:/users/kailu/Documents/14_student_N/N_00/Movie.mp4" -vcodec mjpeg "C:/users/kailu/Documents/14_student_N/N_00/Sequence/seq_%%05d.jpg"


"Z:/home/kailu/Documents/Application_2026_0709/PoseEstimator.exe" 2 "C:/users/kailu/Documents/14_student_N/N_00/Sequence" "C:/users/kailu/Documents/14_student_N/N_00/"


"Z:/home/kailu/Documents/Application_2026_0709/PoseCorrector2.exe" "C:/users/kailu/Documents/14_student_N/N_00/Data2D.csv" "C:/users/kailu/Documents/14_student_N/N_00/" 2


start Z:/home/kailu/Documents/Application_2026_0709/PoseVisualizer2.exe C:/users/kailu/Documents/14_student_N/N_00/


exit
