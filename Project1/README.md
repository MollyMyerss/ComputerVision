Molly Myers
CSC391
The Pinhole, the Pixel, and the Photon

P1_Noise_Characterization.py to run use:
/opt/miniconda3/envs/CSC391CVraw/bin/python /Users/mollymyers/ComputerVision/Project1/P1_Noise_Characterization.py (because my rawpy was being weird)
This shows how much noise is present in light vs dark photos. I had already tried to figure out the light noise before we learned we didn't need it so it is still present in my code. My data is saved in dark_stats.csv, light_stats.csv, dark_hist.png, bright_hist.png.


To run:
python3 P2_Multi_Camera.py \
  --main "/Users/mollymyers/ComputerVision/Project1/images/part2/Normal.dng" \
  --tele "/Users/mollymyers/ComputerVision/Project1/images/part2/Zoom.dng"

My data is save in multi_camera_summary.csv, Normal_norm_hist.png, and Zoom_norm_hist.png


P3_Real_Time_Image_Filtering.py allows you to open your webcam on a users phone or computer and allows you to switch between different filters (1-6). It allows a user to quit running the code by pressing "q" on their keyboard. 