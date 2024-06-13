import os



# source_mp4_dir = "outputs/UniAnimate_infer"
# target_gif_dir = "outputs/UniAnimate_infer_gif"

source_mp4_dir = "outputs/UniAnimate_infer_long"
target_gif_dir = "outputs/UniAnimate_infer_long_gif"

os.makedirs(target_gif_dir, exist_ok=True)
for video in os.listdir(source_mp4_dir):
     video_dir = os.path.join(source_mp4_dir, video)
     gif_dir = os.path.join(target_gif_dir, video.replace(".mp4", ".gif"))
     cmd = f'ffmpeg -i {video_dir} {gif_dir}'
     os.system(cmd)