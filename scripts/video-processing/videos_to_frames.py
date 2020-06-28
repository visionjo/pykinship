from pathlib import Path
from os import system
import time
import multiprocessing as mp
import progressbar

CMD = "ffmpeg -i {} -vf  fps=25  {}/frame-%03d.jpg"


def process_video(f_clip, video, path_out):
    # Read video file
    dout = Path(str(f_clip).replace(str(video), str(path_out))).with_suffix("")
    dout.mkdir(parents=True, exist_ok=True)
    system(command.format(f_clip, dout))


def process_video_multiprocessing(group_number):
    # Read video file
    cap = cv.VideoCapture(file_name)

    cap.set(cv.CAP_PROP_POS_FRAMES, frame_jump_unit * group_number)

    # get height, width and frame count of the video
    width, height = (
        int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    )
    no_of_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    proc_frames = 0

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv.VideoWriter()
    output_file_name = "output_multi.mp4"
    out.open("output_{}.mp4".format(group_number), fourcc, fps, (width, height), True)
    try:
        while proc_frames < frame_jump_unit:
            ret, frame = cap.read()
            if not ret:
                break

            im = frame
            # Perform face detection on each frame
            _, bboxes = detectum.process_frame(im, THRESHOLD)

            # Loop through list (if empty this will be skipped) and overlay green bboxes
            for i in bboxes:
                cv.rectangle(im, (i[0], i[1]), (i[2], i[3]), (0, 255, 0), 3)

            # write the frame
            out.write(im)

            proc_frames += 1
    except:
        # Release resources
        cap.release()
        out.release()

    # Release resources
    cap.release()
    out.release()


def multi_process(num_processes=8):
    print("Video processing using {} processes...".format(num_processes))
    start_time = time.time()

    # Paralle the execution of a function across multiple input values
    p = mp.Pool(num_processes)
    p.map(process_video_multiprocessing, range(num_processes))

    end_time = time.time()

    total_processing_time = end_time - start_time
    print("Time taken: {}".format(total_processing_time))
    # print("FPS : {}".format(frame_count / total_processing_time))


# root = Path("../../data/fiw-mm")
root = Path.home()
# data = Path(f"{root}/FIDs-MM")
data = root / "VIDs-aligned-tp"
# image = Path(f"{data}/visual/image")
video = data  # / "visual/video"
path_out = Path(f"{data}/visual/video-frames")

# print("Video frame count = {}".format(frame_count))
# print("Width = {}, Height = {}".format(width, height))
num_processes = mp.cpu_count()
print("Number of CPU: " + str(num_processes))
# frame_jump_unit = frame_count // num_processes
multi_process()

cmd = ''.join(('ffmpeg -i "{}"'.format(fpath_input),
               ' -af "highpass=f={0}, lowpass=f={1}"'.format(highpass, lowpass),
               ' -loglevel "error" -n',
               ' "{}"'.format(fpath_output)))
proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
output = proc.communicate()[0].decode()

tasks_params = ((media_file, backup_dir, highpass, lowpass, num_passes)
                for media_file, backup_dir in zip(media_files, target_dirs))

with mp.Pool(cpu_count) as pool:
    for res in pool.imap_unordered(self._af_worker, tasks_params_list):
        print(res.output)
        < show
        spinner / percentage
        progress >
    for f_clip in video.glob("F????/MID*/v?????/*.mp4"):
        # dout = Path(str(f_clip).replace(str(video), str(path_out))).with_suffix("")
        # dout.mkdir(parents=True, exist_ok=True)
        # system(command.format(f_clip, dout))
        multi_process(num_processes, f_clip, video, path_out)

# start showing progress
with progress_bar() as p_bar:
    tasks_params = ((media_file, backup_dir, highpass, lowpass, num_passes)
                    for media_file, backup_dir in zip(media_files, target_dirs))

    with multiprocessing.Pool(cpu_count) as pool:
        for res in pool.imap_unordered(self._af_worker, tasks_params_list):
            if not quiet:
                p_bar.info_msg = res.output
            tasks_done += 1
            p_bar.progress = tasks_done / num_tasks * 100
