
import cv2
import os
from processor import process_image



def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    video.release()
    yield None


def process_videos():
    fps = 24

    existing_image_gen = video_frame_generator('output/video.mp4')
    out_path = 'output/temp_video.mp4'
    fourcc = cv2.cv.CV_FOURCC(*'mp4v') # cv2.cv.CV_FOURCC(*'MP4V')
    video_out = cv2.VideoWriter(out_path, fourcc, fps, (360, 640))

    frame_num = 0
    for f in os.listdir('input/video'):
        image_gen = video_frame_generator(os.path.join('input/video', f))
        image = image_gen.next()
        try:
            existing_image = existing_image_gen.next()
        except:
            existing_image = None

        while image is not None:
            frame_num += 1
            print('Processing frame {}'.format(frame_num))
            if existing_image is None:
                image = cv2.resize(image, (360, 640))
                result = process_image(image, save_model=True)

                video_out.write(result)
            else:
                video_out.write(existing_image)

            image = image_gen.next()
            try:
                existing_image = existing_image_gen.next()
            except:
                pass

    video_out.release()


process_videos()