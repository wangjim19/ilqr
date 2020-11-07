import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML

def make_video_fn(frames, flip=True):
    """
        frames : np.ndarray [ n_frames x height x width x channels ]
                 or list of length n_frames
        
        returns : fn that, when called, displays video in jupyter notebook
        
        adapted from https://stackoverflow.com/a/57275596
    """
    
    frames = np.array(frames)

    if True:
        ## flip frames up-down
        ## (should probably be done for all images rendered via MuJoCo)
        frames = np.flip(frames, axis=1)

    ## callbacks
    
    def _init():
        img.set_data(frames[0,:,:,:])

    def _animate(i):
        img.set_data(frames[i,:,:,:])
        return img

    fig = plt.figure()
    ax = plt.gca()
    
    img = plt.imshow(frames[0,:,:,:])
    ax.set_axis_off()
    plt.tight_layout()
    
    plt.close()
    
    n_frames = len(frames)
    anim = animation.FuncAnimation(fig, _animate, init_func=_init,
                                   frames=n_frames, interval=50)
    
    return lambda: HTML(anim.to_html5_video())

def mkdir(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

def save_video(filename, video_frames, fps=60, video_format='mp4'):
    """
        (very lightly) adapted from
            https://github.com/rail-berkeley/
            softlearning/blob/48393e5e645ff2f39d7dadb17956b6a75edee900/
            softlearning/utils/video.py
    """
    assert fps == int(fps), fps
    import skvideo.io
    mkdir(filename)

    skvideo.io.vwrite(
        filename,
        video_frames,
        inputdict={
            '-r': str(int(fps)),
        },
        outputdict={
            '-f': video_format,
            '-pix_fmt': 'yuv420p', # '-pix_fmt=yuv420p' needed for osx https://github.com/scikit-video/scikit-video/issues/74
        }
    )

