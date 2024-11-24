# utils.py

import numpy as np
import imageio
from PIL import Image
import pydicom
import os

class AnimateScans:
    def __init__(self, path, duration):
        self.path = path
        self.duration = duration

    def load_scan(self):
        slices = [pydicom.dcmread(os.path.join(self.path, f)) for f in os.listdir(self.path) if f.endswith('.dcm')]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        return slices

    def get_pixels_hu(self, scans):
        image = np.stack([s.pixel_array for s in scans])
        image = image.astype(np.int16)
        image[image == -2000] = 0
        
        intercept = scans[0].RescaleIntercept
        slope = scans[0].RescaleSlope
        
        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)
            
        image += np.int16(intercept)
        return np.array(image, dtype=np.int16)

    def set_lungwin(self, img, hu=[-1200., 600.]):
        lungwin = np.array(hu)
        newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
        newimg[newimg < 0] = 0
        newimg[newimg > 1] = 1
        newimg = (newimg * 255).astype('uint8')
        return newimg

    def show_animation(self, hu=[-1200., 600.], gif_path=None):
        """
        Create and save an animation from the DICOM scans.

        Parameters:
        hu (List[float]): Hounsfield unit range for lung windowing.
        gif_path (str): Path to save the GIF animation. If None, saves to '/tmp/temp_animation.gif'

        Returns:
        gif_path (str): Path to the saved GIF animation.
        """
        scans = self.load_scan()
        scan_array = self.get_pixels_hu(scans)
        lungwin_images = [self.set_lungwin(img, hu) for img in scan_array]

        if gif_path is None:
            gif_path = '/tmp/temp_animation.gif'

        imageio.mimsave(gif_path, lungwin_images, duration=self.duration)
        
        return gif_path