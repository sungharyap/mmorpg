"""Codes from https://github.com/lxulxu/WatermarkRemover"""

import glob
import sys
from pathlib import Path
from typing import Union

import cv2
import numpy


class WatermarkRemover:
    def __init__(self, threshold: int, kernel_size: int):
        self.threshold = threshold
        self.kernel_size = kernel_size

    def select_roi(self, img: numpy.ndarray, hint: str) -> list:
        COFF = 0.7
        w, h = int(COFF * img.shape[1]), int(COFF * img.shape[0])
        resize_img = cv2.resize(img, (w, h))
        roi = cv2.selectROI(hint, resize_img, False, False)
        cv2.destroyAllWindows()
        watermark_roi = [
            int(roi[0] / COFF),
            int(roi[1] / COFF),
            int(roi[2] / COFF),
            int(roi[3] / COFF),
        ]
        return watermark_roi

    def dilate_mask(self, mask: numpy.ndarray) -> numpy.ndarray:
        kernel = numpy.ones((self.kernel_size, self.kernel_size), numpy.uint8)
        mask = cv2.dilate(mask, kernel)
        return mask

    def generate_single_mask(
        self, img: numpy.ndarray, roi: list, threshold: int
    ) -> numpy.ndarray:
        if len(roi) != 4:
            print("NULL ROI!")
            sys.exit()

        roi_img = numpy.zeros((img.shape[0], img.shape[1]), numpy.uint8)
        start_x, end_x = int(roi[1]), int(roi[1] + roi[3])
        start_y, end_y = int(roi[0]), int(roi[0] + roi[2])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        roi_img[start_x:end_x, start_y:end_y] = gray[start_x:end_x, start_y:end_y]

        _, mask = cv2.threshold(roi_img, threshold, 255, cv2.THRESH_BINARY)
        return mask

    def generate_watermark_mask(self, video_path: Path) -> numpy.ndarray:
        video = cv2.VideoCapture(video_path)
        success, frame = video.read()
        roi = self.select_roi(frame, "select watermark ROI")
        mask = numpy.ones((frame.shape[0], frame.shape[1]), numpy.uint8)
        mask.fill(255)

        step = video.get(cv2.CAP_PROP_FRAME_COUNT) // 5
        index = 0
        while success:
            if index % step == 0:
                mask = cv2.bitwise_and(
                    mask, self.generate_single_mask(frame, roi, self.threshold)
                )
            success, frame = video.read()
            index += 1
        video.release()

        return self.dilate_mask(mask)

    def generate_subtitle_mask(self, frame: numpy.ndarray, roi: list) -> numpy.ndarray:
        mask = self.generate_single_mask(
            frame, [0, roi[1], frame.shape[1], roi[3]], self.threshold
        )
        return self.dilate_mask(mask)

    def inpaint_image(self, img: numpy.ndarray, mask: numpy.ndarray) -> numpy.ndarray:
        telea = cv2.inpaint(img, mask, 1, cv2.INPAINT_TELEA)
        return telea

    def remove_image_watermark(
        self, input_path: Union[str, Path], output_path: Union[str, Path]
    ):
        single_image = False
        if isinstance(input_path, str):
            if input_path.find("*") != -1:
                filenames = [Path(x) for x in glob.glob(input_path)]
            else:
                filenames = [Path(input_path)]
                single_image = True
        else:
            filenames = [input_path]
            single_image = True

        if isinstance(output_path, str):
            output_path = Path(output_path)

        if single_image and output_path.is_dir():
            print("input path is single image, output path should be a file")
            sys.exit()
        if not single_image and not output_path.is_dir():
            print("input path is multiple images, output path should be a dir")
            sys.exit()

        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)
        if output_path.is_dir():
            output_path.mkdir()

        mask = None
        for filename in filenames:
            mask = self.generate_watermark_mask(filename)
            img = cv2.imread(str(filename))
            img = self.inpaint_image(img, mask)

            if output_path.is_dir():
                output_filename = output_path / filename.name
            else:
                output_filename = output_path
            cv2.imwrite(str(output_filename), img)
