import os
import time
import logging
from queue import Queue
from threading import Thread, Event
import argparse
import numpy as np
import cv2

# Функция для парсинга аргументов командной строки
def parse():
    logger.info("Parsing arguments")
    parser = argparse.ArgumentParser(description="Reading sensor data")
    parser.add_argument("camera_name", help="System camera name")
    parser.add_argument("resolution", help="Camera resolution")
    parser.add_argument("framerate", help="Framerate, frames per second")
    args = parser.parse_args()
    cam_width, cam_height = args.resolution.split('x')
    return args.camera_name, int(cam_width), int(cam_height), float(args.framerate)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%m-%y %H:%M:%S', filename='logs/log.txt')
logger = logging.getLogger(__name__)

class Sensor:
    def get(self):
        raise NotImplementedError("Subclasses must implement method get()")

class SensorX(Sensor):
    """Sensor X"""
    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0

    @staticmethod
    def sensor_x_work(stop_event: Event, delay: float, queue: Queue):
        sensor = SensorX(delay)
        while not stop_event.is_set():
            queue.put(sensor.get())

    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data

class SensorCam(Sensor):
    """Sensor Cam"""
    def __init__(self, name: str, width: int, height: int):
        self._name = name
        self._width = width
        self._height = height
        self._camera = cv2.VideoCapture(self._name)
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if not self._camera.isOpened():
            raise ValueError(f"Cannot find camera {name}")

    @staticmethod
    def sensor_cam_job(quit_event: Event, name: str, width: int, height: int, queue: Queue):
        sensor = SensorCam(name, width, height)
        while not quit_event.is_set():
            queue.put(sensor.get())

    def get(self) -> np.ndarray:
        ret, frame = self._camera.read()
        if ret:
            return cv2.resize(frame, (self._width, self._height))
        else:
            logger.error("Frame not caught! Check if the device is connected and the provided name is correct")
            os._exit(1)

    def __del__(self):
        self._camera.release()
        logger.info("Camera released")

class WindowImage:
    """Window Image"""
    def __init__(self, frames_per_second: float):
        self._frames_per_second = frames_per_second

    def show(self, img: np.ndarray):
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt

    def __del__(self):
        cv2.destroyWindow("Image")
        logger.info("Window destroyed")

class ImageProcessor:
    """Image Processor"""
    def __init__(self, sensor_x_queues: list[Queue], sensor_cam_queue: Queue):
        self._sensor_x_data = [0, 0, 0]
        self._sensor_x_queues = sensor_x_queues
        self._sensor_cam_data = np.zeros((640, 360, 3), dtype=np.uint8)
        self._sensor_cam_queue = sensor_cam_queue

    @staticmethod
    def get_last_data(prev_data, queue: Queue):
        data = prev_data
        while not queue.empty():
            data = queue.get()
        return data

    def get_frame(self):
        self._sensor_cam_data = ImageProcessor.get_last_data(self._sensor_cam_data, self._sensor_cam_queue)
        for i in range(3):
            self._sensor_x_data[i] = ImageProcessor.get_last_data(self._sensor_x_data[i], self._sensor_x_queues[i])
        text = f"X Sensor 1: {self._sensor_x_data[0]} X Sensor 2: {self._sensor_x_data[1]} X Sensor 3: {self._sensor_x_data[2]}"
        return cv2.putText(img=self._sensor_cam_data, text=text, org=(10, 30), fontFace=cv2.FONT_ITALIC,
                           fontScale=0.8, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

if __name__ == "__main__":
    logger.info("Starting up...")
    try:
        camera_name, cam_width, cam_height, framerate = parse()
    except Exception as exception:
        logger.exception("Exception! Parsing incomplete!!", exc_info=exception)
        exit(0)
    logger.info("Argument parsing done successfully")

    stop_event = Event()
    sensor_x_queues = [Queue(), Queue(), Queue()]
    sensor_cam_queue = Queue()

    sensor0_thread = Thread(target=SensorX.sensor_x_work, args=(stop_event, 0.01, sensor_x_queues[0]))
    sensor1_thread = Thread(target=SensorX.sensor_x_work, args=(stop_event, 0.1, sensor_x_queues[1]))
    sensor2_thread = Thread(target=SensorX.sensor_x_work, args=(stop_event, 1, sensor_x_queues[2]))

    logger.info("Starting thread for Sensor 0...")
    sensor0_thread.start()
    logger.info("Starting thread for Sensor 1...")
    sensor1_thread.start()
    logger.info("Starting thread for Sensor 2...")
    sensor2_thread.start()


    sensor_cam_thread = Thread(target=SensorCam.sensor_cam_job, args=(stop_event, camera_name, cam_width, cam_height, sensor_cam_queue))
    logger.info("Starting Sensor Cam thread...")
    sensor_cam_thread.start()

    logger.info("Starting Window Image...")
    window_image = WindowImage(framerate)
    logger.info("Starting Frame Assembly...")
    image_processor = ImageProcessor(sensor_x_queues, sensor_cam_queue)

    try:
        while True:
            window_image.show(image_processor.get_frame())
    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt detected, shutting down...")
        stop_event.set()
        sensor0_thread.join()
        sensor1_thread.join()
        sensor2_thread.join()
        sensor_cam_thread.join()
        logger.info("Shutdown complete")
