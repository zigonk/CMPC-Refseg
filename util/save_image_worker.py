import logging
import os
from queue import Queue
from threading import Thread
from time import time
import cv2

class SaveThread(Thread):
    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            # Get the work from the queue and expand the tuple
            save_path, im = self.queue.get()
            try:
                cv2.imwrite(save_path, im)                
            finally:
                self.queue.task_done()

class SaveImageWorker:
    def __init__(self):
        self.save_queue = Queue()
        self.save_thread = SaveThread(self.save_queue)
        self.save_thread.daemon = True
        self.save_thread.start()
    def save_image(self, save_path, im):
        self.save_queue.put((save_path, im))