#!/usr/bin/env python

# Copyright 2017 Vertex.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import importlib
import json
import os
import platform
import sys

import cv2
import imageio
import numpy as np
import pygame
import scipy.misc

# for backwards compat with opencv 2.x
CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4

CAP_WIDTH  = 640
CAP_HEIGHT = 480

class Input:
    def __init__(self, path, stop):
        self.path = path
        self.count = 0
        self.stop = stop

    def open(self):
        if self.path:
            self.cap = cv2.VideoCapture(self.path)
        else:
            self.cap = cv2.VideoCapture(0)
        self.cap.set(CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
        self.cap.set(CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

    def poll(self):
        if self.stop and self.count == self.stop:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        self.count += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        return frame

    def close(self):
        self.cap.release()


class Compositor:

    def __init__(self):
        pygame.font.init()
        self._font = pygame.font.SysFont("monospace", 14, bold=True)
        self._tgt_size = (CAP_WIDTH, CAP_HEIGHT)
        self._tgt = pygame.Surface(self._tgt_size)

    def process(self, frame, clock):
        self._tgt.fill((0, 0, 0))
        # Convert the image into a surface object we can blit to the pygame window.
        surface = pygame.surfarray.make_surface(frame)
        # Fit the image proportionally into the window.
        (tgt_width, tgt_height)    = self._tgt_size
        (img_width, img_height, _) = frame.shape
        result_size = (img_width, img_height)
        # If the image is larger than the window, in both dimensions, scale it
        # down proportionally so that it entirely fills the window.
        if img_width > tgt_width and img_height > tgt_height:
            hscale = float(tgt_width)  / float(img_width)
            vscale = float(tgt_height) / float(img_height)
            if hscale > vscale:
                result_size = (tgt_width, int(img_height * hscale))
            else:
                result_size = (int(img_width * vscale), tgt_height)
        # Center the image in the tgt, cropping if necessary.
        hoff = (tgt_width  - result_size[0]) / 2
        voff = (tgt_height - result_size[1]) / 2
        surface = pygame.transform.scale(surface, result_size)
        self._tgt.blit(surface, (hoff, voff))
        # Print some text explaining what we think the image contains, using some
        # contrasting colors for a little drop-shadow effect.
        # captions = [self.make_caption(x) for x in predictions]
        # for (i, caption) in enumerate(captions):
        #     self.blit_prediction(i, caption)
        # Print the FPS
        fps_text = 'FPS: {:3.1f}'.format(clock.get_fps())
        self.blit_text(fps_text, (8, self._tgt.get_height() - 24))
        return self._tgt

    def make_caption(self, prediction):
        (label_id, label_name, confidence) = prediction
        return label_name + " ({0:.0f}%)".format(confidence * 100.0)

    def blit_prediction(self, i, caption):
        self.blit_text(caption, (8, 18 * i))

    def blit_text(self, text, pos):
        self.blit_text_part(text, pos, -1, (110, 110, 240))
        self.blit_text_part(text, pos, 2, (0, 0, 100))
        self.blit_text_part(text, pos, 1, (100, 100, 255))
        self.blit_text_part(text, pos, 0, (240, 240, 110))

    def blit_text_part(self, caption, pos, offset, color):
        label = self._font.render(caption, 1, color)
        label_pos = (pos[0] + offset, pos[1] + offset)
        self._tgt.blit(label, label_pos)


class OutputScreen:

    def __init__(self):
        self._screen_size = (CAP_WIDTH, CAP_HEIGHT)
        pygame.display.init()
        self._screen = pygame.display.set_mode(self._screen_size)
        pygame.display.set_caption("Plaidvision")

    def close(self):
        pass

    def process(self, surface):
        surface = surface.convert(self._screen)
        self._screen.blit(surface, self._screen.get_rect())
        pygame.display.flip()


def loop(headless):
    if headless:
        return True
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int)
    parser.add_argument('--input')
    args = parser.parse_args()

    clock = pygame.time.Clock()

    input = Input(args.input, args.frames)
    o = OutputScreen()
    compositor = Compositor()

    inference_clock = pygame.time.Clock()
    try:
        input.open()
        while loop(False):
            clock.tick()
            frame = input.poll()
            if frame is None:
                break
            inference_clock.tick()
            surface = compositor.process(frame, clock)
            # output.process(surface)
            o.process(surface)
            # for output in outputs:
            #   output.process(surface)
    except KeyboardInterrupt:
        pass
    except Exception as ex:
        raise
    finally:
        # for output in outputs:
        #   output.close()
        o.close()
        input.close()

if __name__ == "__main__":
    main()


"""
# -*- coding: utf-8 -*-
import pygame
from pygame.locals import *
import sys

def main():
    pygame.init() # 初期化
    screen = pygame.display.set_mode((600, 400)) # ウィンドウサイズの指定
    pygame.display.set_caption("Pygame Test") # ウィンドウの上の方に出てくるアレの指定

    while(True):
        screen.fill((255,63,10,)) # 背景色の指定。RGBだと思う
        pygame.display.update() # 画面更新

        for event in pygame.event.get(): # 終了処理
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

if __name__ == "__main__":
    main()
"""
