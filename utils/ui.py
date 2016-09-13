#!/usr/bin/python
# -*- coding: utf_8 -*-

import sys
import time
import math
import numpy as np


class ProgressBar(object):
    """Progress indicator that works within jupyter-notebooks
       without the need of having JavaScript activated.
       
    References:
        Taken and modified from: Keras:
        https://github.com/fchollet/keras/blob/master/keras/utils/generic_utils.py
    """
    def __init__(self, max_value, width=10):
        """Creates a progress indicator instance.
        Parameters
        ----------
        max_value: int
            The maximum progress value.
        width: int, optional
            The width of the progress bar.
        """
        self.width = width
        self.max_value = max_value
        self.sum_params = {}
        self.unique_params = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0

    def update(self, value, params=[]):
        """Updates the progress bar.
        Parameters
        ----------
        value: int
            The value of the progess.
        params: list(tuple(str, float)), optional
            List of parameters to show as a tuple of (name, value).
            The progress bar will display averages for these values.
        """
        for k, v in params:
            if k not in self.sum_params:
                self.sum_params[k] = [v * (value - self.seen_so_far), value - self.seen_so_far]
                self.unique_params.append(k)
            else:
                self.sum_params[k][0] += v * (value - self.seen_so_far)
                self.sum_params[k][1] += (value - self.seen_so_far)
        self.seen_so_far = value

        now = time.time()

        prev_total_width = self.total_width
        sys.stdout.write("\b" * prev_total_width)
        sys.stdout.write("\r")

        prog = float(value) / self.max_value
        bar = "{:3d}%▕".format(int(round(prog * 100)))
        
        prog_width = int(self.width * prog)
        prog_to_next_step = math.ceil((prog_width + 1) - self.width * prog)
        if prog_width > 0:
            bar += ('█' * (prog_width-1))
            if value == self.max_value:
                bar += '█'
            elif prog_to_next_step < 0.5:
                bar += '▒'
            else:
                bar += '░'
        bar += ('░' * (self.width - prog_width))
        bar += '▏'
        sys.stdout.write(bar)
        self.total_width = len(bar)

        if value:
            time_per_unit = (now - self.start) / value
        else:
            time_per_unit = 0
        eta = time_per_unit * (self.max_value - value)
        info = ''
        if value < self.max_value:
            info += '- ETA: %ds' % eta
        else:
            info += '- %ds' % (now - self.start)
        for k in self.unique_params:
            info += ' - %s:' % k
            if type(self.sum_params[k]) is list:
                avg = self.sum_params[k][0] / max(1, self.sum_params[k][1])
                if abs(avg) > 1e-3:
                    info += ' %.4f' % avg
                else:
                    info += ' %.4e' % avg
            else:
                info += ' %s' % self.sum_params[k]

        self.total_width += len(info)
        if prev_total_width > self.total_width:
            info += ((prev_total_width - self.total_width) * " ")

        sys.stdout.write(info)
        sys.stdout.flush()

        if value >= self.max_value:
            sys.stdout.write("\n")