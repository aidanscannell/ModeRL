#!/usr/bin/env python3
from mosvgpe.experts import SVGPExpert


class TwoExpertsList(list):
    def __init__(self, expert_one: SVGPExpert, expert_two: SVGPExpert):
        super().__init__()
        self.append(expert_one)
        self.append(expert_two)
