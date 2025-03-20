#!/usr/bin/env python3
"""Replay AR data recorded by ARFlow."""

import arflow
import arflow.replay
from core import CustomService


def main():
    arflow.replay.ARFlowPlayer(CustomService, "path/to/frame_data.pkl")


if __name__ == "__main__":
    main()
