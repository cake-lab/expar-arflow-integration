#!/usr/bin/env python3
"""A simple example of extending the ARFlow server."""

# TODO: increase RGB size.


import arflow
from core import CustomService


def main():
    arflow.create_server(CustomService, port=8500, path_to_save=None)


if __name__ == "__main__":
    main()
