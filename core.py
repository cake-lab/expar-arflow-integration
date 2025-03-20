"""Core library."""

import json
import base64
import arflow
import websocket
import numpy as np
import numpy.typing as npt

WS_URL = "ws://mountain-lion.catshome:5034/websocket"

metadata = {
    "intrinsics": [
        1363.1690673828125,
        1363.1690673828125,
        956.9332275390625,
        725.69940185546875,
        0,
    ],
    "image": "test_data/frame_0.jpg",
    "timestamp": 119035.448430708,
    "depth": "test_data/depth_0.bin",
    "objPosition": [0, 0, 0],
}


def unity_to_right_handed(matrix: npt.NDArray) -> npt.NDArray:
    """
    Convert a 4x4 transformation matrix from Unity's left-handed
    (Y-up) coordinate system to a right-handed system where
    Y is up, X is to the right, and Z points toward the viewer.

    Parameters
    ----------
    matrix : np.ndarray
        A 4x4 NumPy array representing the pose in Unity's coordinate system.

    Returns
    -------
    np.ndarray
        A 4x4 NumPy array representing the equivalent pose
        in the specified right-handed coordinate system.
    """
    # 1) Flip Z to convert from LH to RH
    flip_z = np.diag([1, 1, -1, 1])

    # 3) Combine: flip_z * M_Unity * flip_z => Right-handed
    #    Then multiply on the left by Rz(-90) to do the final rotation.
    return flip_z @ matrix @ flip_z


class CustomService(arflow.ARFlowService):
    def __init__(self):
        super().__init__()
        self.ws = websocket.create_connection(WS_URL)

        self.cam_intrinsics = None

    def on_register(self, request: arflow.RegisterRequest):
        """Called when a client registers."""
        print("Client registered!")

        obj_pos = request.camera_depth.data_type.split(",")
        obj_pos = [float(x) for x in obj_pos]
        # obj_pos = [0, 0, 0]

        x, y, z = obj_pos
        obj_transform = np.array(
            [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]
        )
        obj_transform = unity_to_right_handed(obj_transform)
        obj_pos = obj_transform[:3, 3].tolist()

        metadata["objPosition"] = obj_pos

        rx = request.camera_color.resize_factor_x
        ry = request.camera_color.resize_factor_y

        intrinsics = [
            request.camera_intrinsics.focal_length_x * rx,
            request.camera_intrinsics.focal_length_y * ry,
            request.camera_intrinsics.principal_point_x * rx,
            request.camera_intrinsics.principal_point_y * ry,
        ]
        self.cam_intrinsics = np.array(
            [
                [intrinsics[0], 0, intrinsics[2]],
                [0, intrinsics[1], intrinsics[3]],
                [0, 0, 1],
            ]
        )

        self.rgb_resolution = (
            request.camera_depth.resolution_x,
            request.camera_depth.resolution_y,
        )

        try:
            self.ws.send(
                json.dumps(
                    {
                        "type": "initialize",
                        "rgbResolution": (
                            request.camera_depth.resolution_x,
                            request.camera_depth.resolution_y,
                        ),
                        "depthResolution": (
                            request.camera_depth.resolution_x,
                            request.camera_depth.resolution_y,
                        ),
                        "intrinsics": intrinsics,
                    }
                )
            )
        except Exception as e:
            print(e)

    def on_frame_received(self, decoded_frame_data):
        """Called when a frame is received."""

        try:
            color_rgb: npt.NDArray = decoded_frame_data["color_rgb"]
            depth_img: npt.NDArray = decoded_frame_data["depth_img"]
            transform: npt.NDArray = decoded_frame_data["transform"]

            t = unity_to_right_handed(transform)
            t = np.transpose(t)  # VERY IMPORTANT

            metadata["pose4x4"] = t.flatten().tolist()
            metadata["resolution"] = self.rgb_resolution
            metadata["depthResolution"] = self.rgb_resolution

            rgb_img = base64.b64encode(color_rgb.tobytes()).decode("utf-8")
            depth_data = base64.b64encode(depth_img.tobytes()).decode("utf-8")

            frame_msg = {
                "type": "frame",
                "metadata": metadata,
                "rgbResolution": self.rgb_resolution,
                "rgbImage": rgb_img,
                "depthData": depth_data,
            }

            self.ws.send(json.dumps(frame_msg))
        except Exception as e:
            print(e)
