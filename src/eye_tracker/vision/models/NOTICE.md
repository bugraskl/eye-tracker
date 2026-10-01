# Bundled face models: licences and provenance

Eye Tracker ships three model files. They run entirely inside OpenCV (`cv2.dnn`
and `cv2.FaceDetectorYN`); no other inference runtime is used and nothing is
downloaded at run time. `scripts/fetch_models.py` recreates and verifies every
model file from the upstream sources below (`--check` verifies without network
access), and `eye_tracker.vision.backends.MODEL_FILES` pins the same checksums.

| File | Size (bytes) | SHA-256 | Licence |
| --- | ---: | --- | --- |
| `face_landmarks_detector.tflite` | 2 553 590 | `c7d54204ce0448474c7f3fa9af494787c0965cbdd6f20fc72867e43046bd43d5` | Apache-2.0 |
| `geometry_pipeline_metadata_landmarks.binarypb` | 19 376 | `bdbcda96dfcb7da883da124aaa2c55dee49770d934f0fcc71747f8c21bdc75b4` | Apache-2.0 |
| `face_detection_yunet_2023mar.onnx` | 232 589 | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | MIT |

The full licence texts ship with the models, in the `licenses` folder next to
this file. They cover only these model files; Eye Tracker itself is under its
own licence (`LICENSE` in the source tree).

| Licence text | Covers | SHA-256 |
| --- | --- | --- |
| [`licenses/LICENSE-APACHE-2.0.txt`](licenses/LICENSE-APACHE-2.0.txt) | `face_landmarks_detector.tflite`, `geometry_pipeline_metadata_landmarks.binarypb` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| [`licenses/LICENSE-YUNET.txt`](licenses/LICENSE-YUNET.txt) | `face_detection_yunet_2023mar.onnx` | `2ad92c7a6eb7aebede4e19f5ec4930c8bd9d614dbb8e75be1f740edae346734c` |

`LICENSE-APACHE-2.0.txt` is the Apache License, Version 2.0, exactly as
published at <https://www.apache.org/licenses/LICENSE-2.0.txt>.
`LICENSE-YUNET.txt` reproduces the YuNet model's licence notice (copyright line
and MIT permission notice) from
<https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE>.
`scripts/fetch_models.py --check` fails when either text is missing or altered.

## MediaPipe Face Landmarker (Apache-2.0)

Copyright Google LLC. Licensed under the Apache License, Version 2.0; a copy of
the licence is in [`licenses/LICENSE-APACHE-2.0.txt`](licenses/LICENSE-APACHE-2.0.txt)
(also at <https://www.apache.org/licenses/LICENSE-2.0>). Model card and
documentation: <https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker>.

Both files are unmodified members of the official `face_landmarker.task` bundle
(float16, version 1), a stored (uncompressed) zip archive:

- Upstream URL: <https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task>
- Bundle size: 3 758 596 bytes
- Bundle SHA-256: `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`

| Bundle member | Used | SHA-256 |
| --- | --- | --- |
| `face_landmarks_detector.tflite` | yes: 478 face landmarks incl. irises, face-presence score | `c7d54204ce0448474c7f3fa9af494787c0965cbdd6f20fc72867e43046bd43d5` |
| `geometry_pipeline_metadata_landmarks.binarypb` | yes: canonical face mesh and Procrustes landmark basis for head pose | `bdbcda96dfcb7da883da124aaa2c55dee49770d934f0fcc71747f8c21bdc75b4` |
| `face_detector.tflite` | no (YuNet finds faces instead) | `b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f` |
| `face_blendshapes.tflite` | no | `4f36dded049db18d76048567439b2a7f58f1daabc00d78bfe8f3ad396a2d2082` |

The MediaPipe *runtime* (the `mediapipe` Python package and `libmediapipe`) is
deliberately not used: it contains a usage-logging client that uploads to
Google servers. Only the model weights and geometry data are redistributed.

## YuNet face detector (MIT)

From the OpenCV Zoo, originally trained in libfacedetection.train by Shiqi Yu
and contributors. Copyright (c) 2020 Shiqi Yu <shiqi.yu@gmail.com>. Licensed
under the MIT License; the copyright and permission notice are in
[`licenses/LICENSE-YUNET.txt`](licenses/LICENSE-YUNET.txt) (upstream:
<https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE>).

- Upstream URL: <https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx>
- Documentation: <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet>
