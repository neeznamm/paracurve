# paracurve
### Parallel generation of humanlike mouse trajectories (based on [SapiAgent: A Bot Based on Deep Learning to Generate Human-Like Mouse Trajectories](https://ieeexplore.ieee.org/document/9530664))

1. Generate synthetic mouse actions using `create_bezier_actions_sequential.py` & `create_bezier_actions_parallel.py` 
2. Run `anomaly_detection_pyod.py`.

The latter extracts features from both the recorded human (/sapimouse_actions) and synthetic actions, then runs 6 different anomaly
detection algorithms trained on the human data and outputs AUC and EER scores for the actions generated in step 1. The goal here is
to provide a proof of concept that both approaches are equivalent in terms of their output (as can be seen by the similar scores).  