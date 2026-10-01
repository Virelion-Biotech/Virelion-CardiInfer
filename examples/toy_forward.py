from __future__ import annotations

import json
import os


payload = json.loads(os.environ["HEARTTWIN_PAYLOAD"])
parameters = payload["parameters"]
x = float(parameters["x"])
print(json.dumps({"outputs": {"y": [2.0 * x + 1.0], "energy": x * x}}))
