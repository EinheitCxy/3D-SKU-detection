import json
import bson
import logging
import traceback
import cv2
import numpy as np 

logger = logging.getLogger(__name__)


def process(inputs):
    foo = [json.loads(item) for item in inputs.get("foo", [])]
    images = inputs.get("images", [])  # 这里面每个图片都是bytes


    return {
        "foo": [json.dumps(item) for item in foo],
        "bar": 1
    }
