from core import worker, path
import time
import random


args = {
    "prompt": "beautiful woman smoking in a traing station",
    "negative_prompt": "",
    "style_selction": "cinematic-default",
    "performance_selction": "Quality",
    "aspect_ratios_selction": "1152x896",
    "image_number": 3,
    "image_seed": random.randint(1, 1024 * 1024 * 1024),
    "sharpness": 2.0,
    "base_model_name": path.default_base_model_name,
    "refiner_model_name": path.default_refiner_model_name,
    "l1": path.default_lora_name,
    "w1": path.default_lora_weight,
    "l2": "None",
    "w2": 0,
    "l3": "None",
    "w3": 0,
    "l4": "None",
    "w4": 0,
    "l5": "None",
    "w5": 0,
}

worker.buffer.append(args)
finished = False

while not finished:
    time.sleep(0.01)
    if len(worker.outputs) > 0:
        flag, product = worker.outputs.pop(0)
        if flag == "preview":
            percentage, title, image = product
            print(percentage, title, image)
        if flag == "results":
            print(product)
            finished = True
