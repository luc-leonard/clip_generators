import os

import PIL.Image


def upscale(image_path: str, output_path: str) -> str:
    os.system(f'/home/lleonard/local/real_esrgan/realesrgan-ncnn-vulkan -i "{image_path}" -o "{output_path}"')
    return output_path

