import os
import urllib.request

import numpy as np


def progress_print(block_count, block_size, total_size):
    """
    Callback function to display the progress
    (ref: https://qiita.com/jesus_isao/items/ffa63778e7d3952537db)

    Parameters
    ----------
    block_count:
    block_size:
    total_size:
    """
    percentage = 100.0 * block_count * block_size / total_size
    if percentage > 100:
        # Bigger than 100 does not look good, so...
        percentage = 100
    max_bar = 50
    bar_num = int(percentage / (100 / max_bar))
    progress_element = '=' * bar_num
    if bar_num != max_bar:
        progress_element += '>'
    bar_fill = ' '  # fill the blanks
    bar = progress_element.ljust(max_bar, bar_fill)
    total_size_kb = total_size / 1024
    print(f'[{bar} {percentage:.2f}% ( {total_size_kb:.0f}KB )]', end='\r')




def format_input_tensor(tensor, input_details, idx):
    details = input_details[idx]
    dtype = details['dtype']
    if dtype == np.uint8 or dtype == np.int8:
        quant_params = details['quantization_parameters']
        input_tensor = tensor / quant_params['scales'] + quant_params['zero_points']
        if dtype == np.int8:
            input_tensor = input_tensor.clip(-128, 127)
        else:
            input_tensor = input_tensor.clip(0, 255)
        return input_tensor.astype(dtype)
    else:
        return tensor


def get_output_tensor(interpreter, output_details, idx):
    details = output_details[idx]
    if details['dtype'] == np.uint8 or details['dtype'] == np.int8:
        quant_params = details['quantization_parameters']
        int_tensor = interpreter.get_tensor(details['index']).astype(np.int32)
        real_tensor = int_tensor - quant_params['zero_points']
        real_tensor = real_tensor.astype(np.float32) * quant_params['scales']
    else:
        real_tensor = interpreter.get_tensor(details['index'])
    return real_tensor
