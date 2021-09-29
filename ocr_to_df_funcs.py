import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pytesseract
if 'linux' not in sys.platform:
    from PIL import ImageGrab
from IPython.display import display

def clipboard_to_image():  
    if 'linux' in sys.platform:
        fs = os.system('xclip -selection clipboard -t image/png -o > /tmp/clipboard.png')
        if fs !=0 :
            raise OSError('no image in clipboard')
        return cv2.imread('/tmp/clipboard.png', cv2.IMREAD_UNCHANGED)
    else:
        im = ImageGrab.grabclipboard()
        im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        return im
    
def get_tesseract_df(image):
    tesseract_df = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
    tesseract_df = \
    tesseract_df[(pd.notnull(tesseract_df['text']))
                 &(tesseract_df['text']!=' ')
                 &(tesseract_df['text']!='|')].reset_index(drop=True)
    return tesseract_df

def crunch_empty_to_lists(empty):
    lol = []
    for i, x in enumerate(empty):
        if i==0:
            l = [x]
        elif x - empty[i-1] == 1:
            l.append(x)
        else:
            lol.append(l)
            l = [x]
    lol.append(l)
    return lol

def get_significant_lines(empty, end):
    lol = crunch_empty_to_lists(empty)
    lol = [x for x in lol if x[0]!=0]
    lol = [x for x in lol if x[-1]!=(end-1)]
    return [int(np.median(x)) for x in lol]

def get_grid(image, tesseract_df, tight_parameter = 3, verbose=False, conf_th=60):
    tesseract_df['left_fixed'] = tesseract_df['left'] + tight_parameter
    tesseract_df['right'] = tesseract_df['left_fixed'] + tesseract_df['width'] - tight_parameter 
    tesseract_df['top_fixed'] = tesseract_df['top'] + tight_parameter
    tesseract_df['bottom'] = tesseract_df['top'] + tesseract_df['height'] - tight_parameter

    tesseract_df['h_range'] = [list(range(x[0], x[1])) for x in  zip(tesseract_df['left_fixed'], tesseract_df['right'])]
    tesseract_df['v_range'] = [list(range(x[0], x[1])) for x in  zip(tesseract_df['top_fixed'], tesseract_df['bottom'])]

    high_conf_df = tesseract_df[tesseract_df['conf']>=conf_th]
    vertical_filled = sum(high_conf_df['h_range'].values, [])
    horizontal_filled = sum(high_conf_df['v_range'].values, [])

    vertical_empty = [x for x in range(image.shape[1]) if x not in vertical_filled]
    horizontal_empty = [x for x in range(image.shape[0]) if x not in horizontal_filled]

    vertical_lines = get_significant_lines(vertical_empty, image.shape[0])
    horizontal_lines = get_significant_lines(horizontal_empty, image.shape[1])

    if verbose:
        plt.figure(figsize=(15,10))
        plt.vlines(vertical_lines, ymin=0, ymax=image.shape[0])
        plt.hlines(horizontal_lines, xmin=0, xmax=image.shape[1])

        plt.imshow(image)
        plt.show()

    return horizontal_lines, vertical_lines

def get_list_of_dfs_by_boundaries(tesseract_df, lines):
    lodf = []
    boundaries = [0] + lines
    for i, boundary in enumerate(boundaries):
        if i!=len(boundaries)-1:
            df00 = tesseract_df[tesseract_df['left'].between(boundary, boundaries[i+1])]
            if df00.shape[0]>0:
                lodf.append(df00)
    return lodf

def create_col_list(col_tesseract_df, horizontal_lines, i, end):
    l = []
    df0 = col_tesseract_df.copy()
    df0['bottom'] = df0['top']+df0['height']
    h_boundaries = [0] + horizontal_lines + [end]
    for i, boundary in enumerate(h_boundaries[:-1]):
        df00 = df0[(df0['top']>=boundary) & (df0['bottom']<=h_boundaries[i+1])]
        if df00.shape[0]==0:
            l.append(np.nan)
        else:
            l.append(' '.join(df00['text'].values))
        if pd.isnull(l[0]):
            l[0] = f'col_{i}'
    return l[0], l[1:]

def create_result_df(image, cols_lodf, horizontal_lines, header=True):
    result_df = pd.DataFrame()
    for i, df0 in enumerate(cols_lodf):
        first, rest = create_col_list(df0, horizontal_lines, i, image.shape[0])
        if header:
            result_df[first] = rest
        else:
            if first==f'col_{i}':
                first = np.nan
            result_df[f'col_{i}'] = first + rest
    return result_df

def image_to_df(image, header=True, verbose=False, conf_th=60):
    tesseract_df = get_tesseract_df(image)
    horizontal_lines, vertical_lines = get_grid(image, tesseract_df, verbose=verbose, conf_th=conf_th)
    cols_lodf = get_list_of_dfs_by_boundaries(tesseract_df, vertical_lines)
    
    if verbose:
        print(f'image shape: {image.shape}')
        print(f'num of vertical lines: {len(vertical_lines)}')
        print(f'num of horizontal lines: {len(horizontal_lines)}')
#         print()
#         print('tesseract_df:')
#         display(tesseract_df)

    
    return create_result_df(image, cols_lodf, horizontal_lines, header=header).dropna(how='all')

def image_path_to_df(path, header=True, verbose=False, conf_th=60):
    image = cv2.imread(path)
    return image_to_df(image, header=header, verbose=verbose, conf_th=conf_th)

def image_clipboard_to_df(header=True, verbose=False, conf_th=60):
    image = clipboard_to_image()
    return image_to_df(image, header=header, verbose=verbose, conf_th=conf_th)
