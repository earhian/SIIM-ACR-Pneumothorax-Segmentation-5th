import numpy as np


def get_current_consistency_weight(epoch, consistency=10, consistency_rampup=5.0):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


# import numpy as np
def mask2rle(img, width, height):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel += 1;

    return " ".join(rle)


def rle2mask(rle, width, height, fill=255):
    mask = np.zeros(width * height)
    if rle == '-1': return np.zeros((width, height))
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position + lengths[index]] = fill
        current_position += lengths[index]

    return mask.reshape(width, height).T
def run_length_encode(component):
    if component.max() == 0: return '-1'
    component = component.T.flatten()
    start  = np.where(component[1: ] > component[:-1])[0]+1
    end    = np.where(component[:-1] > component[1: ])[0]+1
    length = end-start

    rle = []
    for i in range(len(length)):
        if i==0:
            rle.extend([start[0],length[0]])
        else:
            rle.extend([start[i]-end[i-1],length[i]])

    rle = ' '.join([str(r) for r in rle])
    return rle
