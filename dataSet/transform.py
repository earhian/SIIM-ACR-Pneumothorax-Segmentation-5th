from common import *


def random_scale(image, mask, size_=-1, center_p=1):
    if size_ == -1:
        sizes = [round(SIZE * 0.85), SIZE, round(SIZE * 1.15)]
        size_ = random.choice(sizes)
    image = cv2.resize(image, (size_, size_))
    mask = cv2.resize(mask, (size_, size_))

    if size_ == SIZE:
        return image, mask
    x_start = abs(SIZE - size_) // 2
    y_start = abs(SIZE - size_) // 2
    if random.random() <= (1 - center_p):
        x_start = random.randint(0, abs(SIZE - size_))
        y_start = random.randint(0, abs(SIZE - size_))
    if size_ > SIZE:
        return image[x_start:x_start + SIZE, y_start:y_start + SIZE], mask[x_start:x_start + SIZE,
                                                                      y_start:y_start + SIZE]
    else:
        image_zero = np.zeros((SIZE, SIZE))
        image_zero[x_start:x_start + size_, y_start:y_start + size_] = image
        mask_zero = np.zeros((SIZE, SIZE))
        mask_zero[x_start:x_start + size_, y_start:y_start + size_] = mask
        return image_zero.astype(np.uint8), mask_zero


def random_erase(image, mask, p=0.5):
    if random.random() < p:
        width, height = image.shape
        x = random.randint(0, width)
        y = random.randint(0, height)
        b_w = random.randint(5, 15)
        b_h = random.randint(5, 15)
        image[x:x + b_w, y:y + b_h] = 0
        mask[x:x + b_w, y:y + b_h] = 0
    return image, mask


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def random_angle_rotate(image, mask, angles=(-30, 30)):
    angle = random.randint(0, angles[1] - angles[0]) + angles[0]
    image = rotate(image, angle)
    mask = rotate(mask, angle)
    return image, mask


def do_gaussian_noise(image, sigma=0.5):
    gray = image.astype(np.float32) / 255
    H, W = gray.shape

    noise = np.random.normal(0, sigma, (H, W))
    noisy = gray + noise

    noisy = (np.clip(noisy, 0, 1) * 255).astype(np.uint8)
    return noisy


def do_color_shift(image, alpha0=0, alpha1=0, alpha2=0):
    image = image.astype(np.float32) + np.array([alpha0, alpha1, alpha2]) * 255
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def do_hue_shift(image, alpha=0):
    h = int(alpha * 180)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + h) % 180
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image


def do_speckle_noise(image, sigma=0.5):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray, a, b = cv2.split(lab)
    gray = gray.astype(np.float32) / 255
    H, W = gray.shape

    noise = sigma * np.random.randn(H, W)
    noisy = gray + gray * noise

    noisy = (np.clip(noisy, 0, 1) * 255).astype(np.uint8)
    lab = cv2.merge((noisy, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image


## illumination ====================================================================================

def do_brightness_shift(image, alpha=0.125):
    image = image.astype(np.float32)
    image = image + alpha * 255
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def do_brightness_multiply(image, alpha=1):
    image = image.astype(np.float32)
    image = alpha * image
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def do_contrast(image, alpha=1.0):
    image = image.astype(np.float32)
    gray = image * np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
    gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    image = alpha * image + gray
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


# https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def do_gamma(image, gamma=1.0):
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)  # apply gamma correction using the lookup table


def do_custom_process1(image, gamma=2.0, alpha=0.8, beta=2.0):
    image1 = image.astype(np.float32)
    image1 = image1 ** (gamma)
    image1 = image1 / image1.max() * 255

    image2 = (alpha) * image1 + (1 - alpha) * image
    image2 = np.clip(beta * image2, 0, 255).astype(np.uint8)

    image = image2
    return image


def do_clahe(image, clip=2, grid=16):
    grid = int(grid)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray, a, b = cv2.split(lab)
    gray = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(gray)
    lab = cv2.merge((gray, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return image
