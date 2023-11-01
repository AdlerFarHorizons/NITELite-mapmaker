import cv2

R_ACCEPT = 0.75


def calc_ccoeff(img1, img2):

    assert img1.shape == img2.shape, 'Images must have the same shape.'

    r = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)

    return r


def assert_approx_equal(img1, img2, r_accept=R_ACCEPT):

    r = calc_ccoeff(img1, img2)

    assert r > r_accept
