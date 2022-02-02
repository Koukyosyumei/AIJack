def l2(fake_gradients, received_gradients):
    distance = 0
    for f_g, c_g in zip(fake_gradients, received_gradients):
        distance += ((f_g - c_g) ** 2).sum()
    return distance


def cossim(fake_gradients, received_gradients):
    distance = 0
    pnorm_0 = 0
    pnorm_1 = 0
    for f_g, c_g in zip(fake_gradients, received_gradients):
        pnorm_0 = pnorm_0 + f_g.pow(2).sum()
        pnorm_1 = pnorm_1 + c_g.pow(2).sum()
        distance = distance + (f_g * c_g).sum()
    distance = 1 - distance / pnorm_0.sqrt() / pnorm_1.sqrt()
    return distance
