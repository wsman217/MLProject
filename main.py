import numpy as np
from canvas import canvas
from Project import model as md


def this_is_listen(pixels, can):
    pred = np.argmax(md.predict(model, pixels.reshape(-1, 784)))
    can.update_prediction_text(pred)


model = md.restore_model()
canv = canvas.Canvas().register_listener(this_is_listen).start()
