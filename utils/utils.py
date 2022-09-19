import cv2


def iou_calculater(pred, label):
    pred, label = pred.cpu().numpy()[0][0], label.cpu().numpy()[0][0]

    _, pred = cv2.threshold(pred, 0.5, 1, cv2.THRESH_BINARY)

    bitwise_and = cv2.bitwise_and(label, pred)
    bitwise_or = cv2.bitwise_or(label, pred)

    intersection = cv2.countNonZero(bitwise_and)
    union = cv2.countNonZero(bitwise_or)

    return intersection / union
