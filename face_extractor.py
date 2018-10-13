import face_recognition


def extract_face(img):
    face_locations = face_recognition.face_locations(img)
    assert len(face_locations) == 1
    face_locations, *_ = face_locations
    top, right, bottom, left = face_locations
    return img[top:bottom, left:right]
