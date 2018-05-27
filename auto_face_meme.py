import cv2
import face_recognition

DESTINATION_MEME = r"/home/avielniego/Downloads/WhatsApp Image 2018-05-27 at 23.58.32.jpeg"


def main():
    destination_meme_image = face_recognition.load_image_file(DESTINATION_MEME)
    top, right, bottom, left = face_recognition.face_locations(destination_meme_image)[0]
    top -= 20
    bottom += 20
    left -= 5
    right += 5
    source_image_path = r"/home/avielniego/Downloads/omer.jpeg"
    source_image = face_recognition.load_image_file(source_image_path)
    face_locations = face_recognition.face_locations(source_image)
    face_location = face_locations[0]
    face = source_image[face_location[0]: face_location[2], face_location[3]:face_location[1], :]
    streched_face = cv2.resize(face, (right - left, bottom - top))
    destination_meme_image[top:bottom, left:right, :] = streched_face
    rgb_result = cv2.cvtColor(destination_meme_image, cv2.COLOR_BGR2RGB)
    cv2.imshow("i", rgb_result)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
