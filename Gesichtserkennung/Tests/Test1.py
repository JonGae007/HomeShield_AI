from deepface import DeepFace
import cv2

# Lade ein Bild
img_path = "bild.jpg"
img = cv2.imread(img_path)

# Finde Gesichter im Bild
faces = DeepFace.detectFace(img_path, detector_backend = 'opencv')

# Lade die Bilder der bekannten Personen
known_face_names = ["Raik", "Jonas"]
known_face_imgs = ["Raik.jpg", "Jonas.jpg"]

# Überprüfe, ob das erkannte Gesicht mit den bekannten Gesichtern übereinstimmt
for i, known_img in enumerate(known_face_imgs):
    # Vergleiche das erkannte Gesicht mit jedem bekannten Gesicht
    result = DeepFace.verify(img_path, known_img, detector_backend = 'opencv')
    
    # Prüfe, ob es eine Übereinstimmung gibt
    if result["verified"]:
        print(f"Das Gesicht in {img_path} gehört zu {known_face_names[i]}")
        break
else:
    print("Keine Übereinstimmung gefunden")

# Optional: Anzeige des Bildes mit erkannten Gesichtern
cv2.imshow('Gesichter', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
