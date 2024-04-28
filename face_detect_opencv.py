import cv2 as cv    # import library OpenCV dengan alias 'cv'
import argparse     # import library argparse untuk command-line argument parsing (parser)

def detectAndDisplay(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # konversi citra Blue, Green, Red ke Grayscale
        # tujuan mengkonversi citra ke grayscale: 
        # menyederhanakan pemrosesan citra dan mengurangi computational complexity (dibandingkan 3 channel BGR)
    gray = cv.equalizeHist(gray) # penyetaraan histogram 
    # cara kerja: untuk setiap range value warna dari 0 sampai 255, akan diratakan distribusi warnanya 
    #   sehingga meningkatkan contrast dari bagian-bagian fitur yang penting, 
    #   yang membuatnya lebih bisa dibedakan (misal: tepian hidung, wajah)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5, minSize=(40, 40)) 
    # Menggunakan cascade classifier untuk mendeteksi wajah dalam citra grayscale. 
    # Parameter seperti scaleFactor, minNeighbors, dan minSize digunakan untuk menyesuaikan sensitivitas dan akurasi deteksi.
    # Scale Factor : Membantu untuk melakukan deteksi wajah pada dengan cara melakukan zoom dengan nilai tertentu untuk dapat membantu pengecekan agar lebih detail
    # Min Neighbors : Melakukan pembatasan neighbors yang ada pada objek yang sudah ditandai sebagai wajar (membantu agar pengecekan bahwa objek yang di tandai itu adalah wajah)
    # Min Size : Melakukan pembatasan pixel yang akan di cek sebagai wajah 
    for (x, y, w, h) in faces: #Iterasi melalui setiap wajah yang terdeteksi dalam bingkai gambar.
        center = (x + w//2, y + h//2) # Menghitung titik tengah dari setiap wajah yang terdeteksi.
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4) #Menggambar elips pada setiap wajah yang terdeteksi untuk menandainya.
        eyes = eyes_cascade.detectMultiScale(gray[y:y+h, x:x+w]) # Mendeteksi mata di dalam setiap wajah yang terdeteksi. Posisi deteksi mata dibatasi oleh area yang berada di dalam batas wajah yang terdeteksi.
        for (x2, y2, w2, h2) in eyes: # Iterasi melalui setiap mata yang terdeteksi dalam wajah.
            eye_center = (x + x2 + w2//2, y + y2 + h2//2) # Menghitung titik tengah dari setiap mata yang terdeteksi.
            radius = int(round((w2 + h2)*0.25)) # Menghitung radius lingkaran yang akan digambar di sekitar mata.
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0), 4) # Menggambar lingkaran pada setiap mata yang terdeteksi untuk menandainya.
    cv.imshow('Capture - Face detection', frame) # Menampilkan frame gambar yang telah dimodifikasi dengan deteksci wajah dan mata menggunakan jendela OpenCV dengan judul "Capture - Face detection".

parser = argparse.ArgumentParser(description='Face Detection with Cascade')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='haarcascade_frontalface_default.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera device number.', type=int, default=0)
args = parser.parse_args()

face_cascade, eyes_cascade = cv.CascadeClassifier(), cv.CascadeClassifier()
if not all(cascade.load(cv.samples.findFile(args.face_cascade if i == 0 else args.eyes_cascade)) for i, cascade in enumerate((face_cascade, eyes_cascade))):
    print('--(!)Error loading cascade')
    exit(0)

cap = cv.VideoCapture(args.camera)
if not cap.isOpened():
    print('--(!)Error opening video capture')
    exit(0)

while True: # looping
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    if cv.waitKey(10) == 27:
        break