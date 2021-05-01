import cv2 as cv
import functions
import os

cam = cv.VideoCapture(0)

file_name = "haarcascade_frontalface_alt2.xml"
classifier = face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')#Modelo para reconhecer faces


# Carregando dataframe com as imagens para treinamento
dataframe = functions.load_dataframe() 

# Dividindo conjuntos de treino e teste
X_train, X_test, y_train, y_test = functions.train_test(dataframe) 

# Modelo PCA para extração de features da imagem
pca = functions.pca_model(X_train) 

# Conjunto de treino com features extraídas
X_train = pca.transform(X_train) 

# Conjunto de teste com features extraídas
X_test = pca.transform(X_test) 

# Treinando modelo classificatório KNN
knn = functions.knn(X_train, y_train) 

# Rótulo das classificações
label = {
    0: "acho que nao...",
    1: "acho que sim..."
}

# Abrindo a webcam...
while True:
    # Lendo a imagem e extraindo frame
    status, frame = cam.read() 

    if not status:
       break

    if cv.waitKey(0) & 0xff == ord('q'):
        break
    
    # Transformando a imagem em escala de cinza
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Detectando faces na imagem
    faces = classifier.detectMultiScale(gray)

    # Iterando nas faces encontradas
    for x,y,w,h in faces:
        gray_face = gray[y:y+h, x:x+w] # Recortando região da face

        if gray_face.shape[0] >= 200 and gray_face.shape[1] >= 200:
            gray_face = cv.resize(gray_face, (160,160)) # Redimensionando
            vector = pca.transform([gray_face.flatten()]) # Extraindo features da imagem

            pred = knn.predict(vector)[0] #C lassificando a imagem
            classification = label[pred]

            # Desenhando retangulos em torno da face
            if pred == 0:
                cv.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 3)
                print("\a")
            elif pred == 1:
                cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
            
            # Escrevendo classificação e quantidade de faces vistas
            cv.putText(frame, classification, (x - 20,y + h + 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv.LINE_AA)
            cv.putText(frame, f"{len(faces)} rostos identificados",(20,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv.LINE_AA)

    #Mostrando o frame
    cv.imshow("Cam", frame)
