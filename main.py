#importamos librerias
import cv2
import numpy as np
import face_recognition as fr
import os
import random
from datetime import datetime

#accedemos a la carpeta
path = 'fotos'
images = []
clases = []
lista = os.listdir(path)

#variables
comp1 = 100

for lis in lista:
    #leemos las imagenes de los rostros
    imgdb = cv2.imread(f'{path}/{lis}')
    #almacenamos imagen
    images.append(imgdb)
    #almacenamos nombre
    clases.append(os.path.splitext(lis)[0])

print(clases)

#funcion para codificar los rostros
def codrostros(images):
    listacod = []

    #iteramos
    for img in images:
        #correcion de color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #codificamos la imagen
        cod = fr.face_encodings(img)[0]
        #almacenamos
        listacod.append(cod)

    return listacod

def horario(nombre):
    #abrimos el archivo en modo lectura y escritura
    with open('Horario.csv','r+') as h:
        #leemos la info
        data = h.readline()
        #creamos lista de nombre
        listanombres = []

        #iteramos cada linea del doc
        for line in data:
            #buscamos la entrada y la diferenciamos con
            entrada = line.split(',')
            #almacenamos los nombres
            listanombres.append(entrada[0])

            #verificamos si ya hemos almacenado el nombre
        if nombre not in listanombres:
            #extraemos info actual
            info = datetime.now()
            #extraemos fecha
            fecha = info.strftime('%Y:%m:%d')
            #extraemos hora
            hora = info.strftime('%H:%M:%S')

            #guardamos la informacion
            h.writelines(f'\n{nombre},{fecha},{hora}')
            print(info)

#llamamos la funcion
rostroscod = codrostros(images)

#realizamos videocaptura
cap = cv2.VideoCapture(0)

#empezamos
while True:
    #leemos los fotogramas
    ret, frame = cap.read()
    #reducimos las imagenes para mejor procesamiento
    frame2 = cv2.resize(frame,(0,0),None,0.25,0.25)
    #conversion de color
    rgb = cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
    #buscamos los rostros
    faces = fr.face_locations(rgb)
    facescod = fr.face_encodings(rgb,faces)
    #iteramos
    for facecod, faceloc in zip(facescod,faces):
        comparacion = fr.compare_faces(rostroscod, facecod)
        #calculamos la similitud
        simi = fr.face_distance(rostroscod, facecod)
        #print(simi)
        #buscamos el valor mas bajo
        min = np.argmin(simi)

        if comparacion[min]:
            nombre = clases[min].upper()
            print(nombre)
            #extraemos coordenadas
            yi, xf,yf, xi = faceloc
            #escalamos
            yi,xf,yf,xi, = yi*4,xf*4,yf*4,xi*4
            indice = comparacion.index(True)
            #comparamos
            if comp1 != indice:
                #para dibujar cambiamos colores
                r = random.randrange(0,255,50)
                g = random.randrange(0,255,50)
                b = random.randrange(0,255,50)
                comp1 = indice
            if comp1 == indice:
                #dibujamos
                cv2.rectangle(frame,(xi,yi),(xf,yf),(r,g,b),3)
                cv2.rectangle(frame,(xi,yf-35),(xf,yf),(r,g,b),cv2.FILLED)
                cv2.putText(frame,nombre,(xi+6,yf-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                horario(nombre)

    #mostramos frames
    cv2.imshow("Reconocimiento Facial",frame)

    #leemos el teclado
    t = cv2.waitKey(5)
    if t == 27:
        break

