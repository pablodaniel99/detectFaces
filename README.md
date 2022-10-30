# Script "Detect Faces"

#### All props to Javier Finance (https://www.youtube.com/c/CreditosyR%C3%A1pidos) 


#### The first thing needed is this amount of libraries and also this folder architecture:

<p align="center" width="100%">
    <img width="30%" src="https://user-images.githubusercontent.com/116290888/198827711-0e5c6159-f23c-41e1-9e8e-89a827833731.png"> 
    <img width="20%" src="https://user-images.githubusercontent.com/116290888/198830283-3e9d3fd7-acb7-4f24-923f-6a9c9848cf6b.png"> 
</p>

#### And these are all the images in the folder "faces". But whoever use this script can put the images of ONLY one person that he wants, only with the condition that he has a second image so the algorithm can work.


![image](https://user-images.githubusercontent.com/116290888/198830451-c29e2091-9d49-4815-87d7-2c3052167c76.png)


#### Maybe you have some problems importing cv2, it is highly recommended to use Virtual Studio Code, why? Because usign this platform will make the installation of this package way more easy than using conda o pip, which not only could provoke an Error but may also last 2 hours.

#### The folders containe different images from famous people like Jeff Bezos or Elon Musk. I choose the image of elonj (elon.jpg) for the example, but if you want, you can use other image as long as you have it on your main folder and not only on the path "faces"

### First function: 

´´´python

def encode_faces():

    encoded = {}

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded
    
´´´

### Second function: 


### Third function:


### And finally we print the result:




### Result
#### For the example, I've chosen an image of Elon Musk:

<p align="center" width="100%">
    <img width="60%" src="https://user-images.githubusercontent.com/116290888/198840349-11132b5a-fca8-4354-b71a-a4ef9c07fae4.png"> 
</p>

#### If you want to detect the face of other person contained in the folder faces the only thin you have to do is to add another image of that person in particular and replace in the code (at line 100) the image "elon.jpg"

