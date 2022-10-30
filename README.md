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

```python
"""
This fuction looks through the faces folder and encodes all
the faces, first, creating a empty tuple for the encoded images, 
encoding all the images in the folder

:return: dict of (name, image encoded)
"""
def encode_faces():

    encoded = {}

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded
    
```

### Second function: 
```python
"""
This fuction will encode a face given the file name

:return: the encoded face
"""

def solo_image_encoded(img):
    
    # We encode the image (face)
    face = fr.load_image_file("faces/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding
```
### Third function:
```python
"""
This fuction will find all of the faces in a given image and label
them if it knows what they are

:param im: str of file path
:return: list of face names
"""
def classify_face(im):

    # Key values of the funtion: faces encoded, the list of those faces and their names
    faces = encode_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    #loads an image from the specified file, which is the one with path or name "im" and the number 1 is use for bringing color to the image
    img = cv2.imread(im, 1)

    # The first method is for recognizes all the faces in the path, the second one, given an image, return the 128-dimension face encoding for each face in the image.
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face to point what we are detecting
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)
            # Draw a label with a name below the face, so we know WHO is been recognize
            cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)

    # Display the resulting image, where we will see a red square on the face of the person we are detecting.
    while True:

        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return face_names 
```


### And finally we print the result:

```python
# Print the result:
print(classify_face("elon.jpg"))
```

### Result
#### For the example, I've chosen an image of Elon Musk:

<p align="center" width="100%">
    <img width="60%" src="https://user-images.githubusercontent.com/116290888/198840349-11132b5a-fca8-4354-b71a-a4ef9c07fae4.png"> 
</p>

#### If you want to detect the face of other person contained in the folder faces the only thin you have to do is to add another image of that person in particular and replace in the code (at line 100) the image "elon.jpg"

