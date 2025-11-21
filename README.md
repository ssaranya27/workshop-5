# **WORKSHOP-5**
# **AIM**

To develop a Python program using OpenCV that detects a vehicle’s license plate from an image by applying a Haar Cascade classifier, drawing bounding boxes, and extracting the plate region.

#  **ALGORITHM**

1. **Start**
2. Load the input image using `cv2.imread()`.
3. Convert the image to **grayscale** to simplify computation.
4. Apply preprocessing:

   * Gaussian Blur to reduce noise
   * Histogram Equalization to enhance contrast
5. Load the Haar Cascade file:
   `haarcascade_russian_plate_number.xml`
6. Apply `detectMultiScale()` on the preprocessed image to detect license plates.
7. For each detected plate:

   * Draw a bounding box on the original image
   * Crop the plate region
   * Save the cropped plate as an image
8. Display:

   * Original image
   * Preprocessed images
   * Detection output
9. **End**


#  **PROGRAM**

Copy this directly into your Jupyter Notebook.

```python
# ------------------------------------------
# LICENSE PLATE DETECTION USING HAAR CASCADE
# ------------------------------------------

import cv2
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Input Image
# -----------------------------
img_path = "car.jpg"          # your input image
cascade_path = "haarcascade_russian_plate_number.xml"

img = cv2.imread(img_path)

plt.figure(figsize=(6,6))
plt.title("Input Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# -----------------------------
# 2. Convert to Grayscale
# -----------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 3. Preprocessing
# -----------------------------
blur = cv2.GaussianBlur(gray, (5,5), 0)
eq = cv2.equalizeHist(blur)

# Display preprocessing
plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
plt.title("Gray")
plt.imshow(gray, cmap='gray')
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Blurred")
plt.imshow(blur, cmap='gray')
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Equalized")
plt.imshow(eq, cmap='gray')
plt.axis("off")

plt.show()

# -----------------------------
# 4. Load Cascade
# -----------------------------
plate_cascade = cv2.CascadeClassifier(cascade_path)

# -----------------------------
# 5. Detect Plates
# -----------------------------
plates = plate_cascade.detectMultiScale(eq, 1.1, 4)

print("Number of plates detected:", len(plates))

# -----------------------------
# 6. Draw Bounding Boxes
# -----------------------------
img_copy = img.copy()

for (x, y, w, h) in plates:
    cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0,255,0), 2)

plt.figure(figsize=(6,6))
plt.title("Detected Plates")
plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# -----------------------------
# 7. Crop & Save Plates
# -----------------------------
count = 1
for (x, y, w, h) in plates:
    roi = img[y:y+h, x:x+w]
    filename = f"plate_{count}.jpg"
    cv2.imwrite(filename, roi)
    print("Saved:", filename)

    plt.figure(figsize=(4,4))
    plt.title(f"Cropped Plate {count}")
    plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
    
    count += 1
```

# **OUTPUT**

<img width="1262" height="419" alt="image" src="https://github.com/user-attachments/assets/64861f2d-2867-4e6d-9312-6d0756babefc" />

<img width="813" height="187" alt="image" src="https://github.com/user-attachments/assets/0053492d-9f2d-416c-90d1-f7c4f4157105" />

<img width="624" height="435" alt="image" src="https://github.com/user-attachments/assets/7f07d1c6-a818-4677-889c-e3adef457d3a" />

<img width="490" height="279" alt="image" src="https://github.com/user-attachments/assets/baf1773c-c90b-4977-81a4-a527a5464771" />

**RESULT:**
The license plate of the vehicle was successfully detected using Haar Cascade, and the detected region was extracted and saved.

