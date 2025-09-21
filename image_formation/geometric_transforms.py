import cv2
import numpy as np
import matplotlib.pyplot as plt

#look at images in the images folder
orig = cv2.imread("images/original_image.jpg")
target = cv2.imread("images/transformed_image.jpg") 
if orig is None: raise FileNotFoundError("original_image.jpg not found")
h, w = orig.shape[:2]

#make canvas larger
Hc, Wc = int(h*2), int(w*2) 

# corners of the original image (top-left, top-right, bottom-right, bottom-left)
src = np.float32([[0,0], [w,0], [w,h], [0,h]])

#where the four corners of the original image map to
dst = np.float32([
    [0.36*Wc, 0.10*Hc],  # top-left
    [0.60*Wc, 0.22*Hc],  # top-right 
    [0.86*Wc, 0.97*Hc],  # bottom-right 
    [0.58*Wc, 0.86*Hc],  # bottom-left
])

#move matrix
Hmat = cv2.getPerspectiveTransform(src, dst)

#apply mapping and fix output size + make border black
result = cv2.warpPerspective(
    orig, Hmat, (Wc, Hc),
    borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0)
)

#print destination points
print(dst)

# RGB helper
def bgr_to_rgb_for_display(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# convert BGR to RGB
orig_rgb   = bgr_to_rgb_for_display(orig)
target_rgb = bgr_to_rgb_for_display(target)  
result_rgb = bgr_to_rgb_for_display(result)

# put pics next to each other
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
axes[0].imshow(orig_rgb);   axes[0].set_title("Original")
axes[1].imshow(target_rgb); axes[1].set_title("Provided Transformed")
axes[2].imshow(result_rgb); axes[2].set_title("Reverse-Engineered Result")

#hide axis ticks
for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()