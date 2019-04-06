# Buy Less, Choose Well! Fashion Item Classification
### Task ###
Preditct whether the images are belong to which kind of categories.<br>
The images data have already transform into pixel value from 0(white) to
255(black) in grayscale.
### Kaggle Link ###
[Kaggle](https://ppt.cc/fBUC8x "link")

### Data Discription ###
There are 784 attributes for one instance, which represesnt the grayscale of
the 28x28 image<br>
Pixel: from 0(white) to 255(Black)

### Evaluation ###
It's evalutaed by **Mean F1-Score**<br>
 $$F1 \ score = 2\frac{precision*recall}{precision+recall}$$, where<br>
 $$precision = \frac{True \ Positive}{True \ Positive+False \ Positive}$$ 
 $$recall = \frac{True \ Positive}{True \ Positive+False \ Negative}$$ 