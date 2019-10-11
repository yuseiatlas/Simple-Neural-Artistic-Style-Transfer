# Simple-Neural-Artistic-Style-Transfer
A simple project to illustrate Neural Style Transfer based on [Convolutional neural networks for artistic style transfer](https://harishnarayanan.org/writing/artistic-style-transfer/).

## Prerequites: 
* [Python 3](https://www.python.org/downloads/) (3.7 in my case)
* A recent version of [NumPy](http://www.numpy.org/) and [SciPy](http://www.scipy.org/)
* [Keras](https://keras.io/)
* [matplotlib](https://matplotlib.org)

## How to use: 
After installing the required dependencies, you can run the the `main.py` file using:

    python main.py
If you don't have the [VGG16](https://keras.io/applications/#vgg16) pretrained model already dowloading it might start downloading it now. Once it's done it'll continue executing the code until it displays the result of the style shifting.

To use your own content and styles you can change the `content_image_path`

```python
# Here
content_image_path = 'images/content/neo.jpg'
content_image = Image.open(content_image_path)
content_image = content_image.resize((width, height))
```
and the `style_image_path`
```python
# Here
style_image_path = 'images/styles/scream.jpg'
style_image = Image.open(style_image_path)
style_image = style_image.resize((width, height))
```
Then run `main.py` to check the resutls :D

## Sample
The dimensions were `256x256`. It took 6 iterations for the loss to get below the 5% (the stopping condition in this case) taking ~50s per iteration on my Macbook Pro 2017. On a comparable machine with a an Nvidia 920M GPU it took ~30s per iteration.

Content                               |  Style                                   |  Result
:------------------------------------:|:-----------------------------------------:|:-----------------------------------------:
![](images/content/neo.jpg?raw=true)  |  ![](images/styles/edtaonisl.jpg?raw=true) |  ![](sample_result.png?raw=true)


## References:
* [Convolutional neural networks for artistic style transfer](https://harishnarayanan.org/writing/artistic-style-transfer)
* [How to Generate Art - Intro to Deep Learning #8](https://www.youtube.com/watch?v=Oex0eWoU7AQ&t)