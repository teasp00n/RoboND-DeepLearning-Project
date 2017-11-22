# Follow Me Project

## Fully Convolutional Neural Network

### Encoding

The encoder portion is a convolution network that reduces to a deeper 1x1
convolution layer, rather than a flat fully connected layer that would be
used for basic classification of images. This difference has the effect of
preserving spacial information from the image.

This allows us to learn about features more complicated than just lines
or curves. For example we can begin to recognise things such a hand, nose, or
eyes.

This is particularly important for us as in order to follow our hero we need to
know where in our field of view they are.

Separable Convolutions is a technique that reduces the number of parameters
needed, thus increasing efficiency for the encoder network. The reduction
in parameters also helps defend against over fitting.

* `input_layer` is the input layer
* `filters` is the number of output filters (the depth)
* `kernel_size` is a number that specifies the (width, height) of the kernel
* `padding` is either "same" or "valid"
* `activation` is the activation function, like "relu" for example.

```python
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters, kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)

    output_layer = layers.BatchNormalization()(output_layer)
    return output_layer

def encoder_block(input_layer, filters, strides):
    return separable_conv2d_batchnorm(input_layer, filters, strides)
```

### Decoding

* `input_layer` is the input layer
* `row` is the upsampling factor for the rows of the output layer
* `col` is the upsampling factor for the columns of the output layer
* `output` is the output layer

The bilinear upsampling layer doesn't actually learn like transposed
convolutions but does help speed up performance.

```python
def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2, 2))(input_layer)
    return output_layer
```

Batch normalisation is the practice of normalising all the inputs to each layer
of the network rather than just the data at its initial entry. The name comes from
the technique using the current batch of data (see `batch_size`).

```python
def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                      padding='same', activation='relu')(input_layer)

    output_layer = layers.BatchNormalization()(output_layer)
    return output_layer
```

The full decoder block comprises of bilinear upsampling of the `small_ip_layer`
which is then concatenated with `large_ip_layer`. We then add a separable
convolutional layer to help learn spatial details from the input layers.

```python
def decoder_block(small_ip_layer, large_ip_layer, filters):
    upsampled = bilinear_upsample(small_ip_layer)
    merged = layers.concatenate([upsampled, large_ip_layer])
    output_layer = separable_conv2d_batchnorm(merged, filters)
    return output_layer
```

### FCN

My fully convolutional network consists of three encoder blocks, starting with
a depth of 32 and doubling each time. These attempt to capture the spatial
significance of the features present in the image. These are then connected to
three decoder blocks via a batch normalised 1x1 convolutional layer. The
decoder blocks upsample our deep layers back to a flatter representation
appropriate for classifying pixels in the input image.

![FCN](./fcn.png)

The image from the notes serves as a high level block diagram of my network.

```python
def fcn_model(inputs, num_classes):
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    en1 = encoder_block(inputs, 32, 2)
    en2 = encoder_block(en1, 64, 2)
    en3 = encoder_block(en2, 128, 2)

    cv1 = conv2d_batchnorm(en3, filters=16, kernel_size=1, strides=1)

    dc3 = decoder_block(cv1, en2, 32)
    dc2 = decoder_block(dc3, en1, 64)
    dc1 = decoder_block(dc2, inputs, 128)

    x = dc1
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)
```

### Hyper Parameters

```python
learning_rate = 0.001
batch_size = 20
num_epochs = 36
steps_per_epoch = 200
validation_steps = 50
workers = 2
```

The `learning_rate` was chosen to be what I considered to be a reasonable
value. I iterated the other values around it and didn't change it (permuting
too many of the hyper parameters with a dev-test cycle of up to an hour is not
very efficient). It is possible however that a value as small as the one I have
chosen would allow for getting stuck in a local minimum. In practice, testing
the order of magnitude of this value would be advisable.

I chose a batch size of 20 because I was constrained by the amount of memory I
had available on my gpu (GTX1050). I did some runs on another machine with a
GTX1070 where I used a `batch_size` of 30. The `batch_size` is the number of
training samples that get put through the network in a single pass.  If the
product of `batch_size * num_epochs * steps_per_epoch` is the same the model
will see the same total number of training samples and so should get similar
results.

`num_epochs` was determined by trial and error - initially I had this set to 50
however only produced a result of 39%. I had a look back over the previous
iterations to find one where stopping might produce a better result and thus
tried 36. For interests sake I actually trained models with up to 400 epochs
but didn't see any improvements in the IoU score (in fact I think we would be
getting into over fitting territory with that many epochs).

## Results

My model manages to achieve a final score of `40.77`, there is obviously
significant room for improvement which is discussed in the Future Enhancements
section below.

![IoU](./IoU.png)

We can see here that the model does a pretty good job of recognising the hero
when it is following with an IoU score of `88.04`

![Following Target](./following.png)

The model performs worse when we are patrolling around looking for the hero.
The validation for this scenario saw `68` false positives. Training over more
data with the hero in more poses would help reduce this however I don't think
it would be possible to remove all of the false positives with the masks
configured as they are as some features of the hero are shared by other people
and when viewed in isolation are indistinguishable.

![Patrolling](./patrolling.png)

Finally, when we are patrolling and the target is visible somewhere in the
frame we perform quite poorly with and IoU for the hero as `20.57` and;
> number true positives: 145, number false positives: 3, number false
> negatives: 156

The model could have benefited greatly from having a lot more data with the
hero placed further out in the image.

![Patrol Target Visible](./patrol_w_target.png)

This trained model is not reusable for following another object such as a dog
or a car. Images containing the desired objects would need to be captured and
masks applied before being fed into the network. The resultant model could then
be used to follow that new object.

## Future Enhancements

**One of the best improvements we could make to my model is to train it on a
larger, more comprehensive data set**. I actually attempted to capture more data
of my own but the drone ignored the hero in training mode.

Another enhancement would be to explore **adding more layers to further our
ability to recognise more subtle features of the hero however in this
particular instance, we are limited by the number of discernible features our
hero is rendered with in the simulator**. In a real life implementation the model
could learn to recognise the hero based on more subtle facial features etc
rather than the spatial and color heavy features in the simulator. Other students
actually pointed out that it is possible for the model to incorrectly classify a
hand in the frame as the hero if it has been trained on images with hands in the edges
with masks indicating it is the hero.

I didn't iterate values for the learning rate which obviously would have a huge
impact on the resulting model. **Without having tried a multitude of values for
the learning rate it is possible my model has ended up in a local minimum** and
the learning rate is not sufficient to overcome it.

