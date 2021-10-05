import tensorflow as tf
import pathlib
import time
import os
from os.path import join
from glob import glob
from datetime import datetime
from matplotlib import pyplot as plt

# Path to the input images
PATH = "images"
ANNO_PATH = "annotations"
# The facade training set consist of 400 images
BUFFER_SIZE = 400
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1
# Each image is 256x256 in size
IMG_WIDTH = 128
IMG_HEIGHT = 128
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 1
ssim_loss_weight = 0.2
l1_loss_weight = 0.9
edge_loss_weight = 0.6

@tf.function
def load(image_file):
    depth_image_file = tf.strings.regex_replace(image_file, PATH, ANNO_PATH)

    # Read and decode an image file to a uint8 tensor
    input_image = tf.io.read_file(image_file)
    input_image = tf.image.decode_jpeg(input_image)

    depth_image = tf.io.read_file(depth_image_file)
    depth_image = tf.image.decode_jpeg(depth_image)

    # Convert both images to float32 tensors
    input_image = tf.cast(input_image, tf.float32)
    depth_image = tf.cast(depth_image, tf.float32)

    return input_image, depth_image

def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]

# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image):
    # Resizing to 286x286
    input_image, real_image = resize(input_image, real_image, int(IMG_HEIGHT*1.2), int(IMG_WIDTH*1.2))

    # Random cropping back to 256x256
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # Random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image

def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                    IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def Unet(width=IMG_WIDTH, height=IMG_HEIGHT, in_channel=INPUT_CHANNELS, out_channel=OUTPUT_CHANNELS):
    inputs = tf.keras.layers.Input(shape=[width, height, in_channel])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        # downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        # upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(out_channel, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def model_loss(pred, target):
    # Edges
    dy_true, dx_true = tf.image.image_gradients(target)
    dy_pred, dx_pred = tf.image.image_gradients(pred)
    weights_x = tf.exp(tf.reduce_mean(tf.abs(dx_true)))
    weights_y = tf.exp(tf.reduce_mean(tf.abs(dy_true)))

    # Depth smoothness
    smoothness_x = dx_pred * weights_x
    smoothness_y = dy_pred * weights_y

    depth_smoothness_loss = tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(
        abs(smoothness_y)
    )

    # Structural similarity (SSIM) index
    ssim_loss = tf.reduce_mean(
        1 - tf.image.ssim(
            tf.image.rgb_to_grayscale(target), pred, max_val=IMG_WIDTH, filter_size=3, k1=0.01 ** 2, k2=0.03 ** 2
        )
    )

    # Point-wise depth
    l1_loss = tf.reduce_mean(tf.abs(target - pred))

    loss = (
        (ssim_loss_weight * ssim_loss) +
        (l1_loss_weight * l1_loss) +
        (edge_loss_weight * depth_smoothness_loss)
    )

    # Mean absolute error
    # l1_loss = tf.reduce_mean(tf.abs(target - pred))

    return loss, l1_loss, ssim_loss, depth_smoothness_loss

@tf.function
def train_step(model, model_optimizer, input_image, target, step, summary_writer):
    with tf.GradientTape() as tape:
        model_output = model(input_image, training=True)
        loss, l1_loss, ssim_loss, depth_smoothness_loss = model_loss(model_output, target)

        model_gradients = tape.gradient(loss,
                                        model.trainable_variables)
        model_optimizer.apply_gradients(zip(model_gradients,
                                        model.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('loss', loss, step=step)
        tf.summary.scalar('l1_loss', l1_loss, step=step)
        tf.summary.scalar('ssim_loss', ssim_loss, step=step)
        tf.summary.scalar('depth_smoothness_loss', depth_smoothness_loss, step=step)

    return loss

def generate_images(model, test_input, tar, step, directory):
    prediction = model(test_input, training=True)

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input_Image', 'Ground_Truth', f'Predicted_Image_{step}']

    for d, t in zip(display_list, title):
        tf.keras.preprocessing.image.save_img(join(directory, f'{t}.jpg'), (d.numpy()+1.0)*127.5)

def display_progress(step, time, loss, total=1000, c = 100):
    s = step % total
    print('\r', end='')
    print(
        f'{step:6d}['+'='*((s)//c)+('='if (s+1)%total==0 else '>')+' '*((total-s-1)//c)+']', 
        end=f' Time: {time:3.2f}s, Avg. loss: {loss/(s if s>0 else 1):.5f}, Total loss: {loss:.5f}', 
        flush=True
    )

def fit(model, model_optimizer, train_ds, test_ds, steps, checkpoint_dir, summary_writer):
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    example_input, example_target = next(iter(test_ds.take(1)))
    start = time.time()
    loss_history=[]
    total_loss = 0
    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        step = step.numpy()
        if (step) % 1000 == 0:
            start = time.time()
            
            if len(loss_history)==10 and all(i >= total_loss for i in loss_history):
                print('-'*10,f'early stopping at step {step}','-'*10)
                break
            loss_history.append(total_loss)
            if len(loss_history) > 10:
                del loss_history[0]
            total_loss = 0
            print(f"Step {step}/{steps}")
            display_progress(step, 0, total_loss)

        total_loss += train_step(model, model_optimizer, input_image, target, step, summary_writer)

        # Training step
        if (step+1) % 100 == 0:
            display_progress(step, time.time()-start, total_loss)

        # Save (checkpoint) the model
        if (step+1) % 2000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            generate_images(model, example_input, example_target, step, checkpoint_dir)

if __name__ == "__main__":
    TRAINSET_SIZE = len(glob(join(PATH, "train/*.jpg")))
    print(f"The Training Dataset contains {TRAINSET_SIZE} images.")

    TESTSET_SIZE = len(glob(join(PATH, "test/*.jpg")))
    print(f"The Testing Dataset contains {TESTSET_SIZE} images.")

    train_dataset = tf.data.Dataset.list_files(join(PATH, 'train/*.jpg'))
    train_dataset = train_dataset.map(load_image_train,
                                        num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.list_files(join(PATH, 'test/*.jpg'))
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(BATCH_SIZE)


    model = Unet()
    # model.summary()
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=2.5e-4,
        decay_steps=10000,
        decay_rate=0.9
    )
    optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.5)
    checkpoint_dir = './training_checkpoints'
    
    log_dir="./logs/"
    summary_writer = tf.summary.create_file_writer(
        log_dir + 
        "fit/" + 
        datetime.now().
            strftime("%Y%m%d-%H%M%S")
    )
    fit(model, optimizer, train_dataset, test_dataset, 40000, checkpoint_dir, summary_writer)
    model.save(f'./models/model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5')
