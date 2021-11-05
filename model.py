import tensorflow as tf


cifar_ds = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar_ds.load_data()

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(128)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


class CifarModel(tf.keras.Model):
    def __init__(self):
        super(CifarModel, self).__init__()

        self.num_classes = 10
        self.train_ds = None
        self.test_ds = None

        self.rescaling = tf.keras.layers.Rescaling(1./255)

        self.conv1 = tf.keras.layers.Conv2D(filters=8,
                                            kernel_size=3,
                                            activation=tf.nn.relu,
                                            padding="same", use_bias=True)

        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same")

        self.conv2 = tf.keras.layers.Conv2D(filters=16,
                                            kernel_size=3,
                                            activation=tf.nn.relu,
                                            padding="same", use_bias=True)

        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same")

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256, activation=tf.nn.tanh, use_bias=True)
        self.dense2 = tf.keras.layers.Dense(512, activation=tf.nn.tanh, use_bias=True)
        self.output_layer = tf.keras.layers.Dense(self.num_classes, activation=tf.nn.gelu)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    def call(self, x):
        x = self.rescaling(x)

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.dense2(x)

        return self.output_layer(x)

    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self(images, training=True)
            loss = self.loss_object(labels, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    def test_step(self, images, labels):
        predictions = self(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    def train(self, epochs=5):
        if self.train_ds is None or self.test_ds is None:
            print("None datasets!")
            return

        for epoch in range(epochs):
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

            for images, labels in self.train_ds:
                self.train_step(images, labels)

            for images, labels in self.train_ds:
                self.test_step(images, labels)

            print(
                f'Epoch {epoch + 1}\n'
                f'Loss: {self.train_loss.result()}\n'
                f'Accuracy: {self.train_accuracy.result()}\n'
                f'Test Loss: {self.test_loss.result()}\n'
                f'Test Accuracy: {self.test_accuracy.result()}\n'
            )

            self.save_weights("./checkpoints/epoch" + str(epoch) + ".h5")


model = CifarModel()

model.train_ds = train_ds
model.test_ds = test_ds

model.train(epochs=100)
model.save("./cifar10_2")
