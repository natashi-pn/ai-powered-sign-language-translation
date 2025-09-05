import tensorflow as tf

Sequential = tf.keras.models.Sequential
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator


dataset_path = "C:\\everything from natashi\\Files\\Level 5 HND\\CRP\\Sign Language Project\\Sign-Language-To-Text-and-Speech-Conversion\\AtoZ_3.1" 
save_model_path = "C:\\everything from natashi\\Files\\Level 5 HND\\CRP\\Sign Language Project\\Sign-Language-To-Text-and-Speech-Conversion\\cnn9.h5"

img_height, img_width = 400, 400
batch_size = 16

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=10)

# Save the trained model
model.save(save_model_path)
print("Model saved at:", save_model_path)
