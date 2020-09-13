from detecto import core, utils, visualize



model = core.Model.load('model/model_weights.pth', ['rust'])

image = utils.read_image('dataset/train_images/typewriter-1248089__340.png')
predictions = model.predict(image)

# predictions format: (labels, boxes, scores)
labels, boxes, scores = predictions

print(labels)

print(boxes)

print(scores)

visualize.show_labeled_image(image, boxes, labels)
