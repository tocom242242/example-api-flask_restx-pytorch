import torch
import torchvision.transforms as transforms
from flask import Flask
from flask_restx import Api, Resource
from PIL import Image
from werkzeug.datastructures import FileStorage

from model import Net

transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
model = Net()

app = Flask(__name__)
api = Api(app, doc="/doc/")

upload_parser = api.parser()
upload_parser.add_argument("file", location="files", type=FileStorage, required=True)


def pre_process(image):
    image = transform(image)
    return image


def post_process(model_output):
    return model_output


@api.route("/prediction")
@api.expect(upload_parser)
class ExampleResource(Resource):
    def post(self):
        args = upload_parser.parse_args()
        uploaded_file = args["file"]
        img = Image.open(uploaded_file).convert("RGB")
        image = pre_process(img)
        result = model(image.unsqueeze(0))

        pred = int(torch.argmax(result))
        return {"pred": pred}, 201


if __name__ == "__main__":
    app.run(debug=True)
