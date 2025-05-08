import onnx

model = onnx.load("./person_pose640x384.onnx")

print(onnx.helper.printable_graph(model.graph))