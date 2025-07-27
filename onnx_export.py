import torch, sys
from model import Model

checkpoint_path, out_path = sys.argv[1:]

checkpoint = torch.load(checkpoint_path, "cpu")
model = Model(8)
model.load_state_dict(checkpoint["model"])
model.eval()

torch.onnx.export(
    model,
    (torch.zeros((1, 200)), torch.zeros((1, 5), dtype=torch.long)),
    out_path,
    input_names=["field", "queue"],
    output_names=["pred"],
    opset_version=14,
)
