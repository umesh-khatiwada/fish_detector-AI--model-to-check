import torch
import torchvision

def load_model(model_path, device):
    model = torch.jit.load(model_path).to(device)
    # model.eval()
    return model

def run_inference(model, image_np, device):
    x = torch.from_numpy(image_np).to(device)
    with torch.no_grad():
        x = x.permute(2, 0, 1).float()
        y = model(x)
    to_keep = torchvision.ops.nms(y['pred_boxes'], y['scores'], 0.5)
    pred_boxes = y['pred_boxes'][to_keep].cpu().numpy()
    pred_classes = y['pred_classes'][to_keep].cpu().numpy()
    pred_scores = y['scores'][to_keep].cpu().numpy()
    return {
        'pred_boxes': pred_boxes,
        'pred_classes': pred_classes,
        'scores': pred_scores
    }
