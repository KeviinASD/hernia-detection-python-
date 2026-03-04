import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.transforms import functional as F
from PIL import Image
import argparse

# ---------------------------------------------------------
# 1. Adaptador del Dataset (YOLO a PyTorch)
# ---------------------------------------------------------
class YoloDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.img_dir = "/teamspace/studios/this_studio/dataset/train/images"
        self.label_dir = "/teamspace/studios/this_studio/dataset/train/labels"
        self.imgs = [img for img in sorted(os.listdir(self.img_dir)) if img.endswith(('.jpg', '.png', '.jpeg'))]
        
    def __getitem__(self, idx):
        # Cargar imagen
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        image_tensor = F.to_tensor(image)

        # Cargar etiquetas
        label_path = os.path.join(self.label_dir, self.imgs[idx].rsplit('.', 1)[0] + '.txt')
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    
                    # Convertir formato YOLO a [xmin, ymin, xmax, ymax] absoluto
                    xmin = (x_center - width / 2) * w
                    ymin = (y_center - height / 2) * h
                    xmax = (x_center + width / 2) * w
                    ymax = (y_center + height / 2) * h
                    
                    boxes.append([xmin, ymin, xmax, ymax])
                    # PyTorch requiere que el fondo sea la clase 0. 
                    # Tus clases ('disc', 'hdisc') serán 1 y 2.
                    labels.append(int(class_id) + 1)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        return image_tensor, target

    def __len__(self):
        return len(self.imgs)

# Necesario para manejar lotes con diferente cantidad de bounding boxes
def collate_fn(batch):
    return tuple(zip(*batch))

# ---------------------------------------------------------
# 2. Creación de Modelos
# ---------------------------------------------------------
def get_model(model_name, num_classes):
    if model_name == 'faster_rcnn':
        # Cargar modelo preentrenado con ResNet50
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # Reemplazar la cabecera para tus clases (2 clases + 1 fondo)
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model
        
    elif model_name == 'ssd':
        # Nota: PyTorch utiliza MobileNetV3 (más eficiente y moderno que V2) por defecto para SSD
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights='DEFAULT')
        in_channels = torchvision.models.detection._utils.retrieve_out_channels(model.backbone, (320, 320))
        num_anchors = model.anchor_generator.num_anchors_per_location()
        # Reemplazar la cabecera
        model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
        return model
    else:
        raise ValueError("Modelo no soportado. Elige 'faster_rcnn' o 'ssd'.")

# ---------------------------------------------------------
# 3. Bucle de Entrenamiento
# ---------------------------------------------------------
def train(args):
    # Detectar tu GPU automáticamente
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando el dispositivo: {device}")

    # Número de clases: 1 (Fondo) + 2 ('disc', 'hdisc') = 3
    num_classes = 3
    
    # Preparar Dataset y DataLoader
    dataset = YoloDataset(root_dir='.', split='train')
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # Cargar modelo al GPU
    model = get_model(args.model, num_classes)
    model.to(device)

    # Optimizador
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Entrenamiento
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        print(f"Época {epoch+1}/{args.epochs} | Pérdida (Loss) Total: {epoch_loss/len(data_loader):.4f}")

    # Guardar el modelo entrenado
    torch.save(model.state_dict(), f"{args.model}_lumbar_disc.pth")
    print(f"¡Entrenamiento finalizado! Modelo guardado como {args.model}_lumbar_disc.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento de Detección de Hernia de Disco")
    parser.add_argument('--model', type=str, required=True, choices=['faster_rcnn', 'ssd'], help='Modelo a entrenar')
    parser.add_argument('--epochs', type=int, default=10, help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=4, help='Tamaño del lote')
    args = parser.parse_args()
    train(args)