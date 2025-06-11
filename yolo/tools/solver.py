from math import ceil
from pathlib import Path
import torch
from torchmetrics.detection import MeanAveragePrecision
import logging
from tqdm import tqdm # Added for progress bar

from yolo.config.config import Config
from yolo.model.yolo import create_model
from yolo.tools.data_loader import create_dataloader
from yolo.tools.drawer import draw_bboxes
from yolo.tools.loss_functions import create_loss_function
from yolo.utils.bounding_box_utils import create_converter, to_metrics_format
from yolo.utils.model_utils import PostProcess, create_optimizer, create_scheduler

# Basic logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class BaseModel(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__() # Changed super call
        self.cfg = cfg # Store cfg
        self.model = create_model(cfg.model, class_num=cfg.dataset.class_num, weight_path=cfg.weight)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Device handling
        self.model.to(self.device)

    def forward(self, x): # This forward is for the underlying yolo model
        return self.model(x)


class ValidateModel(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        # self.cfg = cfg # Already in BaseModel
        if self.cfg.task.task == "validation":
            self.validation_cfg = self.cfg.task
        else:
            self.validation_cfg = self.cfg.task.validation
        self.metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy", backend="faster_coco_eval")
        self.metric.warn_on_many_detections = False
        self.val_loader = create_dataloader(self.validation_cfg.data, self.cfg.dataset, self.validation_cfg.task)
        # self.ema = self.model # EMA needs to be handled explicitly if required

        # Setup formerly in `setup` method
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )
        self.post_process = PostProcess(self.vec2box, self.validation_cfg.nms)
        self.metric.to(self.device)

    @torch.no_grad() # Disable gradients for validation
    def validation_loop(self, current_epoch=None):
        self.model.eval() # Set model to evaluation mode
        self.metric.reset()
        logger.info(f"Running validation for epoch {current_epoch+1}" if current_epoch is not None else "Running validation")
        pbar = tqdm(self.val_loader, desc="Validating")

        for batch_idx, batch in enumerate(pbar):
            batch_size, images, targets, rev_tensor, img_paths = batch # Assuming this structure
            images = images.to(self.device)
            targets = [t.to(self.device) for t in targets] # Assuming targets might be a list of tensors

            H, W = images.shape[2:] # Get image dimensions for post_process

            # Forward pass
            # Assuming self.model handles EMA if it was used (e.g. self.ema in old code)
            # If EMA is handled separately, this needs adjustment. For now, using self.model directly.
            predictions_raw = self.model(images)

            # Post-processing
            # The `image_size` argument for `post_process` might need to be dynamic if images can have variable sizes
            # For now, assuming it's handled or `[W,H]` is appropriate.
            predicts = self.post_process(predictions_raw, image_size=[W,H])

            # Update metrics
            # Ensure `to_metrics_format` is compatible with the new `predicts` and `targets`
            self.metric.update(
                [to_metrics_format(p) for p in predicts],
                [to_metrics_format(t) for t in targets]
            )

        epoch_metrics = self.metric.compute()
        # `classes` key might not exist or be needed depending on metric version/config
        if "classes" in epoch_metrics:
            del epoch_metrics["classes"]

        logger.info(f"Validation Metrics: {epoch_metrics}")
        # Example of more specific logging, similar to old log_dict calls
        if "map" in epoch_metrics and "map_50" in epoch_metrics:
            logger.info(f"PyCOCO/AP @ .5:.95: {epoch_metrics['map']:.4f}, PyCOCO/AP @ .5: {epoch_metrics['map_50']:.4f}")

        self.model.train() # Set model back to training mode if called during training
        return epoch_metrics


class TrainModel(ValidateModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        # self.cfg = cfg # Already in BaseModel
        self.train_loader = create_dataloader(self.cfg.task.data, self.cfg.dataset, self.cfg.task.task)

        # Setup formerly in `setup` method (related to loss)
        # vec2box is already initialized in ValidateModel's __init__
        self.loss_fn = create_loss_function(self.cfg, self.vec2box)
        self.loss_fn.to(self.device)

        # Optimizers and schedulers are now initialized here or passed to the training loop
        self.optimizer = create_optimizer(self.model, self.cfg.task.optimizer)
        self.scheduler = create_scheduler(self.optimizer, self.cfg.task.scheduler)

    def train_loop(self, epochs: int):
        self.model.train()
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Training")
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(pbar):
                batch_size, images, targets, *_ = batch
                images = images.to(self.device)
                targets = [t.to(self.device) for t in targets] # Assuming targets might be a list of tensors

                # Forward pass
                predicts = self.model(images) # Calls BaseModel's forward -> self.model(x)

                # Preprocess predictions for loss function
                # This part depends on how vec2box and loss_fn expect their inputs
                # Assuming vec2box is applied to model outputs before loss calculation
                # This might need adjustment based on actual yolo model output structure
                if isinstance(predicts, dict) and "AUX" in predicts and "Main" in predicts:
                    aux_predicts = self.vec2box(predicts["AUX"])
                    main_predicts = self.vec2box(predicts["Main"])
                else: # Handle cases where output is not dict or doesn't have AUX/Main
                    # This is a placeholder, actual handling might be more complex
                    # Or, the model might always return the dict structure expected by loss_fn
                    main_predicts = self.vec2box(predicts)
                    aux_predicts = main_predicts # Or handle appropriately if no aux preds

                loss, loss_item = self.loss_fn(aux_predicts, main_predicts, targets)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                # Logging
                log_items = {f"loss/{k}": v for k, v in loss_item.items()}
                log_items["loss/total_loss"] = loss.item()
                log_items["lr"] = self.optimizer.param_groups[0]["lr"]

                pbar.set_postfix(log_items)
                if batch_idx % self.cfg.task.log_interval == 0: # Assuming log_interval in cfg
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx}: {log_items}")

            avg_epoch_loss = epoch_loss / len(self.train_loader)
            logger.info(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss}")

            if self.scheduler:
                self.scheduler.step() # Or scheduler.step(avg_epoch_loss) if it's ReduceLROnPlateau

            # Optional: Run validation loop at the end of each epoch
            if hasattr(self, 'validation_loop') and self.cfg.task.get("run_validation_during_training", True):
                 # Pass current epoch for logging or checkpointing if needed
                self.validation_loop(current_epoch=epoch)


class InferenceModel(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        # self.cfg = cfg # Already in BaseModel
        # TODO: Add FastModel
        self.predict_loader = create_dataloader(cfg.task.data, cfg.dataset, cfg.task.task)

        # Setup formerly in `setup` method
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )
        self.post_process = PostProcess(self.vec2box, self.cfg.task.nms)

    @torch.no_grad() # Disable gradients for inference
    def inference_loop(self, save_dir="predictions", display_stream=False):
        self.model.eval() # Set model to evaluation mode
        logger.info(f"Running inference, saving predictions to {save_dir}")

        # Ensure save_dir exists
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        pbar = tqdm(self.predict_loader, desc="Inferring")
        all_results = [] # To store results if needed

        for batch_idx, batch in enumerate(pbar):
            # Assuming batch structure: images, rev_tensor, origin_frame
            # This might vary based on the specific data loader for prediction
            images, rev_tensor, origin_frame = batch
            images = images.to(self.device)

            # Forward pass
            predictions_raw = self.model(images)

            # Post-processing
            # rev_tensor might be used in post_process to scale bboxes to original image size
            predicts = self.post_process(predictions_raw, rev_tensor=rev_tensor)

            # Draw bounding boxes
            # Assuming origin_frame is a single image or a list of images from the batch
            # And cfg.dataset.class_list is accessible
            # This part might need adjustment if origin_frame is a batch of images
            # For simplicity, assuming one image at a time for drawing, or draw_bboxes handles batches
            if isinstance(origin_frame, list): # If batch of frames
                 img_to_draw_on = origin_frame[0] # Example: use first frame, or loop
            else: # Single frame
                 img_to_draw_on = origin_frame

            processed_img = draw_bboxes(img_to_draw_on, predicts, idx2label=self.cfg.dataset.class_list)

            fps = None # Placeholder for FPS calculation
            if display_stream and hasattr(self.predict_loader, "is_stream") and self.predict_loader.is_stream:
                # fps = self._display_stream(processed_img) # _display_stream logic needs to be defined/ported
                logger.info("Stream display is not implemented in this refactor.")
                pass


            if getattr(self.cfg.task, "save_predict", True): # Default to True if not specified
                self._save_image(processed_img, batch_idx, save_dir=save_dir)

            all_results.append({"predictions": predicts, "processed_image": processed_img, "fps": fps})

        logger.info("Inference complete.")
        return all_results

    # _save_image can remain, but self.trainer.default_root_dir needs replacement
    def _save_image(self, img, batch_idx, save_dir="predictions"):
        save_path = Path(save_dir)
        # save_path.mkdir(parents=True, exist_ok=True) # Already done in inference_loop
        save_image_path = save_path / f"frame{batch_idx:03d}.png"
        img.save(save_image_path)
        logger.info(f"💾 Saved visualize image at {save_image_path}")
