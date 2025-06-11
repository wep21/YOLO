import unittest
from unittest.mock import MagicMock, patch, call
import torch

# Assuming yolo.config.config.Config exists and can be imported or mocked simply
# For simplicity, we'll create a mock Config on the fly.
# from yolo.config.config import Config

from yolo.tools.solver import TrainModel, ValidateModel, InferenceModel
from yolo.utils.bounding_box_utils import create_converter # Actual import for type hint or direct mock
from yolo.tools.loss_functions import create_loss_function # Actual import for type hint or direct mock


# Helper function to create a generic config mock
def create_mock_config():
    cfg = MagicMock()
    cfg.task.task = "train" # Default task
    cfg.task.epoch = 1 # Default epochs
    cfg.task.log_interval = 1
    cfg.task.run_validation_during_training = False # Default
    cfg.dataset.class_num = 80
    cfg.dataset.class_list = [f"class_{i}" for i in range(80)]
    cfg.model.name = "yolov8n"
    cfg.model.anchor = None # Or some mock value
    cfg.weight = None # No pretrained weights for tests
    cfg.image_size = [640, 640]

    cfg.task.optimizer.name = "Adam"
    cfg.task.optimizer.lr = 1e-3
    cfg.task.optimizer.weight_decay = 0.0
    cfg.task.scheduler.name = "Cosine"
    cfg.task.scheduler.warmup_epochs = 0

    cfg.task.loss.iou_type = "ciou"
    cfg.task.loss.beta_dist = False # Example, add other loss params as needed

    cfg.task.data.train_batch_size = 2 # Small batch size for tests
    cfg.task.data.val_batch_size = 2
    cfg.task.data.num_workers = 0 # No parallel workers for tests

    cfg.task.validation.nms.confidence_threshold = 0.5
    cfg.task.validation.nms.iou_threshold = 0.5
    cfg.task.validation.nms.max_detections = 100

    cfg.task.inference.nms.confidence_threshold = 0.5
    cfg.task.inference.nms.iou_threshold = 0.5
    cfg.task.inference.nms.max_detections = 100
    cfg.task.save_predict = True

    # Mock device attribute if accessed directly in solver, e.g. cfg.device
    cfg.device = "cpu" # For testing purposes

    return cfg

# Mock batch for dataloader
def create_mock_batch(batch_size=2, image_size=(3, 640, 640), num_classes=80, device='cpu'):
    images = torch.rand(batch_size, *image_size).to(device)
    # Targets format can vary, this is a simplified list of tensors
    targets = [torch.rand(5, 5).to(device) for _ in range(batch_size)] # 5 detections, (cls, x,y,w,h) or similar
    rev_tensor = torch.rand(batch_size, 2).to(device) # Example rev_tensor
    origin_frame = [MagicMock() for _ in range(batch_size)] # Mocked image frames (e.g. PIL Images)
    img_paths = [f"/fake/path/img_{i}.jpg" for i in range(batch_size)]
    return batch_size, images, targets, rev_tensor, img_paths


class TestTrainModelLoops(unittest.TestCase):

    @patch('yolo.tools.solver.create_dataloader')
    @patch('yolo.tools.solver.create_model')
    @patch('yolo.tools.solver.create_optimizer')
    @patch('yolo.tools.solver.create_scheduler')
    @patch('yolo.tools.solver.create_loss_function')
    @patch('yolo.tools.solver.create_converter')
    @patch('yolo.tools.solver.ValidateModel.validation_loop') # Mock validation loop
    @patch('yolo.tools.solver.logger') # Mock logger
    @patch('yolo.tools.solver.tqdm') # Mock tqdm
    def test_train_loop_basic_run(self, mock_tqdm, mock_logger, mock_validation_loop,
                                  mock_create_converter, mock_create_loss_fn,
                                  mock_create_scheduler, mock_create_optimizer,
                                  mock_create_model, mock_create_dataloader):

        cfg = create_mock_config()
        cfg.task.epoch = 2
        cfg.task.run_validation_during_training = True

        # Setup mocks
        mock_model_instance = MagicMock(spec=torch.nn.Module)
        mock_model_instance.device = torch.device('cpu')
        mock_model_instance.return_value = torch.rand(cfg.task.data.train_batch_size, 10) # Mock model output
        mock_create_model.return_value = mock_model_instance

        mock_dataloader_instance = [
            create_mock_batch(batch_size=cfg.task.data.train_batch_size, device='cpu'),
            create_mock_batch(batch_size=cfg.task.data.train_batch_size, device='cpu')
        ]
        mock_create_dataloader.return_value = mock_dataloader_instance
        mock_tqdm.return_value = mock_dataloader_instance # tqdm wraps the iterator

        mock_optimizer_instance = MagicMock()
        mock_create_optimizer.return_value = mock_optimizer_instance

        mock_scheduler_instance = MagicMock()
        mock_create_scheduler.return_value = mock_scheduler_instance

        mock_loss_fn_instance = MagicMock()
        mock_loss_fn_instance.return_value = (torch.tensor(0.5, requires_grad=True), {"loss_cls": 0.2, "loss_box": 0.3})
        mock_create_loss_fn.return_value = mock_loss_fn_instance

        mock_converter_instance = MagicMock()
        mock_converter_instance.return_value = torch.rand(cfg.task.data.train_batch_size, 5, 6) # Mock converter output
        mock_create_converter.return_value = mock_converter_instance

        # Instantiate TrainModel
        trainer = TrainModel(cfg)
        trainer.device = torch.device('cpu') # Ensure device is cpu for mocks
        trainer.model.to = MagicMock() # Mock to call on model
        trainer.loss_fn.to = MagicMock() # Mock to call on loss_fn

        # Call the training loop
        trainer.train_loop(epochs=cfg.task.epoch)

        # Assertions
        self.assertEqual(mock_tqdm.call_count, cfg.task.epoch) # tqdm called for each epoch
        mock_logger.info.assert_any_call(f"Epoch 1/{cfg.task.epoch}")
        mock_logger.info.assert_any_call(f"Epoch 2/{cfg.task.epoch}")

        # Model forward calls
        # Total batches = epochs * len(mock_dataloader_instance)
        self.assertEqual(mock_model_instance.call_count, cfg.task.epoch * len(mock_dataloader_instance))

        # Optimizer calls
        self.assertEqual(mock_optimizer_instance.zero_grad.call_count, cfg.task.epoch * len(mock_dataloader_instance))
        self.assertEqual(mock_optimizer_instance.step.call_count, cfg.task.epoch * len(mock_dataloader_instance))

        # Scheduler calls
        self.assertEqual(mock_scheduler_instance.step.call_count, cfg.task.epoch)

        # Loss function calls
        self.assertEqual(mock_loss_fn_instance.call_count, cfg.task.epoch * len(mock_dataloader_instance))

        # Converter calls (vec2box) - depends on model output structure. Assuming it's called for Main and AUX if dict.
        # If model output is not a dict with 'AUX' and 'Main', it's called once per batch.
        # The current train_loop calls vec2box on predicts["AUX"] and predicts["Main"] if they exist.
        # If predicts is not a dict, it calls vec2box once.
        # Mocking model output as simple tensor, so vec2box is called once.
        # If model_instance returns dict: {'AUX': ..., 'Main': ...}, then count would be 2*epochs*len(dataloader)
        # Current mock_model_instance returns a tensor, so it should be called once per batch.
        num_batches = cfg.task.epoch * len(mock_dataloader_instance)

        # If the model output is a dict with 'AUX' and 'Main' (change mock_model_instance.return_value for that)
        # mock_model_instance.return_value = {"AUX": torch.rand(cfg.task.data.train_batch_size, 10), "Main": torch.rand(cfg.task.data.train_batch_size, 10)}
        # self.assertEqual(mock_converter_instance.call_count, num_batches * 2)
        # Else (current case)
        self.assertEqual(mock_converter_instance.call_count, num_batches)


        # Validation loop calls
        self.assertEqual(mock_validation_loop.call_count, cfg.task.epoch)
        mock_validation_loop.assert_called_with(current_epoch=unittest.mock.ANY)

    @patch('yolo.tools.solver.create_dataloader')
    @patch('yolo.tools.solver.create_model')
    @patch('yolo.tools.solver.create_optimizer')
    @patch('yolo.tools.solver.create_scheduler')
    @patch('yolo.tools.solver.create_loss_function')
    @patch('yolo.tools.solver.create_converter')
    @patch('yolo.tools.solver.ValidateModel.validation_loop') # Mock validation loop
    @patch('yolo.tools.solver.logger') # Mock logger
    @patch('yolo.tools.solver.tqdm') # Mock tqdm
    def test_train_loop_no_validation_no_scheduler(self, mock_tqdm, mock_logger, mock_validation_loop,
                                                  mock_create_converter, mock_create_loss_fn,
                                                  mock_create_scheduler, mock_create_optimizer,
                                                  mock_create_model, mock_create_dataloader):
        cfg = create_mock_config()
        cfg.task.epoch = 1
        cfg.task.run_validation_during_training = False
        # How to signal no scheduler? Assume create_scheduler returns None if cfg.task.scheduler is None or specific name
        cfg.task.scheduler.name = None # Or some indicator that results in create_scheduler returning None

        # Setup mocks
        mock_model_instance = MagicMock(spec=torch.nn.Module)
        mock_model_instance.device = torch.device('cpu')
        mock_model_instance.return_value = torch.rand(cfg.task.data.train_batch_size, 10)
        mock_create_model.return_value = mock_model_instance

        mock_dataloader_instance = [create_mock_batch(batch_size=cfg.task.data.train_batch_size, device='cpu')]
        mock_create_dataloader.return_value = mock_dataloader_instance
        mock_tqdm.return_value = mock_dataloader_instance

        mock_optimizer_instance = MagicMock()
        mock_create_optimizer.return_value = mock_optimizer_instance

        mock_create_scheduler.return_value = None # No scheduler

        mock_loss_fn_instance = MagicMock()
        mock_loss_fn_instance.return_value = (torch.tensor(0.5, requires_grad=True), {"loss_total": 0.5})
        mock_create_loss_fn.return_value = mock_loss_fn_instance

        mock_converter_instance = MagicMock()
        mock_converter_instance.return_value = torch.rand(cfg.task.data.train_batch_size, 5, 6)
        mock_create_converter.return_value = mock_converter_instance

        trainer = TrainModel(cfg)
        trainer.device = torch.device('cpu')
        trainer.model.to = MagicMock()
        trainer.loss_fn.to = MagicMock()

        trainer.train_loop(epochs=cfg.task.epoch)

        # Assertions
        self.assertEqual(mock_model_instance.call_count, len(mock_dataloader_instance))
        self.assertEqual(mock_optimizer_instance.zero_grad.call_count, len(mock_dataloader_instance))
        self.assertEqual(mock_optimizer_instance.step.call_count, len(mock_dataloader_instance))

        # Scheduler assertions
        mock_create_scheduler.assert_called_once() # Ensure it was called to check for a scheduler
        # mock_scheduler_instance is None, so its step() method should not have been called.

        # Validation loop calls
        mock_validation_loop.assert_not_called() # Should not be called


class TestValidateModelLoops(unittest.TestCase):

    @patch('yolo.tools.solver.create_dataloader')
    @patch('yolo.tools.solver.create_model')
    @patch('yolo.tools.solver.create_converter')
    @patch('yolo.tools.solver.PostProcess') # Mock PostProcess class
    @patch('yolo.tools.solver.MeanAveragePrecision') # Mock MeanAveragePrecision class
    @patch('yolo.tools.solver.logger') # Mock logger
    @patch('yolo.tools.solver.tqdm') # Mock tqdm
    @patch('yolo.tools.solver.to_metrics_format') # Mock helper function
    def test_validation_loop_basic_run(self, mock_to_metrics_format, mock_tqdm, mock_logger,
                                       mock_map_class, mock_post_process_class,
                                       mock_create_converter, mock_create_model,
                                       mock_create_dataloader):
        cfg = create_mock_config()
        cfg.task.task = "validation" # Set task for ValidateModel specific configs

        # Setup mocks
        mock_model_instance = MagicMock(spec=torch.nn.Module)
        mock_model_instance.device = torch.device('cpu')
        # Mock model's forward method, which is called by self.model(images) in the loop
        mock_model_instance.return_value = torch.rand(cfg.task.data.val_batch_size, 10) # Mock raw model output
        mock_create_model.return_value = mock_model_instance

        # Mock for PostProcess instance
        mock_post_process_instance = MagicMock()
        # Mock return value for post_process_instance call
        mock_post_process_instance.return_value = [MagicMock()] * cfg.task.data.val_batch_size # List of processed predictions
        mock_post_process_class.return_value = mock_post_process_instance

        # Mock for MeanAveragePrecision instance
        mock_map_instance = MagicMock()
        mock_map_instance.compute.return_value = {"map": 0.75, "map_50": 0.9}
        mock_map_class.return_value = mock_map_instance

        mock_dataloader_instance = [
            create_mock_batch(batch_size=cfg.task.data.val_batch_size, device='cpu')
        ]
        mock_create_dataloader.return_value = mock_dataloader_instance
        mock_tqdm.return_value = mock_dataloader_instance # tqdm wraps the iterator

        mock_converter_instance = MagicMock() # vec2box
        mock_create_converter.return_value = mock_converter_instance

        # Mock for to_metrics_format helper
        mock_to_metrics_format.side_effect = lambda x: x # Pass through mock objects

        # Instantiate ValidateModel
        validator = ValidateModel(cfg)
        validator.device = torch.device('cpu')
        validator.model.to = MagicMock()
        validator.metric.to = MagicMock() # metric is MeanAveragePrecision instance

        # Call the validation loop
        metrics = validator.validation_loop(current_epoch=0)

        # Assertions
        self.assertEqual(mock_tqdm.call_count, 1)
        mock_logger.info.assert_any_call("Running validation for epoch 1")

        # Model related calls
        mock_model_instance.eval.assert_called_once()
        self.assertEqual(mock_model_instance.call_count, len(mock_dataloader_instance)) # Called for each batch
        mock_model_instance.train.assert_called_once() # Called at the end

        # PostProcess calls
        mock_post_process_class.assert_called_once_with(mock_converter_instance, cfg.task.validation.nms)
        self.assertEqual(mock_post_process_instance.call_count, len(mock_dataloader_instance))

        # Metric related calls
        mock_map_instance.reset.assert_called_once()
        # update called for each batch, with processed predicts and targets
        self.assertEqual(mock_map_instance.update.call_count, len(mock_dataloader_instance))
        mock_map_instance.compute.assert_called_once()
        self.assertEqual(metrics["map"], 0.75)

        # to_metrics_format calls (2 * num_batches because it's called for predicts and targets)
        self.assertEqual(mock_to_metrics_format.call_count, 2 * len(mock_dataloader_instance))

        # Check logging of metrics
        mock_logger.info.assert_any_call(f"Validation Metrics: {{'map': 0.75, 'map_50': 0.9}}")
        mock_logger.info.assert_any_call(f"PyCOCO/AP @ .5:.95: {0.75:.4f}, PyCOCO/AP @ .5: {0.9:.4f}")


class TestInferenceModelLoops(unittest.TestCase):

    @patch('yolo.tools.solver.create_dataloader')
    @patch('yolo.tools.solver.create_model')
    @patch('yolo.tools.solver.create_converter')
    @patch('yolo.tools.solver.PostProcess')
    @patch('yolo.tools.solver.draw_bboxes')
    @patch('yolo.tools.solver.Path') # To mock Path(...).mkdir and Path(...).save
    @patch('yolo.tools.solver.logger')
    @patch('yolo.tools.solver.tqdm')
    def test_inference_loop_basic_run_with_save(self, mock_tqdm, mock_logger, mock_path_class,
                                                mock_draw_bboxes, mock_post_process_class,
                                                mock_create_converter, mock_create_model,
                                                mock_create_dataloader):
        cfg = create_mock_config()
        cfg.task.task = "inference"
        cfg.task.save_predict = True # Ensure saving is enabled

        # Setup mocks for model and dataloader
        mock_model_instance = MagicMock(spec=torch.nn.Module)
        mock_model_instance.device = torch.device('cpu')
        mock_model_instance.return_value = torch.rand(cfg.task.data.val_batch_size, 10) # val_batch_size for inference too
        mock_create_model.return_value = mock_model_instance

        # Mock for PostProcess instance
        mock_post_process_instance = MagicMock()
        mock_post_process_instance.return_value = [MagicMock()] * cfg.task.data.val_batch_size
        mock_post_process_class.return_value = mock_post_process_instance

        # Mock for dataloader (predict_loader)
        # create_mock_batch returns: batch_size, images, targets, rev_tensor, origin_frame
        # For inference, targets might not be present or used.
        # The inference_loop expects: images, rev_tensor, origin_frame from batch
        mock_batch_data = create_mock_batch(batch_size=cfg.task.data.val_batch_size, device='cpu')
        # Adjust batch to match expected structure for inference_loop's predict_loader
        mock_inference_batch = (mock_batch_data[1], mock_batch_data[3], mock_batch_data[4]) # images, rev_tensor, origin_frame

        mock_dataloader_instance = [mock_inference_batch]
        mock_create_dataloader.return_value = mock_dataloader_instance
        mock_tqdm.return_value = mock_dataloader_instance

        mock_converter_instance = MagicMock()
        mock_create_converter.return_value = mock_converter_instance

        # Mock for draw_bboxes
        mock_drawn_image = MagicMock() # This would be like a PIL Image object
        mock_drawn_image.save = MagicMock() # Mock the save method on the image
        mock_draw_bboxes.return_value = mock_drawn_image

        # Mock for Path object
        mock_path_instance = MagicMock()
        mock_path_instance.mkdir = MagicMock()
        mock_path_instance.__truediv__ = lambda self, other: mock_path_instance # Path() / "filename"
        mock_path_class.return_value = mock_path_instance

        # Instantiate InferenceModel
        inferer = InferenceModel(cfg)
        inferer.device = torch.device('cpu')
        inferer.model.to = MagicMock()

        # Call the inference loop
        save_dir = "test_predictions"
        results = inferer.inference_loop(save_dir=save_dir)

        # Assertions
        self.assertEqual(mock_tqdm.call_count, 1)
        mock_logger.info.assert_any_call(f"Running inference, saving predictions to {save_dir}")

        mock_model_instance.eval.assert_called_once()
        self.assertEqual(mock_model_instance.call_count, len(mock_dataloader_instance))

        mock_post_process_class.assert_called_once_with(mock_converter_instance, cfg.task.inference.nms)
        self.assertEqual(mock_post_process_instance.call_count, len(mock_dataloader_instance))

        mock_draw_bboxes.assert_called_once()

        # Path and save assertions
        mock_path_class.assert_any_call(save_dir) # Path(save_dir)
        mock_path_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True) # For save_dir itself

        # _save_image related:
        # Path(save_dir) is called again inside _save_image, then / f"frame..."
        # The mock_path_instance is reused.
        self.assertTrue(mock_path_instance.mkdir.call_count >= 1) # Called in loop and possibly _save_image
        mock_drawn_image.save.assert_called_once() # Path(save_dir) / "frame..."
        mock_logger.info.assert_any_call(unittest.mock.string_containing("Saved visualize image at"))


        self.assertEqual(len(results), len(mock_dataloader_instance))
        mock_logger.info.assert_any_call("Inference complete.")

    @patch('yolo.tools.solver.create_dataloader')
    @patch('yolo.tools.solver.create_model')
    @patch('yolo.tools.solver.create_converter')
    @patch('yolo.tools.solver.PostProcess')
    @patch('yolo.tools.solver.draw_bboxes')
    @patch('yolo.tools.solver.Path')
    @patch('yolo.tools.solver.logger')
    @patch('yolo.tools.solver.tqdm')
    def test_inference_loop_no_save(self, mock_tqdm, mock_logger, mock_path_class,
                                    mock_draw_bboxes, mock_post_process_class,
                                    mock_create_converter, mock_create_model,
                                    mock_create_dataloader):
        cfg = create_mock_config()
        cfg.task.task = "inference"
        cfg.task.save_predict = False # Ensure saving is disabled

        mock_model_instance = MagicMock(spec=torch.nn.Module)
        mock_model_instance.device = torch.device('cpu')
        mock_model_instance.return_value = torch.rand(cfg.task.data.val_batch_size, 10)
        mock_create_model.return_value = mock_model_instance

        mock_post_process_instance = MagicMock()
        mock_post_process_instance.return_value = [MagicMock()] * cfg.task.data.val_batch_size
        mock_post_process_class.return_value = mock_post_process_instance

        mock_batch_data = create_mock_batch(batch_size=cfg.task.data.val_batch_size, device='cpu')
        mock_inference_batch = (mock_batch_data[1], mock_batch_data[3], mock_batch_data[4])
        mock_dataloader_instance = [mock_inference_batch]
        mock_create_dataloader.return_value = mock_dataloader_instance
        mock_tqdm.return_value = mock_dataloader_instance

        mock_converter_instance = MagicMock()
        mock_create_converter.return_value = mock_converter_instance

        mock_drawn_image = MagicMock()
        mock_drawn_image.save = MagicMock() # This save should NOT be called
        mock_draw_bboxes.return_value = mock_drawn_image

        mock_path_instance = MagicMock() # Path instance for save_dir
        mock_path_instance.mkdir = MagicMock()
        mock_path_class.return_value = mock_path_instance


        inferer = InferenceModel(cfg)
        inferer.device = torch.device('cpu')
        inferer.model.to = MagicMock()

        save_dir = "test_predictions_no_save"
        results = inferer.inference_loop(save_dir=save_dir)

        # Assertions
        mock_model_instance.eval.assert_called_once()
        self.assertEqual(mock_model_instance.call_count, len(mock_dataloader_instance))
        self.assertEqual(mock_post_process_instance.call_count, len(mock_dataloader_instance))
        mock_draw_bboxes.assert_called_once()

        # Path mkdir for save_dir itself is still called at the start of inference_loop
        mock_path_class.assert_any_call(save_dir)
        mock_path_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Save related assertions
        mock_drawn_image.save.assert_not_called() # Crucial check
        # Check that logger did not log about saving
        for call_arg in mock_logger.info.call_args_list:
            self.assertNotIn("Saved visualize image at", call_arg[0][0])

        self.assertEqual(len(results), len(mock_dataloader_instance))
        mock_logger.info.assert_any_call("Inference complete.")


if __name__ == '__main__':
    unittest.main()
