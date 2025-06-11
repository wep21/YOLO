import sys
from pathlib import Path
import torch # For torch.set_float32_matmul_precision if needed

import hydra

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.tools.solver import InferenceModel, TrainModel, ValidateModel
from yolo.utils.logging_utils import setup # setup might still be useful for save_path or other setup


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    # Call setup, it might initialize loggers or create save_path used by new loops indirectly or directly
    # callbacks and loggers returned by setup were for Lightning, may not be directly used now.
    _callbacks, _loggers, save_path = setup(cfg)

    # Common settings based on former Trainer options - apply if necessary
    # Example: precision="16-mixed" could map to torch.autocast or similar if doing mixed precision manually
    # For now, this is not implemented, assuming model handles its own precision or uses default.
    # if cfg.get("precision") == "16-mixed":
    #     torch.set_float32_matmul_precision('medium') # or 'high', requires PyTorch 1.12+
        # AMP (Automatic Mixed Precision) would typically be used within the training loop itself.

    # deterministic = True -> This should be handled by setting seeds if required (e.g. torch.manual_seed)
    # This was a Lightning setting, manual seed setting would be needed if strict determinism is paramount.
    if cfg.get("deterministic", False): # Assuming a new top-level 'deterministic' flag in config
        torch.manual_seed(cfg.get("seed", 42)) # Assuming a seed in config
        # Potentially add torch.cuda.manual_seed_all and other settings for full determinism

    # enable_progress_bar is handled by tqdm within the loops themselves.

    if cfg.task.task == "train":
        model = TrainModel(cfg)
        model.to(model.device) # Ensure model is on the correct device
        epochs = getattr(cfg.task, "epoch", 1) # Default to 1 epoch if not specified
        model.train_loop(epochs=epochs)
        # Validation can be part of train_loop or called separately:
        # if cfg.task.get("run_final_validation", True):
        #     model.validation_loop() # Run validation at the end of all epochs

    elif cfg.task.task == "validation":
        model = ValidateModel(cfg)
        model.to(model.device)
        model.validation_loop()

    elif cfg.task.task == "inference":
        model = InferenceModel(cfg)
        model.to(model.device)
        # Use save_path from setup for storing inference results
        # display_stream can be a new config option, defaulting to False
        display_stream = getattr(cfg.task, "display_stream", False)
        model.inference_loop(save_dir=str(save_path), display_stream=display_stream)

    else:
        print(f"Unknown task: {cfg.task.task}")


if __name__ == "__main__":
    main()
